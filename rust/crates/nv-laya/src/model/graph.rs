use super::{graph_token_bucket, DecisionModel, Layout, VarlenRows};
use anyhow::{ensure, Result};
use candle_core::{DType, Tensor};
use nv_kernels::graph::{
    trim_device_graph_mempool_because_graph_destroy_keeps_the_reserved_pages as trim_graph_mempool,
    CudaGraphRunner,
};
use nv_quant::capture_exclusion::{no_capture_in_flight, NoCaptureInFlight};
use nv_quant::graph_teardown::GraphTeardown;
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};

pub const LIVE_GRAPHS_A_REPLICA_KEEPS_BEFORE_IT_DESTROYS_THE_LEAST_RECENTLY_REPLAYED: usize = 32;
pub const UNPROVEN_SHAPES_A_REPLICA_COUNTS_BEFORE_IT_FORGETS_THE_LEAST_RECENTLY_REQUESTED: usize =
    256;

pub const REPLICAS_SHARING_A_CUDA_STREAM_TAKE_TURNS_BECAUSE_ANOTHER_THREADS_WORK_JOINS_A_CAPTURE_IN_FLIGHT:
    &str = "every checkpoint's replica i runs on the same candle device, so one CUDA stream, and \
     the service loads and warms a new checkpoint from the request thread while worker i may be \
     capturing on that stream. Work another thread enqueues onto a capturing stream returns \
     success and is silently recorded into that thread's graph -- weight uploads from pageable \
     host memory included -- and a synchronize of it fails with \
     CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and invalidates the capture. Loading, every forward \
     pass and the drop of a model therefore each hold a Turn on the Lane of their stream";

#[derive(Default)]
pub(super) struct Lane {
    busy: Mutex<bool>,
    freed: Condvar,
}

pub(super) struct Turn(Arc<Lane>);

impl Lane {
    pub(super) fn of(device: &candle_core::Device) -> Arc<Lane> {
        static LANES: Mutex<Vec<(usize, Weak<Lane>)>> = Mutex::new(Vec::new());
        let Ok(cuda) = device.as_cuda_device() else {
            return Arc::default();
        };
        let stream = cuda.cuda_stream().cu_stream() as usize;
        let mut lanes = LANES.lock().unwrap_or_else(PoisonError::into_inner);
        lanes.retain(|(_, lane)| lane.strong_count() > 0);
        if let Some(lane) = lanes
            .iter()
            .find(|(key, _)| *key == stream)
            .and_then(|(_, lane)| lane.upgrade())
        {
            return lane;
        }
        let lane = Arc::<Lane>::default();
        lanes.push((stream, Arc::downgrade(&lane)));
        lane
    }

    pub(super) fn take_turn(self: &Arc<Self>) -> Turn {
        let mut busy = self.busy.lock().unwrap_or_else(PoisonError::into_inner);
        while *busy {
            busy = self
                .freed
                .wait(busy)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *busy = true;
        Turn(self.clone())
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap_or_else(PoisonError::into_inner) = false;
        self.0.freed.notify_one();
    }
}

pub(super) struct Teardown {
    _turn: Turn,
    _no_capture_in_flight: NoCaptureInFlight,
}

impl Teardown {
    pub(super) fn begin(lane: &Arc<Lane>) -> Self {
        let _ = REPLICAS_SHARING_A_CUDA_STREAM_TAKE_TURNS_BECAUSE_ANOTHER_THREADS_WORK_JOINS_A_CAPTURE_IN_FLIGHT;
        let _turn = lane.take_turn();
        Self {
            _turn,
            _no_capture_in_flight: no_capture_in_flight(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphStats {
    pub captures: usize,
    pub replays: usize,
    pub evictions: usize,
    pub live: usize,
    pub eager_because_unproven: usize,
}

struct Slot {
    shape: u64,
    inputs: Tensor,
    out: Tensor,
}

#[derive(Default)]
pub(super) struct Graphs {
    runner: Option<CudaGraphRunner>,
    least_recently_replayed_first: Vec<Slot>,
    unproven_least_recently_requested_first: Vec<(u64, usize)>,
    stats: GraphStats,
    gave_up: bool,
}

impl Drop for Graphs {
    fn drop(&mut self) {
        if let Some(runner) = &mut self.runner {
            let stream = runner.stream().clone();
            GraphTeardown::for_a_stream_this_engine_did_not_fork(&stream)
                .run(|| runner.invalidate());
        }
    }
}

impl DecisionModel {
    pub fn graph_stats(&self) -> GraphStats {
        let graphs = self.graphs.lock().unwrap();
        GraphStats {
            live: graphs.least_recently_replayed_first.len(),
            ..graphs.stats
        }
    }

    pub(super) fn graphed(
        &self,
        ids: &[u32],
        types: &[u32],
        rows: &VarlenRows,
        consumed: &[u32],
        n_markers: usize,
    ) -> Result<Option<Vec<f32>>> {
        let Some(bucket) = graph_token_bucket(ids.len())
            .filter(|bucket| self.cuda_graph && bucket - ids.len() <= self.max_len)
        else {
            return Ok(None);
        };
        let shape = bucket as u64
            | (rows.cu.len() as u64) << 16
            | (n_markers as u64) << 32
            | (self.full_final_head as u64) << 48;
        let mut graphs = self.graphs.lock().unwrap();
        if graphs.gave_up || !graphs.repeats(shape, self.requests_that_prove_a_shape_repeats) {
            return Ok(None);
        }
        match graphs.run(self, shape, bucket, ids, types, rows, consumed, n_markers) {
            Ok(flat) => Ok(Some(flat)),
            Err(e) => {
                eprintln!(
                    "laya: CUDA graphs are off for this replica and every pass runs eager: {e:#}"
                );
                graphs.gave_up = true;
                Ok(None)
            }
        }
    }
}

impl Graphs {
    fn repeats(&mut self, shape: u64, proof: usize) -> bool {
        if self
            .least_recently_replayed_first
            .iter()
            .any(|s| s.shape == shape)
        {
            return true;
        }
        let unproven = &mut self.unproven_least_recently_requested_first;
        let requests = 1 + unproven
            .iter()
            .position(|(s, _)| *s == shape)
            .map_or(0, |at| unproven.remove(at).1);
        if requests >= proof {
            return true;
        }
        unproven.push((shape, requests));
        if unproven.len()
            > UNPROVEN_SHAPES_A_REPLICA_COUNTS_BEFORE_IT_FORGETS_THE_LEAST_RECENTLY_REQUESTED
        {
            unproven.remove(0);
        }
        self.stats.eager_because_unproven += 1;
        false
    }

    fn run(
        &mut self,
        model: &DecisionModel,
        shape: u64,
        bucket: usize,
        ids: &[u32],
        types: &[u32],
        rows: &VarlenRows,
        consumed: &[u32],
        n_markers: usize,
    ) -> Result<Vec<f32>> {
        let unread = bucket - ids.len();
        let sequences = rows.cu.len();
        let pad = |real: &[u32], padding: &mut dyn Iterator<Item = u32>| {
            real.iter().copied().chain(padding).collect::<Vec<u32>>()
        };
        let packed = [
            pad(ids, &mut std::iter::repeat(model.pad_id).take(unread)),
            pad(types, &mut std::iter::repeat(0).take(unread)),
            pad(&rows.positions, &mut (0..unread as u32)),
            pad(&rows.cu, &mut std::iter::once(bucket as u32)),
            consumed.to_vec(),
        ]
        .concat();
        let fresh = Tensor::from_vec(
            packed,
            3 * bucket + sequences + 1 + consumed.len(),
            &model.device,
        )?;
        let slot = match self
            .least_recently_replayed_first
            .iter()
            .position(|s| s.shape == shape)
        {
            Some(at) => {
                let slot = self.least_recently_replayed_first.remove(at);
                slot.inputs.slice_set(&fresh, 0, 0)?;
                slot
            }
            None => Slot {
                shape,
                inputs: fresh,
                out: Tensor::zeros(
                    n_markers + (consumed.len() - n_markers) * model.cfg.hidden,
                    DType::F32,
                    &model.device,
                )?,
            },
        };
        let body = || -> Result<Tensor> {
            let part = |at: usize, len: usize| slot.inputs.narrow(0, at, len);
            let layout = Layout::Varlen {
                cu: part(3 * bucket, sequences + 1)?,
                max_len: model.max_len,
                positions: part(2 * bucket, bucket)?,
            };
            model.encode(
                &part(0, bucket)?,
                &part(bucket, bucket)?,
                &part(3 * bucket + sequences + 1, consumed.len())?,
                &layout,
                n_markers,
            )
        };

        let cuda = model.device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let runner = self
            .runner
            .get_or_insert_with(|| CudaGraphRunner::new(stream.clone()));
        let eager = if runner.has_cached_token(shape) {
            None
        } else {
            if stream.context().is_event_tracking() {
                stream.synchronize()?;
                unsafe { stream.context().disable_event_tracking() };
            }
            Some((
                cuda.enable_cuda_graph_htod_cache(),
                body()?.to_vec1::<f32>()?,
            ))
        };
        runner.run(shape, |_| Ok(slot.out.slice_set(&body()?, 0, 0)?))?;
        let eager = eager
            .map(|(_small_host_parameters_come_from_the_cache_the_eager_pass_filled, eager)| eager);
        let flat = slot.out.to_vec1::<f32>()?;
        match eager {
            Some(eager) => {
                ensure!(
                    eager.iter().map(|v| v.to_bits()).eq(flat.iter().map(|v| v.to_bits())),
                    "the graph captured for {bucket} tokens in {sequences} sequences must replay bitwise what the eager pass over the same padded buffers computed"
                );
                self.stats.captures += 1;
            }
            None => self.stats.replays += 1,
        }
        self.least_recently_replayed_first.push(slot);
        if self.least_recently_replayed_first.len()
            > LIVE_GRAPHS_A_REPLICA_KEEPS_BEFORE_IT_DESTROYS_THE_LEAST_RECENTLY_REPLAYED
        {
            runner.invalidate_token(self.least_recently_replayed_first.remove(0).shape);
            trim_graph_mempool(stream.context().ordinal());
            self.stats.evictions += 1;
        }
        Ok(flat)
    }
}
