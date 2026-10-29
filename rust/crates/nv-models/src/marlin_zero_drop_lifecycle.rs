use crate::gemma4_e4b::{
    marlin_zero_launch, marlin_zero_snapshot_host_entries_uploaded_device_list_and_count,
    marlin_zero_upload, MarlinLinear,
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, DevicePtrMut};
use half::bf16;

const SYNTHETIC_OUT_256_IN_256_GROUPS_2_IS_THE_SMALLEST_LN_SUPPORTED_SHAPE: (usize, usize, usize) =
    (256, 256, 2);
const SAME_SIZE_ALLOC_ATTEMPTS_BEFORE_DECLARING_THE_REUSE_ARM_VACUOUS: usize = 64;

fn synthetic_marlin(
    device: &Device,
    out: usize,
    inp: usize,
    ngroups: usize,
) -> Result<MarlinLinear> {
    let pbytes = vec![0u8; out * (inp / 8) * 4];
    let scale = Tensor::ones((out, ngroups), DType::BF16, device)?;
    MarlinLinear::from_raw(&pbytes, scale, out, inp, device)
}

#[test]
fn dropping_the_first_model_after_an_m1_use_must_pull_its_persist_row_out_of_the_zero_launch_list_before_the_allocator_reuses_the_address(
) -> Result<()> {
    let device = Device::new_cuda(0).expect(
        "this is the marlin drop-side use-after-free gate and must not report success without \
         the CUDA card",
    );
    let Device::Cuda(dev) = device.clone() else {
        unreachable!()
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let (out, inp, ngroups) = SYNTHETIC_OUT_256_IN_256_GROUPS_2_IS_THE_SMALLEST_LN_SUPPORTED_SHAPE;

    let model_a = synthetic_marlin(&device, out, inp, ngroups)?;
    let x = Tensor::zeros((1usize, inp), DType::BF16, &device)?;
    let _y = model_a.forward(&x)?;
    marlin_zero_upload(&stream)?;
    marlin_zero_launch(&stream)?;
    stream.synchronize()?;

    let (entries, dev_list, count) =
        marlin_zero_snapshot_host_entries_uploaded_device_list_and_count(&stream)?;
    assert!(
        count >= 1,
        "m==1 use must have populated the zero-launch list"
    );
    let ptr_a = entries
        .chunks(2)
        .rev()
        .find(|c| c[1] == out as u64)
        .expect("model A's forward must have registered its persist row")[0];
    assert!(
        dev_list.chunks(2).any(|c| c[0] == ptr_a),
        "the uploaded device list must carry model A's persist pointer while A is alive"
    );

    drop(model_a);

    let (entries_after, dev_list_after, _count_after) =
        marlin_zero_snapshot_host_entries_uploaded_device_list_and_count(&stream)?;
    assert!(
        !entries_after.chunks(2).any(|c| c[0] == ptr_a),
        "host registry still holds the dropped model's persist pointer: the next zero launch \
         scribbles whatever the allocator hands out at that address"
    );
    assert!(
        !dev_list_after.chunks(2).any(|c| c[0] == ptr_a),
        "the uploaded device list still holds the dropped model's persist pointer: a captured \
         decode graph replays the zero launch against the freed address"
    );

    let freed_lo = ptr_a;
    let freed_hi = ptr_a + 2 * out as u64;
    let mut held: Vec<CudaSlice<bf16>> = Vec::new();
    let mut reused: Option<(CudaSlice<bf16>, usize, usize)> = None;
    'attempts: for _ in 0..SAME_SIZE_ALLOC_ATTEMPTS_BEFORE_DECLARING_THE_REUSE_ARM_VACUOUS {
        for elems in [out, 512, 65536] {
            let mut b: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(elems) }?;
            let p = {
                let (p, _g) = b.device_ptr_mut(&stream);
                p
            };
            let (lo, hi) = (p, p + 2 * elems as u64);
            if lo < freed_hi && freed_lo < hi {
                let w_lo = (freed_lo.max(lo) - p) as usize / 2;
                let w_hi = (freed_hi.min(hi) - p) as usize / 2;
                reused = Some((b, w_lo, w_hi));
                break 'attempts;
            }
            held.push(b);
        }
    }
    match reused {
        Some((mut model_b_tensor, w_lo, w_hi)) => {
            let elems = model_b_tensor.len();
            let pattern = vec![bf16::from_f32(1.0); elems];
            stream.memcpy_htod(&pattern, &mut model_b_tensor)?;
            marlin_zero_upload(&stream)?;
            marlin_zero_launch(&stream)?;
            stream.synchronize()?;
            let back: Vec<bf16> = stream.clone_dtoh(&model_b_tensor)?;
            let zeroed = back[w_lo..w_hi].iter().filter(|v| v.to_bits() == 0).count();
            assert_eq!(
                zeroed,
                0,
                "the per-step zero launch scribbled {zeroed}/{} elements of the second model's \
                 allocation where it overlaps the first model's freed persist row",
                w_hi - w_lo
            );
            eprintln!(
                "[marlin-drop] scribble arm ran: second-model allocation overlapped the freed \
                 persist row [{w_lo}..{w_hi}) and stayed intact across a zero launch"
            );
        }
        None => eprintln!(
            "[marlin-drop] scribble arm unreachable this run: the allocator never re-handed the \
             freed persist range; the registry and device-list assertions above are the gate"
        ),
    }
    drop(held);
    Ok(())
}
