use crate::engine::{Checkpoint, Prepared, Settings};
use crate::json::Json;
use crate::router::{self, Hints, Policy};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Instant;

pub const PRESETS: &str = include_str!("presets.json");
pub const UNPROCESSABLE: u16 = 422;
pub const UNAVAILABLE: u16 = 503;

pub type FineTuned = Arc<RwLock<HashMap<String, PathBuf>>>;

#[derive(Clone, Copy, PartialEq)]
pub enum Shape {
    Native,
    Jev,
}

#[derive(Debug)]
pub struct Failure {
    pub status: u16,
    pub message: String,
}

impl Failure {
    fn invalid(e: impl std::fmt::Display) -> Self {
        Self {
            status: UNPROCESSABLE,
            message: e.to_string(),
        }
    }
    fn unavailable(e: impl std::fmt::Display) -> Self {
        Self {
            status: UNAVAILABLE,
            message: e.to_string(),
        }
    }
    pub fn json(&self) -> Json {
        let kind = if self.status == UNPROCESSABLE {
            "invalid_request_error"
        } else {
            "server_error"
        };
        Json::obj([(
            "error",
            Json::obj([
                ("type", kind.into()),
                ("message", self.message.clone().into()),
            ]),
        )])
    }
}

struct Job {
    checkpoint: Arc<Checkpoint>,
    prepared: Prepared,
    reply: Sender<Result<Json, String>>,
}

#[derive(Default)]
struct Queue {
    jobs: Mutex<VecDeque<Job>>,
    ready: Condvar,
    closed: AtomicBool,
}

struct Pending {
    started: Instant,
    prepare_ms: f64,
    served_as: String,
    routing: Json,
    answer: Receiver<Result<Json, String>>,
}

pub struct Service {
    models_dir: PathBuf,
    settings: Settings,
    policy: Policy,
    max_batch_tokens: usize,
    checkpoints: RwLock<HashMap<String, Arc<Checkpoint>>>,
    fine_tuned: FineTuned,
    loading: Mutex<()>,
    queue: Arc<Queue>,
    presets: Json,
}

impl Service {
    pub fn start(
        models_dir: PathBuf,
        settings: Settings,
        policy: Policy,
        max_batch_tokens: usize,
        fine_tuned: FineTuned,
    ) -> Arc<Self> {
        let queue = Arc::new(Queue::default());
        let workers = settings.devices.len();
        for replica in 0..workers {
            let queue = queue.clone();
            std::thread::Builder::new()
                .name(format!("laya-worker-{replica}"))
                .spawn(move || worker(&queue, replica, workers, max_batch_tokens))
                .expect("spawning a laya worker thread");
        }
        Arc::new(Self {
            models_dir,
            settings,
            policy,
            max_batch_tokens,
            checkpoints: RwLock::new(HashMap::new()),
            fine_tuned,
            loading: Mutex::new(()),
            queue,
            presets: Json::parse(PRESETS).expect("embedded presets are valid JSON"),
        })
    }

    pub fn available(&self) -> Vec<&'static str> {
        router::MODELS
            .iter()
            .copied()
            .filter(|m| {
                self.models_dir
                    .join(router::directory(m))
                    .join("model.safetensors")
                    .exists()
            })
            .collect()
    }

    pub fn checkpoint(&self, model: &'static str) -> Result<Arc<Checkpoint>, Failure> {
        self.checkpoint_at(model, &self.models_dir.join(router::directory(model)))
    }

    fn checkpoint_at(&self, model: &str, dir: &Path) -> Result<Arc<Checkpoint>, Failure> {
        if let Some(hit) = self.checkpoints.read().unwrap().get(model) {
            return Ok(hit.clone());
        }
        let _one_at_a_time = self.loading.lock().unwrap();
        if let Some(hit) = self.checkpoints.read().unwrap().get(model) {
            return Ok(hit.clone());
        }
        let started = Instant::now();
        let loaded = Arc::new(Checkpoint::load(dir, &self.settings).map_err(|e| {
            Failure::unavailable(format!("loading {model} from {}: {e:#}", dir.display()))
        })?);
        let state = Json::obj([("message", "warm-up request: the export job failed again after the nightly deploy and the customer is asking for a refund ".repeat(3).into())]);
        if let Some(questions) = self.presets.get("triage").and_then(Json::as_obj) {
            let warm = loaded
                .prepare(&state, questions)
                .map_err(Failure::unavailable)?;
            for replica in 0..loaded.replicas() {
                loaded
                    .run(replica, &[&warm])
                    .map_err(|e| Failure::unavailable(format!("warming {model}: {e:#}")))?;
            }
        }
        eprintln!(
            "laya: loaded and warmed {model} x{} from {} in {:.2}s",
            loaded.replicas(),
            dir.display(),
            started.elapsed().as_secs_f64()
        );
        self.checkpoints
            .write()
            .unwrap()
            .insert(model.to_string(), loaded.clone());
        Ok(loaded)
    }

    pub fn describe(&self) -> Json {
        let names = |list: Vec<&str>| Json::Arr(list.into_iter().map(Json::from).collect());
        Json::obj([
            ("status", "ok".into()),
            (
                "loaded",
                names(
                    self.checkpoints
                        .read()
                        .unwrap()
                        .keys()
                        .map(String::as_str)
                        .collect(),
                ),
            ),
            (
                "fine_tuned",
                names(
                    self.fine_tuned
                        .read()
                        .unwrap()
                        .keys()
                        .map(String::as_str)
                        .collect(),
                ),
            ),
            ("available", names(self.available())),
            ("workers", self.settings.devices.len().into()),
            ("max_batch_tokens", self.max_batch_tokens.into()),
            (
                "presets",
                names(
                    self.presets
                        .as_obj()
                        .unwrap_or(&[])
                        .iter()
                        .map(|(k, _)| k.as_str())
                        .collect(),
                ),
            ),
        ])
    }

    pub fn presets(&self) -> &Json {
        &self.presets
    }

    fn questions<'a>(&'a self, request: &'a Json) -> Result<&'a [(String, Json)], Failure> {
        if let Some(q) = request.get("questions") {
            return q
                .as_obj()
                .ok_or_else(|| Failure::invalid("questions must be a map of id -> question"));
        }
        let name = request
            .get("preset")
            .and_then(Json::as_str)
            .ok_or_else(|| Failure::invalid("give questions, or a preset name"))?;
        self.presets
            .get(name)
            .and_then(Json::as_obj)
            .ok_or_else(|| Failure::invalid(format!("unknown preset {name:?}")))
    }

    fn submit(&self, request: &Json) -> Result<Pending, Failure> {
        let state = request
            .get("state")
            .ok_or_else(|| Failure::invalid("request has no state"))?;
        let questions = self.questions(request)?;
        let hint = |key: &str| request.get(key).and_then(Json::as_str);
        let model = hint("model").filter(|m| *m != "auto" && !m.starts_with("jev"));
        let fine_tuned = model.and_then(|m| {
            self.fine_tuned
                .read()
                .unwrap()
                .get(m)
                .cloned()
                .map(|dir| (m.to_string(), dir))
        });
        let (served_as, checkpoint, routing) = match fine_tuned {
            Some((name, dir)) => {
                let routing = Json::obj([
                    ("model", name.clone().into()),
                    ("reason", "explicit fine-tuned checkpoint".into()),
                ]);
                (name.clone(), self.checkpoint_at(&name, &dir)?, routing)
            }
            None => {
                let hints = Hints {
                    model,
                    task: hint("task"),
                    lang: hint("lang"),
                };
                let (model, routing) = router::route(state, questions, &hints, &self.policy)
                    .map_err(Failure::invalid)?;
                (
                    router::directory(model).to_string(),
                    self.checkpoint(model)?,
                    routing,
                )
            }
        };
        let started = Instant::now();
        let prepared = checkpoint
            .prepare(state, questions)
            .map_err(Failure::invalid)?;
        let prepare_ms = ms(started.elapsed());
        let (reply, answer) = channel();
        self.queue.jobs.lock().unwrap().push_back(Job {
            checkpoint,
            prepared,
            reply,
        });
        self.queue.ready.notify_one();
        Ok(Pending {
            started,
            prepare_ms,
            served_as,
            routing,
            answer,
        })
    }

    fn finish(&self, pending: Pending, shape: Shape) -> Result<Json, Failure> {
        let mut result = pending
            .answer
            .recv()
            .map_err(|_| Failure::unavailable("a laya worker dropped the request"))?
            .map_err(Failure::unavailable)?;
        result.set("model", pending.served_as.clone().into());
        match shape {
            Shape::Native => {
                result.set("routing", pending.routing);
                result.set(
                    "timing",
                    Json::obj([
                        ("prepare_ms", pending.prepare_ms.into()),
                        ("total_ms", ms(pending.started.elapsed()).into()),
                    ]),
                );
            }
            Shape::Jev => {
                if let Some(Json::Obj(answers)) = result.get("answers").cloned() {
                    let strict = answers.into_iter().map(|(id, mut answer)| {
                        answer.remove("action");
                        if answer.get("type").and_then(Json::as_str) == Some("noul") {
                            answer.remove("confidence");
                        }
                        (id, answer)
                    });
                    result.set("answers", Json::Obj(strict.collect()));
                }
            }
        }
        Ok(result)
    }

    pub fn predict(&self, request: &Json, shape: Shape) -> Result<Json, Failure> {
        let pending = self.submit(request)?;
        self.finish(pending, shape)
    }

    pub fn predict_many(&self, requests: &[Json], shape: Shape) -> Vec<Result<Json, Failure>> {
        let pending: Vec<Result<Pending, Failure>> =
            requests.iter().map(|r| self.submit(r)).collect();
        pending
            .into_iter()
            .map(|p| self.finish(p?, shape))
            .collect()
    }
}

fn ms(d: std::time::Duration) -> f64 {
    (d.as_secs_f64() * 1e5).round() / 100.0
}

impl Drop for Service {
    fn drop(&mut self) {
        self.queue.closed.store(true, Ordering::SeqCst);
        self.queue.ready.notify_all();
    }
}

fn take_batch(queue: &Queue, workers: usize, max_batch_tokens: usize) -> Vec<Job> {
    let mut jobs = queue.jobs.lock().unwrap();
    while jobs.is_empty() {
        if queue.closed.load(Ordering::SeqCst) {
            return Vec::new();
        }
        jobs = queue.ready.wait(jobs).unwrap();
    }
    let share = jobs.len().div_ceil(workers);
    let mut batch = vec![jobs.pop_front().expect("the queue is not empty")];
    let mut tokens = batch[0].prepared.tokens();
    while batch.len() < share {
        match jobs.front() {
            Some(next)
                if Arc::ptr_eq(&next.checkpoint, &batch[0].checkpoint)
                    && tokens + next.prepared.tokens() <= max_batch_tokens =>
            {
                tokens += next.prepared.tokens();
                batch.push(jobs.pop_front().expect("front was just seen"));
            }
            _ => break,
        }
    }
    if !jobs.is_empty() {
        queue.ready.notify_one();
    }
    batch
}

fn worker(queue: &Queue, replica: usize, workers: usize, max_batch_tokens: usize) {
    loop {
        let jobs = take_batch(queue, workers, max_batch_tokens);
        if jobs.is_empty() {
            return;
        }
        let prepared: Vec<&Prepared> = jobs.iter().map(|j| &j.prepared).collect();
        match jobs[0].checkpoint.run(replica, &prepared) {
            Ok(results) => {
                for (job, result) in jobs.iter().zip(results) {
                    let _ = job.reply.send(Ok(result));
                }
            }
            Err(e) => {
                for job in &jobs {
                    let _ = job.reply.send(Err(format!("{e:#}")));
                }
            }
        }
    }
}
