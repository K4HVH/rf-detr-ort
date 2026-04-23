use std::{
    sync::Arc,
    thread::{self, JoinHandle},
};

use crossbeam_channel::{Receiver, Sender, bounded};
use image::DynamicImage;

use crate::{
    Detection, ModelInfo, Timings,
    config::EngineConfig,
    engine::Engine,
    error::{Error, Result},
};

// ─── FrameResult ───────────────────────────────────────────────────────────

/// Result of a single pipeline frame.
pub struct FrameResult {
    pub frame_id: u64,
    pub detections: Vec<Detection>,
    pub timings: Timings,
}

// ─── Internal job types ─────────────────────────────────────────────────────

struct Job {
    frame_id: u64,
    image: DynamicImage,
    conf_threshold: f32,
}

// ─── PipelinedEngine ───────────────────────────────────────────────────────

/// Throughput-oriented wrapper around [`Engine`] that overlaps preprocessing,
/// inference, and the caller's own postprocessing across frames.
///
/// Optimal for video / camera streams.  Single-frame latency is the same as
/// [`Engine::infer`]; throughput increases because the worker can be processing
/// the next frame while the caller is consuming the current result.
pub struct PipelinedEngine {
    tx: Sender<Option<Job>>,
    rx: Receiver<FrameResult>,
    model_info: Arc<ModelInfo>,
    _worker: JoinHandle<()>,
}

unsafe impl Send for PipelinedEngine {}

impl PipelinedEngine {
    /// Create a new pipelined engine.
    ///
    /// `queue_depth` controls the maximum number of frames queued for the
    /// worker. A value of 2–4 is typically optimal; too large wastes memory
    /// and increases latency.
    pub fn new(config: EngineConfig, queue_depth: usize) -> Result<Self> {
        let engine = Engine::new(config)?;
        let model_info = Arc::new(engine.model_info().clone());
        let mi_ret = Arc::clone(&model_info);

        let depth = queue_depth.max(1);
        let (tx, in_rx) = bounded::<Option<Job>>(depth);
        let (out_tx, rx) = bounded::<FrameResult>(depth);

        let worker = thread::spawn(move || {
            let mut eng = engine;
            while let Ok(Some(job)) = in_rx.recv() {
                let res = match eng.infer(&job.image, job.conf_threshold) {
                    Ok(dets) => FrameResult {
                        frame_id: job.frame_id,
                        detections: dets,
                        timings: eng.last_timings(),
                    },
                    Err(_) => FrameResult {
                        frame_id: job.frame_id,
                        detections: vec![],
                        timings: Timings::default(),
                    },
                };
                // If the receiver is gone, stop the worker.
                if out_tx.send(res).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            tx,
            rx,
            model_info: mi_ret,
            _worker: worker,
        })
    }

    /// Submit a frame for inference.
    ///
    /// Blocks if the internal queue is full.  Returns `Err` if the pipeline
    /// has been stopped.
    pub fn submit(
        &self,
        frame_id: u64,
        image: DynamicImage,
        conf_threshold: f32,
    ) -> Result<()> {
        self.tx
            .send(Some(Job { frame_id, image, conf_threshold }))
            .map_err(|_| Error::PipelineStopped)
    }

    /// Block until the next result is available.
    ///
    /// Returns `None` when the pipeline has been stopped and all pending
    /// results have been consumed.
    pub fn next_result(&self) -> Option<FrameResult> {
        self.rx.recv().ok()
    }

    /// Signal that no more frames will be submitted and drain the queue.
    pub fn stop(&self) {
        let _ = self.tx.send(None);
    }

    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }
}
