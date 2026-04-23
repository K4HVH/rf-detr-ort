//! # rfdetr-ort
//!
//! High-performance RF-DETR object-detection inference backed by ONNX Runtime.
//!
//! ## Quick start
//! ```no_run
//! use rfdetr_ort::{Engine, EngineConfig, Device, Precision};
//! use image::DynamicImage;
//!
//! let config = EngineConfig {
//!     model_path: "models/inference_model.onnx".into(),
//!     device: Device::Auto,
//!     precision: Precision::Fp16,
//!     ..Default::default()
//! };
//!
//! let mut engine = Engine::new(config).expect("failed to build engine");
//! let img = image::open("test2.png").expect("failed to open image");
//! let detections = engine.infer(&img, 0.5).expect("inference failed");
//!
//! for det in &detections {
//!     println!(
//!         "class {} ({:.1}%) @ {},{}  {}×{}",
//!         det.class_id,
//!         det.confidence * 100.0,
//!         det.x, det.y, det.width, det.height
//!     );
//! }
//! ```

pub mod config;
pub mod engine;
pub mod error;
pub mod pipeline;
pub mod preprocess;

mod postprocess;

pub use config::{Device, EngineConfig, Precision};
pub use engine::{Engine, ModelInfo, Timings};
pub use error::{Error, Result};
pub use pipeline::{FrameResult, PipelinedEngine};

/// A single object detection result.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Left edge of the bounding box in pixels (original image coordinates).
    pub x: i32,
    /// Top edge of the bounding box in pixels (original image coordinates).
    pub y: i32,
    /// Width of the bounding box in pixels.
    pub width: i32,
    /// Height of the bounding box in pixels.
    pub height: i32,
    /// Zero-based class index.
    pub class_id: usize,
    /// Confidence score in [0, 1].
    pub confidence: f32,
}
