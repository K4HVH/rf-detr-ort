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
pub(crate) mod postprocess;
pub(crate) mod preprocess;

pub use config::{Device, EngineConfig, Precision};
pub use engine::{Engine, ModelInfo, Timings};
pub use error::{Error, Result};
pub use postprocess::Detection;
