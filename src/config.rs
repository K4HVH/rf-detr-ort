use std::path::PathBuf;

/// Execution provider / hardware target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// Pick the best available: TensorRT → CUDA → CPU.
    Auto,
    Cpu,
    Cuda,
    TensorRt,
}

/// Floating-point precision for GPU inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp16,
}

/// Full configuration for [`Engine`](crate::Engine).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,

    /// Requested execution provider.
    pub device: Device,

    /// Floating-point precision (GPU only; CPU always uses FP32).
    pub precision: Precision,

    /// GPU device index.
    pub device_id: i32,

    /// Enable CUDA graph capture/replay (CUDA EP only). Big latency win on
    /// repeated inference with the same input shape.
    pub enable_cuda_graph: bool,

    /// TensorRT engine builder workspace limit in bytes (default 4 GiB).
    pub trt_workspace_bytes: usize,

    /// Directory to store TensorRT engine and timing caches.
    pub trt_cache_dir: PathBuf,

    /// Maximum batch size this engine instance supports.
    pub max_batch_size: usize,

    /// Intra-op parallelism threads (0 = ORT default).
    pub intra_op_threads: usize,

    /// Inter-op parallelism threads (0 = ORT default).
    pub inter_op_threads: usize,

    /// ImageNet-default channel means used during normalisation.
    pub mean: [f32; 3],

    /// ImageNet-default channel stds used during normalisation.
    pub std: [f32; 3],

    /// If the requested provider cannot be initialised, automatically fall back
    /// through the hierarchy (TRT → CUDA → CPU).
    pub auto_fallback: bool,

    /// Set TensorRT builder optimization level to 5 (maximum).
    pub trt_max_optimization: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            device: Device::Auto,
            precision: Precision::Fp16,
            device_id: 0,
            enable_cuda_graph: false,
            trt_workspace_bytes: 4 * 1024 * 1024 * 1024,
            trt_cache_dir: PathBuf::from("trt_cache"),
            max_batch_size: 1,
            intra_op_threads: 0,
            inter_op_threads: 0,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            auto_fallback: true,
            trt_max_optimization: true,
        }
    }
}
