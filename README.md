# rf-detr-ort

[![CI](https://github.com/K4HVH/rf-detr-ort/actions/workflows/ci.yml/badge.svg)](https://github.com/K4HVH/rf-detr-ort/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/rf-detr-ort.svg)](https://crates.io/crates/rf-detr-ort)
[![docs.rs](https://docs.rs/rf-detr-ort/badge.svg)](https://docs.rs/rf-detr-ort)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

[RF-DETR](https://github.com/roboflow/rf-detr) object-detection inference in Rust via [ONNX Runtime](https://onnxruntime.ai/). Supports TensorRT, CUDA, and CPU execution providers with automatic fallback.

## Platform support

| Platform | CPU | CUDA | TensorRT |
|----------|-----|------|----------|
| Linux x86_64 | ✓ | ✓ | ✓ |
| Windows x86_64 | ✓ | ✓ | ✓ |
| macOS (x86_64 / Apple Silicon) | ✓ | — | — |

## Performance

Measured on Intel Core Ultra 9 185H + NVIDIA RTX 3000 Ada (sm_89), RF-DETR 384×384 input, batch 1. TRT/CUDA: 1000 iterations after 100 warmup. CPU: 100 iterations after 10 warmup.

| EP | Precision | Inference | Total | FPS |
|----|-----------|-----------|-------|-----|
| TensorRT | FP16 | 1.99 ms | 2.08 ms | **482** |
| CUDA | FP16 | 6.11 ms | 6.20 ms | 161 |
| CPU | FP32 | 74.18 ms | 74.36 ms | 13 |

Pre-processing (~0.9 ms) is included in the total. Post-processing is negligible (<0.01 ms).

## Setup

### 1. Download ONNX Runtime

Download the prebuilt ORT package from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases). Use a GPU build for CUDA/TRT, or the standard package for CPU-only.

| Platform | Example package (1.24.4) |
|----------|--------------------------|
| Linux x86_64 (GPU) | `onnxruntime-linux-x64-gpu-1.24.4.tgz` |
| Windows x86_64 (GPU) | `onnxruntime-win-x64-gpu-1.24.4.zip` |
| macOS | `onnxruntime-osx-universal2-1.24.4.tgz` |

### 2. Point the build at ORT

Set `ORT_DYLIB_PATH` to the full path of the shared library before building:

```sh
# Linux
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
# macOS
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.dylib
# Windows (PowerShell)
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime\lib\onnxruntime.dll"
```

On Linux/macOS you can bake this in via `.cargo/config.toml` with `ORT_LIB_LOCATION` + an rpath rustflag. On Windows, add the ORT `lib` directory to `PATH`.

### 3. Get an RF-DETR ONNX model

Export from the [RF-DETR repository](https://github.com/roboflow/rf-detr). The model expects inputs `[batch, 3, H, W]` (float32, ImageNet-normalized) and outputs boxes `[batch, queries, 4]` and logits `[batch, queries, classes]`.

## Usage

```toml
[dependencies]
rf-detr-ort = "0.1"                                                               # TensorRT + CUDA
rf-detr-ort = { version = "0.1", default-features = false }                       # CPU only
rf-detr-ort = { version = "0.1", default-features = false, features = ["cuda"] }  # CUDA, no TRT
```

```rust
use rf_detr_ort::{Engine, EngineConfig, Device, Precision};

let mut engine = Engine::new(EngineConfig {
    model_path: "model.onnx".into(),
    device: Device::Auto,       // tries TensorRT → CUDA → CPU
    precision: Precision::Fp16,
    ..Default::default()
})?;

let img = image::open("photo.jpg")?;
for det in engine.infer(&img, 0.5)? {
    println!("class {} ({:.1}%) @ {},{} {}×{}",
        det.class_id, det.confidence * 100.0,
        det.x, det.y, det.width, det.height);
}
```

Batch inference (`infer_batch`) and raw BGR frames for video pipelines (`infer_frame`) are also supported — see [`examples/detect_image.rs`](examples/detect_image.rs) and the [docs](https://docs.rs/rf-detr-ort).

## Config reference

| Field | Default | Notes |
|-------|---------|-------|
| `model_path` | — | Path to `.onnx` file (required) |
| `device` | `Auto` | `Auto` / `Cpu` / `Cuda` / `TensorRt` |
| `precision` | `Fp16` | `Fp16` or `Fp32` |
| `trt_cache_dir` | `"trt_cache"` | Avoids recompilation on restart |
| `max_batch_size` | `1` | Set >1 to enable `infer_batch` |
| `enable_cuda_graph` | `false` | Fixed-shape CUDA graph capture |
| `trt_workspace_bytes` | 4 GiB | TRT workspace per layer |
| `auto_fallback` | `true` | Fall back to weaker EP on build failure |

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda` | yes | CUDA execution provider |
| `tensorrt` | yes | TensorRT EP (implies `cuda`) |
| `cli` | no | `rfdetr` binary; requires OpenCV + Clang ([setup](https://github.com/twistedfall/opencv-rust#getting-started)) |

## CLI

```sh
cargo build --release --features cli

rfdetr --mode image     --model model.onnx --input photo.jpg --output out.jpg
rfdetr --mode benchmark --model model.onnx --input photo.jpg --device tensorrt --precision fp16
rfdetr --mode video     --model model.onnx --input video.mp4 --output out.mp4
```

`benchmark` mode outputs mean / stdev / p50 / p90 / p99 / min / max per pipeline stage plus overall FPS.

## Notes

- **TRT engine cache** — the first TRT run compiles and caches the engine; subsequent runs skip compilation.
- **GPU teardown** — on GPU builds the process calls `std::process::exit(0)` after completion to avoid a known ORT/TRT destructor segfault. This is safe and intentional.
- **`Engine` is `Send`** — the engine can be moved across threads; all `infer*` methods take `&mut self`.

## License

[GPL-3.0](LICENSE)
