/// Minimal demo: load an image, run RF-DETR inference, print detections.
///
/// Usage:
///   cargo run --example detect_image --release -- \
///       --model /path/to/inference_model.onnx \
///       --image /path/to/image.jpg \
///       [--conf 0.5]
use rfdetr_ort::{Device, Engine, EngineConfig, Precision};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
    };

    let model_path = get("--model").ok_or_else(|| anyhow::anyhow!("--model <path> is required"))?;
    let image_path = get("--image").ok_or_else(|| anyhow::anyhow!("--image <path> is required"))?;
    let conf: f32 = get("--conf").and_then(|s| s.parse().ok()).unwrap_or(0.5);

    let config = EngineConfig {
        model_path: model_path.into(),
        device: Device::Auto,
        precision: Precision::Fp16,
        ..Default::default()
    };

    println!("Loading model…");
    let mut engine = Engine::new(config)?;
    println!(
        "Engine ready  [{:?} / {:?}]",
        engine.active_device(),
        engine.active_precision()
    );

    let img = image::open(&image_path)
        .map_err(|e| anyhow::anyhow!("failed to open {image_path}: {e}"))?;
    println!("Image loaded  [{}×{}]", img.width(), img.height());

    // TensorRT JIT-compiles CUDA kernels on the first few calls; warmup
    // ensures the timed inference below reflects steady-state performance.
    let warmup = 10;
    print!("Warming up ({warmup} iterations)… ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    for _ in 0..warmup {
        engine.infer(&img, conf)?;
    }
    println!("done");

    let detections = engine.infer(&img, conf)?;
    let t = engine.last_timings();

    println!(
        "Inference done  [pre {:.2}ms  infer {:.2}ms  post {:.2}ms]",
        t.preprocess_ms, t.inference_ms, t.postprocess_ms
    );
    println!("Detections ({}):", detections.len());
    for det in &detections {
        println!(
            "  class {:>2}  conf {:.1}%  box [{}, {}, {}×{}]",
            det.class_id,
            det.confidence * 100.0,
            det.x,
            det.y,
            det.width,
            det.height,
        );
    }

    // engine is still alive here — process::exit skips all Rust drop glue,
    // so the ORT/TRT session destructor never runs (avoids a TRT teardown crash).
    std::process::exit(0);
}
