use std::{
    io::Write,
    path::PathBuf,
    time::Instant,
};

use clap::{Parser, ValueEnum};
use crossbeam_channel::bounded;
use image::DynamicImage;
use opencv::{prelude::*, videoio};
use rfdetr_ort::{Detection, Device, Engine, EngineConfig, Precision};

// ─── CLI ───────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "rfdetr", about = "RF-DETR inference / benchmark tool")]
struct Cli {
    #[arg(long, default_value = "benchmark")]
    mode: Mode,

    #[arg(long)]
    model: PathBuf,

    /// Input image (required for `image` mode; used as the benchmark frame in
    /// `benchmark` mode).
    #[arg(long)]
    input: PathBuf,

    #[arg(long, default_value = "auto")]
    device: DeviceArg,

    #[arg(long, default_value = "fp32")]
    precision: PrecisionArg,

    /// Number of warmup iterations.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Number of measured iterations.
    #[arg(long, default_value_t = 200)]
    iters: usize,

    /// Confidence threshold.
    #[arg(long, default_value_t = 0.5)]
    conf: f32,

    /// TRT / engine cache directory.
    #[arg(long, default_value = "trt_cache")]
    cache_dir: PathBuf,

    /// GPU device id.
    #[arg(long, default_value_t = 0)]
    device_id: i32,

    /// Output file: annotated image (image mode) or annotated video (video mode).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Comma-separated class names, e.g. "body,head". Takes priority over --classes.
    #[arg(long)]
    classes_csv: Option<String>,

    /// File with one class name per line.
    #[arg(long)]
    classes: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Mode {
    Image,
    Benchmark,
    Video,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Tensorrt,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PrecisionArg {
    Fp32,
    Fp16,
}

// ─── main ──────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let device = match cli.device {
        DeviceArg::Auto => Device::Auto,
        DeviceArg::Cpu => Device::Cpu,
        DeviceArg::Cuda => Device::Cuda,
        DeviceArg::Tensorrt => Device::TensorRt,
    };
    let precision = match cli.precision {
        PrecisionArg::Fp32 => Precision::Fp32,
        PrecisionArg::Fp16 => Precision::Fp16,
    };

    let config = EngineConfig {
        model_path: cli.model.clone(),
        device,
        precision,
        device_id: cli.device_id,
        trt_cache_dir: cli.cache_dir,
        auto_fallback: true,
        ..Default::default()
    };

    println!("Building engine…");
    let t_build = Instant::now();
    let mut engine = Engine::new(config)?;
    println!("Engine ready in {:.1} s", t_build.elapsed().as_secs_f64());

    let mi = engine.model_info().clone();
    let active = engine.active_device().clone();
    let prec = engine.active_precision();
    let device_label = format!("{:?} / {:?}", active, prec);
    println!("Active device : {device_label}");
    println!("Input size    : {}×{}×{}", mi.input_width, mi.input_height, mi.input_channels);
    println!("Queries       : {}", mi.num_queries);
    println!("Classes       : {}", mi.num_classes);

    let classes = resolve_classes(cli.classes_csv.as_deref(), cli.classes.as_deref());

    match cli.mode {
        Mode::Video => {
            run_video(&mut engine, &cli.input, cli.conf, cli.output.as_deref(), &classes)?;
            std::process::exit(0); // avoid GPU ORT teardown segfault
        }
        _ => {
            let img = image::open(&cli.input)?;
            match cli.mode {
                Mode::Image => run_image(&mut engine, &img, cli.conf, &classes),
                Mode::Benchmark => run_benchmark(
                    &mut engine,
                    &img,
                    cli.conf,
                    cli.warmup,
                    cli.iters,
                    &device_label,
                ),
                Mode::Video => unreachable!(),
            }
        }
    }

    Ok(())
}

// ─── Image mode ────────────────────────────────────────────────────────────

fn run_image(
    engine: &mut Engine,
    img: &DynamicImage,
    conf: f32,
    classes: &[String],
) {
    let detections = engine.infer(img, conf).expect("inference failed");
    let t = engine.last_timings();
    println!(
        "\nTimings: pre={:.2} ms  inf={:.2} ms  post={:.2} ms  total={:.2} ms",
        t.preprocess_ms, t.inference_ms, t.postprocess_ms, t.total_ms()
    );
    println!("{} detection(s):", detections.len());

    for d in &detections {
        let label = classes.get(d.class_id).map(String::as_str).unwrap_or("?");
        println!(
            "  [{:3}] {:<20} conf={:.3}  x={} y={} w={} h={}",
            d.class_id, label, d.confidence, d.x, d.y, d.width, d.height
        );
    }
}

fn resolve_classes(csv: Option<&str>, file: Option<&std::path::Path>) -> Vec<String> {
    if let Some(s) = csv {
        return s.split(',').map(|p| p.trim().to_owned()).collect();
    }
    if let Some(p) = file {
        if let Ok(text) = std::fs::read_to_string(p) {
            return text.lines().map(|l| l.to_owned()).collect();
        }
    }
    vec![]
}

// ─── Benchmark mode ────────────────────────────────────────────────────────

struct Stats {
    mean: f64,
    stdev: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn compute(values: &mut Vec<f64>) -> Self {
        assert!(!values.is_empty());
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let stdev = variance.sqrt();

        let p = |pct: f64| -> f64 {
            let idx = ((pct / 100.0) * (n - 1) as f64).round() as usize;
            values[idx.min(n - 1)]
        };

        Self {
            mean,
            stdev,
            p50: p(50.0),
            p90: p(90.0),
            p99: p(99.0),
            min: values[0],
            max: values[n - 1],
        }
    }
}

fn run_benchmark(
    engine: &mut Engine,
    img: &DynamicImage,
    conf: f32,
    warmup: usize,
    iters: usize,
    device_label: &str,
) {
    println!("\nWarming up ({warmup} iterations)…");
    for _ in 0..warmup {
        let _ = engine.infer(img, conf).expect("warmup failed");
    }

    println!("Benchmarking ({iters} iterations)…");
    let mut pre_ms: Vec<f64> = Vec::with_capacity(iters);
    let mut inf_ms: Vec<f64> = Vec::with_capacity(iters);
    let mut post_ms: Vec<f64> = Vec::with_capacity(iters);
    let mut total_ms: Vec<f64> = Vec::with_capacity(iters);

    for _ in 0..iters {
        let _ = engine.infer(img, conf).expect("inference failed");
        let t = engine.last_timings();
        pre_ms.push(t.preprocess_ms);
        inf_ms.push(t.inference_ms);
        post_ms.push(t.postprocess_ms);
        total_ms.push(t.total_ms());
    }

    let pre = Stats::compute(&mut pre_ms);
    let inf = Stats::compute(&mut inf_ms);
    let post = Stats::compute(&mut post_ms);
    let tot = Stats::compute(&mut total_ms);
    let fps = 1000.0 / tot.mean;

    // ── Output (matches C++ benchmark format) ────────────────────────────
    println!("\n=== RF-DETR Benchmark ===");
    println!("Device     : {device_label}");
    println!("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  [ms]",
             "Stage", "mean", "stdev", "p50", "p90", "p99", "min", "max");
    print_row("preprocess", &pre);
    print_row("inference", &inf);
    print_row("postprocess", &post);
    print_row("total", &tot);
    println!("FPS (mean) : {fps:.1}");

    // ORT CUDA teardown can segfault during destructor cleanup.
    // All output has been printed; bypass CUDA teardown with a clean exit.
    std::process::exit(0);
}

fn print_row(name: &str, s: &Stats) {
    println!(
        "{:<14} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
        name, s.mean, s.stdev, s.p50, s.p90, s.p99, s.min, s.max
    );
}

// ─── Video mode ────────────────────────────────────────────────────────────

fn run_video(
    engine: &mut Engine,
    input: &std::path::Path,
    conf: f32,
    output: Option<&std::path::Path>,
    classes: &[String],
) -> anyhow::Result<()> {
    // ── Probe dimensions ─────────────────────────────────────────────────
    let (src_w, src_h, fps) = {
        let cap = videoio::VideoCapture::from_file(
            input.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF-8 path"))?,
            videoio::CAP_ANY,
        )?;
        anyhow::ensure!(cap.is_opened()?, "cannot open video: {}", input.display());
        let w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as u32;
        let h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as u32;
        let f = cap.get(videoio::CAP_PROP_FPS)?;
        (w, h, f)
    };
    println!("Video      : {src_w}\u{d7}{src_h} @ {fps:.2} fps");

    // ── Optional encoder thread ───────────────────────────────────────────
    // VideoWriter::write is software encoding and takes ~0.4ms per frame —
    // enough to drop throughput by ~20% if run on the inference thread.
    // Offloading to a dedicated thread lets the GPU run without stalling.
    // Channel depth=2 so the inference thread can run 2 frames ahead.
    type EncMsg = Option<(Vec<u8>, Vec<Detection>)>;
    let (enc_tx, enc_thread) = if let Some(out) = output {
        let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let mut writer = videoio::VideoWriter::new(
            out.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF-8 output path"))?,
            fourcc,
            fps,
            opencv::core::Size::new(src_w as i32, src_h as i32),
            true,
        )?;
        anyhow::ensure!(writer.is_opened()?, "cannot open video writer: {}", out.display());
        let (tx, rx) = bounded::<EncMsg>(2);
        let classes_enc = classes.to_vec();
        let t = std::thread::spawn(move || {
            while let Ok(Some((mut raw, dets))) = rx.recv() {
                for d in &dets { draw_box_bgr(&mut raw, src_w, src_h, &d, &classes_enc); }
                if let Ok(flat) = opencv::core::Mat::from_slice(&raw) {
                    if let Ok(bgr_mat) = flat.reshape(3, src_h as i32) {
                        let _ = writer.write(&bgr_mat);
                    }
                }
            }
            // writer dropped here → file flushed/closed
        });
        (Some(tx), Some(t))
    } else {
        (None, None)
    };

    let mut cap = videoio::VideoCapture::from_file(
        input.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF-8 path"))?,
        videoio::CAP_ANY,
    )?;;
    let mut bgr_frame = opencv::core::Mat::default();

    let mut frame_count = 0u64;
    let mut pre_ms  : Vec<f64> = Vec::new();
    let mut inf_ms  : Vec<f64> = Vec::new();
    let mut post_ms : Vec<f64> = Vec::new();
    let mut tot_ms  : Vec<f64> = Vec::new();
    let t_wall_start = Instant::now();

    // ── Inference loop ────────────────────────────────────────────────────
    // Single-threaded: engine.infer_frame() handles BGR→NCHW (with optional
    // resize) directly into the GPU-pinned buffer, then runs session.run().
    loop {
        match cap.read(&mut bgr_frame) { Ok(true) => {} _ => break }
        if bgr_frame.empty() { break; }

        let bgr_bytes = match bgr_frame.data_bytes() { Ok(b) => b, Err(_) => break };

        let t0 = Instant::now();
        let dets = engine.infer_frame(bgr_bytes, src_w, src_h, conf)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t = engine.last_timings();

        pre_ms.push(t.preprocess_ms);
        inf_ms.push(t.inference_ms);
        post_ms.push(t.postprocess_ms);
        tot_ms.push(elapsed_ms);
        frame_count += 1;

        if let Some(ref tx) = enc_tx {
            // Copy the raw frame bytes for the encoder — cap.read() will overwrite
            // bgr_frame on the next iteration, so we must capture now.
            let _ = tx.send(Some((bgr_bytes.to_vec(), dets)));
        }

        if frame_count % 50 == 0 {
            let cur_fps = frame_count as f64 / t_wall_start.elapsed().as_secs_f64();
            print!("\r  [{frame_count} frames]  {cur_fps:.1} fps \u{2026}   ");
            let _ = std::io::stdout().flush();
        }
    }
    println!("\r  [{frame_count} frames] done.          ");

    // Signal encoder to finish and wait for it to flush the file.
    if let Some(tx) = enc_tx   { let _ = tx.send(None); }
    if let Some(t)  = enc_thread { let _ = t.join(); }

    let wall_secs = t_wall_start.elapsed().as_secs_f64();
    if frame_count == 0 {
        anyhow::bail!("No frames decoded from {}", input.display());
    }

    let pre  = Stats::compute(&mut pre_ms);
    let inf  = Stats::compute(&mut inf_ms);
    let post = Stats::compute(&mut post_ms);
    let tot  = Stats::compute(&mut tot_ms);
    let lat_fps        = 1000.0 / tot.mean;
    let throughput_fps = frame_count as f64 / wall_secs;

    let device_label = format!("{:?} / {:?}", engine.active_device(), engine.active_precision());
    println!("\n=== RF-DETR Video Benchmark ===");
    println!("Device     : {device_label}");
    println!("Source     : {}", input.display());
    println!("Resolution : {src_w}\u{d7}{src_h}");
    println!("Frames     : {frame_count}");
    println!("Wall time  : {wall_secs:.2} s");
    println!("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  [ms]",
             "Stage", "mean", "stdev", "p50", "p90", "p99", "min", "max");
    print_row("preprocess", &pre);
    print_row("inference",  &inf);
    print_row("postprocess", &post);
    print_row("total",      &tot);
    println!("FPS (latency)  : {lat_fps:.1}");
    println!("FPS (throughput): {throughput_fps:.1}");
    Ok(())
}

/// Draw a 2-pixel thick bounding box directly on a raw BGR24 frame buffer.
fn draw_box_bgr(buf: &mut [u8], w: u32, h: u32, d: &Detection, _classes: &[String]) {
    const COLORS: [[u8; 3]; 5] = [
        [0, 255, 0],     // green   (B,G,R)
        [50, 50, 255],   // red
        [255, 150, 50],  // blue
        [0, 220, 255],   // yellow
        [220, 50, 220],  // magenta
    ];
    let color = COLORS[d.class_id % COLORS.len()];
    let x1 = d.x.max(0) as u32;
    let y1 = d.y.max(0) as u32;
    let x2 = (d.x + d.width).min(w as i32 - 1).max(0) as u32;
    let y2 = (d.y + d.height).min(h as i32 - 1).max(0) as u32;
    for t in 0..2u32 {
        let y1t = y1.saturating_sub(t);
        let y2t = (y2 + t).min(h - 1);
        let x1t = x1.saturating_sub(t);
        let x2t = (x2 + t).min(w - 1);
        for x in x1t..=x2t {
            set_px(buf, w, x, y1t, color);
            set_px(buf, w, x, y2t, color);
        }
        for y in y1t..=y2t {
            set_px(buf, w, x1t, y, color);
            set_px(buf, w, x2t, y, color);
        }
    }
}

#[inline]
fn set_px(buf: &mut [u8], w: u32, x: u32, y: u32, color: [u8; 3]) {
    let idx = (y * w + x) as usize * 3;
    if idx + 2 < buf.len() {
        buf[idx] = color[0];
        buf[idx + 1] = color[1];
        buf[idx + 2] = color[2];
    }
}

