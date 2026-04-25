use std::{path::Path, time::Instant};

#[cfg(any(feature = "cuda", feature = "tensorrt"))]
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::{
    ep,
    session::{
        Session,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
    value::{Tensor, TensorRef, ValueType},
};

use crate::{
    config::{Device, EngineConfig, Precision},
    error::{Error, Result},
    postprocess::{Detection, postprocess},
    preprocess::{preprocess_bgr_into_slice, preprocess_into, preprocess_into_slice, resize_u8x3},
};

/// Returns the milliseconds elapsed since `t`, in milliseconds.
#[inline(always)]
fn elapsed_ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Timing breakdown for a single inference call (milliseconds).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub postprocess_ms: f64,
}

impl Timings {
    #[inline]
    pub fn total_ms(&self) -> f64 {
        self.preprocess_ms + self.inference_ms + self.postprocess_ms
    }
}

/// Model input/output shape metadata resolved at session-creation time.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub input_width: usize,
    pub input_height: usize,
    pub input_channels: usize,
    pub num_queries: usize,
    pub num_classes: usize,
}

/// High-performance RF-DETR inference engine backed by ONNX Runtime.
///
/// Create with [`Engine::new`], then call [`Engine::infer`] or
/// [`Engine::infer_batch`] on `image::DynamicImage` values.
pub struct Engine {
    session: Session,
    model_info: ModelInfo,
    active_device: Device,
    active_precision: Precision,
    config: EngineConfig,
    last_timings: Timings,
    /// ORT-owned CPU tensor used as the fast-path input buffer.
    /// Bound to io_binding ONCE at engine creation; its heap address never changes,
    /// allowing the TRT EP to capture and replay a CUDA graph without re-capture.
    /// `None` for CPU device or when IoBinding setup fails.
    /// NOTE: the field value itself is the ownership anchor; we access the data via
    /// `input_ptr` to avoid re-borrowing through the Option on every inference call.
    #[allow(dead_code)]
    input_tensor: Option<Tensor<f32>>,
    /// Cached raw pointer into `input_tensor`'s data buffer for zero-overhead writes.
    /// SAFETY: valid for `input_len` f32s as long as `input_tensor` is alive.
    input_ptr: *mut f32,
    /// Cached element count of the input tensor buffer.
    input_len: usize,
    /// Reusable NCHW float buffer for the fallback path (batch > 1 or CPU device).
    preprocess_buf: Vec<f32>,
    /// Reusable scratch buffer for rgb8 conversion / slow-path resize.
    scratch_rgb: Vec<u8>,
    /// Reusable output accumulator buffers for the batch fallback path.
    boxes_out: Vec<f32>,
    logits_out: Vec<f32>,
}

// ORT sessions are Send (but not Sync, because run() requires &mut self).
unsafe impl Send for Engine {}

impl Engine {
    /// Build an engine from the given configuration.
    ///
    /// Attempts to initialise the best available execution provider according
    /// to `config.device`, falling back through TRT → CUDA → CPU when
    /// `config.auto_fallback` is true.
    pub fn new(config: EngineConfig) -> Result<Self> {
        if !config.model_path.exists() {
            return Err(Error::ModelNotFound(
                config.model_path.display().to_string(),
            ));
        }

        let chain: Vec<Device> = build_device_chain(&config);
        let mut last_err = Error::NoProviderAvailable;
        let mut session_result: Option<(Session, Device)> = None;

        for dev in &chain {
            match build_session(&config, dev) {
                Ok(s) => {
                    session_result = Some((s, dev.clone()));
                    break;
                }
                Err(e) => {
                    eprintln!("[rfdetr] {:?} failed: {e}", dev);
                    last_err = e;
                }
            }
        }

        let (session, active_device) = session_result.ok_or(last_err)?;

        // Device::Auto resolves to whichever EP succeeded (TRT/CUDA/CPU), so
        // honour config.precision rather than hard-coding Fp32.  Only a
        // plain Device::Cpu request always runs FP32.
        let active_precision = match &active_device {
            Device::Cpu => Precision::Fp32,
            _ => config.precision,
        };

        let model_info = resolve_model_info(&session)?;

        let cap = model_info.input_channels
            * model_info.input_width
            * model_info.input_height
            * config.max_batch_size;

        let nq = model_info.num_queries;
        let nc = model_info.num_classes;
        let h = model_info.input_height;
        let w = model_info.input_width;

        let input_len =
            model_info.input_channels * model_info.input_height * model_info.input_width;

        // For TRT/CUDA: allocate a persistent ORT-owned CPU tensor as the input buffer.
        // Use CUDA pinned (page-locked) memory so the H2D DMA transfer can bypass the
        // CPU staging bounce buffer — faster H2D especially combined with CUDA graph.
        // Falls back to default pageable allocator if the CUDA pinned allocator is
        // unavailable (e.g. no CUDA runtime or wrong device_id).
        #[cfg(any(feature = "cuda", feature = "tensorrt"))]
        let mut input_tensor: Option<Tensor<f32>> =
            if matches!(active_device, Device::TensorRt | Device::Cuda) {
                let alloc = MemoryInfo::new(
                    AllocationDevice::CUDA_PINNED,
                    config.device_id,
                    AllocatorType::Device,
                    MemoryType::CPUOutput,
                )
                .and_then(|mi| Allocator::new(&session, mi))
                .unwrap_or_else(|_| Allocator::default());
                match Tensor::<f32>::new(
                    &alloc,
                    [
                        1usize,
                        model_info.input_channels,
                        model_info.input_height,
                        model_info.input_width,
                    ],
                ) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        eprintln!(
                            "[rfdetr] Tensor::new failed ({e}); falling back to TensorRef path"
                        );
                        None
                    }
                }
            } else {
                None
            };

        #[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
        let mut input_tensor: Option<Tensor<f32>> = None;

        // SAFETY: ORT allocates the tensor data on the heap; the pointer is stable
        // for the Engine's lifetime because input_tensor is never resized or dropped
        // (it lives in the Engine struct).
        let input_ptr: *mut f32 = input_tensor
            .as_mut()
            .map(|t| t.data_ptr_mut().cast::<f32>())
            .unwrap_or(std::ptr::null_mut());

        Ok(Self {
            session,
            model_info,
            active_device,
            active_precision,
            config,
            last_timings: Timings::default(),
            input_tensor,
            input_ptr,
            input_len,
            preprocess_buf: Vec::with_capacity(cap),
            scratch_rgb: Vec::with_capacity(3 * w * h),
            boxes_out: Vec::with_capacity(nq * 4),
            logits_out: Vec::with_capacity(nq * nc),
        })
    }

    /// Run inference on a single image.
    ///
    /// `image` can be any size and format; it is resized internally.
    pub fn infer(
        &mut self,
        image: &image::DynamicImage,
        conf_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let mut out = self.infer_batch_impl(std::slice::from_ref(image), conf_threshold)?;
        Ok(out.pop().unwrap_or_default())
    }

    /// Run inference on a batch of images.
    ///
    /// Panics if `images.len() > config.max_batch_size`.
    pub fn infer_batch(
        &mut self,
        images: &[image::DynamicImage],
        conf_threshold: f32,
    ) -> Result<Vec<Vec<Detection>>> {
        self.infer_batch_impl(images, conf_threshold)
    }

    /// Run inference on a raw BGR24 frame buffer (e.g. from a video decoder or camera).
    ///
    /// `bgr` must be exactly `src_w * src_h * 3` bytes in HWC layout with B,G,R channel order.
    /// Resizes to the model's input dimensions internally when `src_w`/`src_h` differ, using
    /// fast_image_resize (Lanczos3 SIMD). Writes directly into the GPU-pinned buffer — the
    /// same fast path as [`Engine::infer`] on a pre-sized image. Preprocess timing is recorded.
    pub fn infer_frame(
        &mut self,
        bgr: &[u8],
        src_w: u32,
        src_h: u32,
        conf_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let mi = &self.model_info;
        let w = mi.input_width as u32;
        let h = mi.input_height as u32;
        let c = mi.input_channels;
        let nq = mi.num_queries;
        let nc = mi.num_classes;

        let t0 = Instant::now();

        if !self.input_ptr.is_null() {
            // SAFETY: input_ptr is valid for input_len f32s for the Engine's lifetime.
            let dst = unsafe { std::slice::from_raw_parts_mut(self.input_ptr, self.input_len) };
            if src_w == w && src_h == h {
                preprocess_bgr_into_slice(bgr, w, h, &self.config.mean, &self.config.std, dst);
            } else {
                let target = (3 * w * h) as usize;
                self.scratch_rgb.resize(target, 0);
                resize_u8x3(bgr, src_w, src_h, &mut self.scratch_rgb, w, h);
                preprocess_bgr_into_slice(
                    &self.scratch_rgb,
                    w,
                    h,
                    &self.config.mean,
                    &self.config.std,
                    dst,
                );
            }
        } else {
            // Pageable memory path: CPU device or pinned alloc unavailable.
            let n = (3 * w * h) as usize;
            self.preprocess_buf.resize(n, 0.0);
            if src_w == w && src_h == h {
                preprocess_bgr_into_slice(
                    bgr,
                    w,
                    h,
                    &self.config.mean,
                    &self.config.std,
                    &mut self.preprocess_buf,
                );
            } else {
                self.scratch_rgb.resize((3 * w * h) as usize, 0);
                resize_u8x3(bgr, src_w, src_h, &mut self.scratch_rgb, w, h);
                preprocess_bgr_into_slice(
                    &self.scratch_rgb,
                    w,
                    h,
                    &self.config.mean,
                    &self.config.std,
                    &mut self.preprocess_buf,
                );
            }
        }

        let preprocess_ms = elapsed_ms(t0);

        let t1 = Instant::now();
        let in_shape = [1i64, c as i64, h as i64, w as i64];
        let input_data: &[f32] = if !self.input_ptr.is_null() {
            unsafe { std::slice::from_raw_parts(self.input_ptr, self.input_len) }
        } else {
            &self.preprocess_buf
        };
        let input_t = TensorRef::<f32>::from_array_view((in_shape, input_data))?;
        let outputs = self.session.run(ort::inputs![input_t])?;
        let inference_ms = elapsed_ms(t1);

        let t2 = Instant::now();
        let (_, boxes_raw) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, logits_raw) = outputs[1].try_extract_tensor::<f32>()?;
        let detections = postprocess(boxes_raw, logits_raw, nq, nc, src_w, src_h, conf_threshold);
        let postprocess_ms = elapsed_ms(t2);

        self.last_timings = Timings {
            preprocess_ms,
            inference_ms,
            postprocess_ms,
        };
        Ok(detections)
    }

    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    pub fn last_timings(&self) -> Timings {
        self.last_timings
    }

    pub fn active_device(&self) -> &Device {
        &self.active_device
    }

    pub fn active_precision(&self) -> Precision {
        self.active_precision
    }

    fn infer_batch_impl(
        &mut self,
        images: &[image::DynamicImage],
        conf_threshold: f32,
    ) -> Result<Vec<Vec<Detection>>> {
        if images.is_empty() {
            return Ok(vec![]);
        }
        if images.len() > self.config.max_batch_size {
            return Err(Error::BatchTooLarge {
                requested: images.len(),
                max: self.config.max_batch_size,
            });
        }

        let mi = &self.model_info;
        let batch = images.len();
        let w = mi.input_width as u32;
        let h = mi.input_height as u32;
        let c = mi.input_channels;
        let nq = mi.num_queries;
        let nc = mi.num_classes;

        // Preprocess directly into the CUDA-pinned ORT tensor so the H2D DMA reads
        // from page-locked memory (no CPU staging copy needed inside CUDA).
        if batch == 1 && !self.input_ptr.is_null() {
            let img = &images[0];
            let orig_w = img.width();
            let orig_h = img.height();

            // SAFETY: input_ptr was initialised from input_tensor.data_ptr_mut() in
            // Engine::new; input_tensor lives for the Engine's lifetime; this borrow
            // ends before session.run() is called.
            let t0 = Instant::now();
            let preprocess_slice =
                unsafe { std::slice::from_raw_parts_mut(self.input_ptr, self.input_len) };
            preprocess_into_slice(
                img,
                w,
                h,
                &self.config.mean,
                &self.config.std,
                preprocess_slice,
                &mut self.scratch_rgb,
            );
            let preprocess_ms = elapsed_ms(t0);

            let t1 = Instant::now();
            let in_shape = [1i64, c as i64, h as i64, w as i64];
            // SAFETY: pinned_ref is not mutated again until the next infer call.
            let pinned_ref = unsafe { std::slice::from_raw_parts(self.input_ptr, self.input_len) };
            let input_t = TensorRef::<f32>::from_array_view((in_shape, pinned_ref))?;
            let outputs = self.session.run(ort::inputs![input_t])?;
            let inference_ms = elapsed_ms(t1);

            let t2 = Instant::now();
            let (_, boxes_raw) = outputs[0].try_extract_tensor::<f32>()?;
            let (_, logits_raw) = outputs[1].try_extract_tensor::<f32>()?;
            let detections = postprocess(
                boxes_raw,
                logits_raw,
                nq,
                nc,
                orig_w,
                orig_h,
                conf_threshold,
            );
            let postprocess_ms = elapsed_ms(t2);

            self.last_timings = Timings {
                preprocess_ms,
                inference_ms,
                postprocess_ms,
            };
            return Ok(vec![detections]);
        }

        // Pageable memory path: CPU device, batch > 1, or pinned alloc unavailable.
        let t0 = Instant::now();
        self.preprocess_buf.clear();
        let orig_sizes: Vec<(u32, u32)> = images
            .iter()
            .map(|img| {
                let ow = img.width();
                let oh = img.height();
                preprocess_into(
                    img,
                    w,
                    h,
                    &self.config.mean,
                    &self.config.std,
                    &mut self.preprocess_buf,
                    &mut self.scratch_rgb,
                );
                (ow, oh)
            })
            .collect();
        let preprocess_ms = elapsed_ms(t0);

        let t1 = Instant::now();
        {
            let in_shape = [batch as i64, c as i64, h as i64, w as i64];
            let input_tensor =
                TensorRef::<f32>::from_array_view((in_shape, self.preprocess_buf.as_slice()))?;
            let outputs = self.session.run(ort::inputs![input_tensor])?;
            let (_, boxes_raw) = outputs[0].try_extract_tensor::<f32>()?;
            let (_, logits_raw) = outputs[1].try_extract_tensor::<f32>()?;
            self.boxes_out.clear();
            self.boxes_out.extend_from_slice(boxes_raw);
            self.logits_out.clear();
            self.logits_out.extend_from_slice(logits_raw);
        }
        let inference_ms = elapsed_ms(t1);

        let t2 = Instant::now();
        let result: Vec<Vec<Detection>> = (0..batch)
            .map(|b| {
                let (ow, oh) = orig_sizes[b];
                let box_slice = &self.boxes_out[b * nq * 4..(b + 1) * nq * 4];
                let log_slice = &self.logits_out[b * nq * nc..(b + 1) * nq * nc];
                postprocess(box_slice, log_slice, nq, nc, ow, oh, conf_threshold)
            })
            .collect();
        let postprocess_ms = elapsed_ms(t2);

        self.last_timings = Timings {
            preprocess_ms,
            inference_ms,
            postprocess_ms,
        };
        Ok(result)
    }
}

fn build_device_chain(config: &EngineConfig) -> Vec<Device> {
    match config.device {
        Device::Auto => {
            #[cfg(all(feature = "tensorrt", feature = "cuda"))]
            return vec![Device::TensorRt, Device::Cuda, Device::Cpu];
            #[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
            return vec![Device::Cuda, Device::Cpu];
            #[cfg(not(feature = "cuda"))]
            vec![Device::Cpu]
        }
        Device::TensorRt => {
            #[cfg(feature = "tensorrt")]
            if config.auto_fallback {
                #[cfg(feature = "cuda")]
                return vec![Device::TensorRt, Device::Cuda, Device::Cpu];
                #[cfg(not(feature = "cuda"))]
                return vec![Device::TensorRt, Device::Cpu];
            }
            #[cfg(feature = "tensorrt")]
            return vec![Device::TensorRt];
            #[cfg(not(feature = "tensorrt"))]
            vec![Device::Cpu]
        }
        Device::Cuda => {
            #[cfg(feature = "cuda")]
            if config.auto_fallback {
                return vec![Device::Cuda, Device::Cpu];
            }
            #[cfg(feature = "cuda")]
            return vec![Device::Cuda];
            #[cfg(not(feature = "cuda"))]
            vec![Device::Cpu]
        }
        Device::Cpu => vec![Device::Cpu],
    }
}

fn build_session(config: &EngineConfig, device: &Device) -> Result<Session> {
    let mut ep_list: Vec<ep::ExecutionProviderDispatch> = vec![];

    match device {
        #[cfg(feature = "tensorrt")]
        Device::TensorRt => {
            std::fs::create_dir_all(&config.trt_cache_dir)?;
            let cache = config.trt_cache_dir.to_string_lossy().to_string();
            let opt_level: u8 = if config.trt_max_optimization { 5 } else { 3 };

            ep_list.push(
                ep::TensorRT::default()
                    .with_device_id(config.device_id)
                    .with_fp16(config.precision == Precision::Fp16)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&cache)
                    .with_timing_cache(true)
                    .with_timing_cache_path(&cache)
                    .with_builder_optimization_level(opt_level)
                    .with_max_workspace_size(config.trt_workspace_bytes)
                    // Capture all CUDA ops (H2D + TRT kernel + D2H) as a CUDA graph
                    // on the first run_binding call; replay cheaply for every frame.
                    // Requires stable input/output memory addresses — satisfied by the
                    // persistent input_tensor + pre-bound output IoBinding.
                    .with_cuda_graph(config.enable_cuda_graph)
                    .build(),
            );
            // CUDA EP as fallback for any ops TRT cannot compile.
            ep_list.push(ep::CUDA::default().with_device_id(config.device_id).build());
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            ep_list.push(
                ep::CUDA::default()
                    .with_device_id(config.device_id)
                    .with_cuda_graph(config.enable_cuda_graph)
                    .build(),
            );
        }
        _ => {}
    }

    ep_list.push(ep::CPU::default().build());

    let sb = |e: ort::Error<SessionBuilder>| Error::SessionBuild(e.to_string());

    let mut builder = Session::builder()?
        .with_execution_providers(ep_list)
        .map_err(sb)?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(sb)?;

    if config.intra_op_threads > 0 {
        builder = builder
            .with_intra_threads(config.intra_op_threads)
            .map_err(sb)?;
    }
    if config.inter_op_threads > 0 {
        builder = builder
            .with_inter_threads(config.inter_op_threads)
            .map_err(sb)?;
    }

    Ok(builder.commit_from_file(Path::new(&config.model_path))?)
}

fn resolve_model_info(session: &Session) -> Result<ModelInfo> {
    let in_type = session
        .inputs()
        .first()
        .ok_or_else(|| Error::InvalidModel("Model has no inputs".into()))?;

    // Fallback values are RF-DETR canonical defaults used when the ONNX model
    // was exported with dynamic/symbolic dimensions (shape value <= 0):
    //   c=3          — RGB input channels
    //   h=384, w=384 — standard RF-DETR-B/L input resolution
    let (ic, ih, iw) = match in_type.dtype() {
        ValueType::Tensor { shape, .. } if shape.len() == 4 => {
            let c = if shape[1] > 0 { shape[1] as usize } else { 3 };
            let h = if shape[2] > 0 { shape[2] as usize } else { 384 };
            let w = if shape[3] > 0 { shape[3] as usize } else { 384 };
            (c, h, w)
        }
        ValueType::Tensor { shape, .. } => {
            return Err(Error::InvalidModel(format!(
                "Expected 4-D NCHW input, got {}-D",
                shape.len()
            )));
        }
        _ => return Err(Error::InvalidModel("Expected tensor input".into())),
    };

    let outs = session.outputs();
    if outs.len() < 2 {
        return Err(Error::InvalidModel(format!(
            "Expected at least 2 outputs (pred_boxes, pred_logits), found {}",
            outs.len()
        )));
    }

    // Fallback values for dynamic output shapes:
    //   nq=300 — standard DETR object query count (RF-DETR default)
    //   nc=91  — COCO 80-class + 1 background = 91 logits per query
    let nq = match outs[0].dtype() {
        ValueType::Tensor { shape, .. } => {
            if shape.len() >= 2 && shape[1] > 0 {
                shape[1] as usize
            } else {
                300
            }
        }
        _ => 300,
    };

    let nc = match outs[1].dtype() {
        ValueType::Tensor { shape, .. } => {
            if shape.len() >= 3 && shape[2] > 0 {
                shape[2] as usize
            } else {
                91
            }
        }
        _ => 91,
    };

    Ok(ModelInfo {
        input_width: iw,
        input_height: ih,
        input_channels: ic,
        num_queries: nq,
        num_classes: nc,
    })
}
