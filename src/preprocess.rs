use fast_image_resize as fir;
use fast_image_resize::images::Image as FirImage;
use image::DynamicImage;
use rayon::prelude::*;

/// Normalize + HWC→CHW scatter into a pre-allocated `dst` slice of exactly
/// `3 * dst_w * dst_h` elements.
///
/// `scratch` is a caller-provided reusable `Vec<u8>` used only when format
/// conversion is required (avoids a heap allocation on the hot paths where the
/// image is already RGB8 or RGBA8 at the target dimensions).
pub(crate) fn preprocess_into_slice(
    image: &DynamicImage,
    dst_w: u32,
    dst_h: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    dst: &mut [f32],
    scratch: &mut Vec<u8>,
) {
    let area = (dst_w * dst_h) as usize;
    debug_assert_eq!(
        dst.len(),
        3 * area,
        "preprocess_into_slice: dst length mismatch"
    );

    let inv_std_r = 1.0_f32 / (255.0 * std[0]);
    let inv_std_g = 1.0_f32 / (255.0 * std[1]);
    let inv_std_b = 1.0_f32 / (255.0 * std[2]);
    // Correct ImageNet normalisation: (px/255 - mean) / std = px/(255*std) - mean/std
    let mean_r = mean[0] / std[0];
    let mean_g = mean[1] / std[1];
    let mean_b = mean[2] / std[2];

    let (r_plane, rest) = dst.split_at_mut(area);
    let (g_plane, b_plane) = rest.split_at_mut(area);

    // Helper: normalize a packed RGB8 pixel stream into the three CHW planes.
    // Uses zip iteration to eliminate bounds checks and give LLVM clearer
    // loop-carried independence between the three output streams.
    macro_rules! normalize_rgb3 {
        ($pixels:expr, $stride:literal) => {
            for (((r, g), b), px) in r_plane
                .iter_mut()
                .zip(g_plane.iter_mut())
                .zip(b_plane.iter_mut())
                .zip($pixels.chunks_exact($stride))
            {
                *r = px[0] as f32 * inv_std_r - mean_r;
                *g = px[1] as f32 * inv_std_g - mean_g;
                *b = px[2] as f32 * inv_std_b - mean_b;
            }
        };
    }

    match image {
        // Fast path A: RGB8 at target size — single-pass normalize, zero alloc.
        DynamicImage::ImageRgb8(buf) if buf.width() == dst_w && buf.height() == dst_h => {
            normalize_rgb3!(buf.as_raw(), 3);
        }
        // Fast path B: RGBA8 at target size — single-pass, stride-4 read, zero alloc.
        // Avoids a full intermediate 442 KB RGB scratch buffer, saving ~33% bandwidth.
        DynamicImage::ImageRgba8(buf) if buf.width() == dst_w && buf.height() == dst_h => {
            normalize_rgb3!(buf.as_raw(), 4);
        }
        // Slow path: resize using fast_image_resize (AVX2 SIMD — ~10-20x faster than
        // image crate's Triangle filter for typical inference input sizes).
        // Resizes into the pre-allocated `scratch` buffer to avoid extra heap allocation.
        _ => {
            let target_bytes = 3 * area;
            // Ensure scratch is exactly the right size (no realloc after first call).
            scratch.resize(target_bytes, 0u8);

            // Convert to RGB8 first to guarantee U8x3 pixel layout for the resizer.
            // `into_raw()` moves the Vec without copying; `to_rgb8()` is a clone only
            // for formats already RGB8 (unavoidable — need a separate source buffer).
            let rgb = image.to_rgb8();
            let src = FirImage::from_vec_u8(
                rgb.width(),
                rgb.height(),
                rgb.into_raw(),
                fir::PixelType::U8x3,
            )
            .expect("preprocess: fir source dimensions mismatch");

            {
                let mut dst_img = FirImage::from_slice_u8(
                    dst_w,
                    dst_h,
                    scratch.as_mut_slice(),
                    fir::PixelType::U8x3,
                )
                .expect("preprocess: fir destination dimensions mismatch");

                fir::Resizer::new()
                    .resize(&src, &mut dst_img, None)
                    .expect("preprocess: fir resize failed");
                // dst_img dropped here, releasing the mutable borrow of scratch.
            }

            normalize_rgb3!(scratch.as_slice(), 3);
        }
    }
}

/// Normalize a raw BGR (HWC u8) buffer of exactly `dst_w × dst_h` pixels
/// directly into `dst` (NCHW f32), swapping R↔B while scattering so the
/// output planes are in the [R, G, B] order the model expects.
/// Only valid when the source is already at the model's input dimensions.
pub(crate) fn preprocess_bgr_into_slice(
    bgr: &[u8],
    dst_w: u32,
    dst_h: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    dst: &mut [f32],
) {
    let area = (dst_w * dst_h) as usize;
    debug_assert_eq!(
        bgr.len(),
        3 * area,
        "preprocess_bgr_into_slice: source length mismatch"
    );
    debug_assert_eq!(
        dst.len(),
        3 * area,
        "preprocess_bgr_into_slice: dst length mismatch"
    );

    let inv_std_r = 1.0_f32 / (255.0 * std[0]);
    let inv_std_g = 1.0_f32 / (255.0 * std[1]);
    let inv_std_b = 1.0_f32 / (255.0 * std[2]);
    let mean_r = mean[0] / std[0];
    let mean_g = mean[1] / std[1];
    let mean_b = mean[2] / std[2];

    let (r_plane, rest) = dst.split_at_mut(area);
    let (g_plane, b_plane) = rest.split_at_mut(area);

    // BGR layout: chunk[0]=B, chunk[1]=G, chunk[2]=R
    for (((r, g), b), px) in r_plane
        .iter_mut()
        .zip(g_plane.iter_mut())
        .zip(b_plane.iter_mut())
        .zip(bgr.chunks_exact(3))
    {
        *r = px[2] as f32 * inv_std_r - mean_r;
        *g = px[1] as f32 * inv_std_g - mean_g;
        *b = px[0] as f32 * inv_std_b - mean_b;
    }
}

/// Resize `image` to `(dst_w, dst_h)`, normalise and append the NCHW result
/// to `dst`.  `scratch` is a reusable intermediate byte buffer (see
/// [`preprocess_into_slice`]).
pub(crate) fn preprocess_into(
    image: &DynamicImage,
    dst_w: u32,
    dst_h: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    dst: &mut Vec<f32>,
    scratch: &mut Vec<u8>,
) {
    let area = (dst_w * dst_h) as usize;
    let plane_base = dst.len();
    let needed = plane_base + 3 * area;
    if dst.capacity() < needed {
        dst.reserve(needed - dst.len());
    }
    // SAFETY: every element in [plane_base, needed) is immediately written by
    // preprocess_into_slice below; capacity has been ensured above.
    unsafe { dst.set_len(needed) };
    preprocess_into_slice(
        image,
        dst_w,
        dst_h,
        mean,
        std,
        &mut dst[plane_base..],
        scratch,
    );
}

/// Spatially resize a packed U8×3 buffer (any channel order — BGR, RGB, etc.)
/// from `(src_w, src_h)` to `(dst_w, dst_h)` using fast_image_resize (Lanczos3 SIMD).
///
/// The resize is purely spatial; channel values are interpolated independently,
/// so channel order is preserved exactly as-is in the output.
/// `dst` must be pre-allocated to exactly `dst_w * dst_h * 3` bytes.
pub(crate) fn resize_u8x3(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
) {
    let src_img = FirImage::from_vec_u8(src_w, src_h, src.to_vec(), fir::PixelType::U8x3)
        .expect("resize_u8x3: source dimensions mismatch");
    let mut dst_img = FirImage::from_slice_u8(dst_w, dst_h, dst, fir::PixelType::U8x3)
        .expect("resize_u8x3: destination dimensions mismatch");
    fir::Resizer::new()
        .resize(&src_img, &mut dst_img, None)
        .expect("resize_u8x3: resize failed");
}

/// Convert a NV12 semi-planar frame (already at the model's input dimensions)
/// directly into an NCHW f32 slice, fusing the YUV→RGB conversion and
/// ImageNet normalisation in a single pass with no intermediate buffer.
///
/// Layout of `nv12`:
/// - Y plane  : `w * h` bytes at offset 0
/// - UV plane : `w * h / 2` bytes immediately after (interleaved U,V pairs,
///   each pair covers a 2×2 luma block)
///
/// Total size must be exactly `w * h * 3 / 2`. `w` and `h` must be even
/// (true for any standard NV12 source — 384×384 satisfies this).
///
/// Colour space: BT.601 full-range (PC/JPEG swing, Y∈[0,255], UV∈[0,255]).
///
/// ## Performance
///
/// Processes NV12 in **2×2 macro-blocks** with Rayon parallelism across
/// row-pairs.  Each (U,V) pair is loaded once and shared across all 4 luma
/// pixels.  Output planes are split into independent per-thread chunks via
/// `par_chunks_mut` (one chunk per CPU thread to avoid work-stealing overhead),
/// so threads hold no shared mutable state.
pub(crate) fn preprocess_nv12_into_slice(
    nv12: &[u8],
    w: u32,
    h: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    dst: &mut [f32],
) {
    let w = w as usize;
    let h = h as usize;
    let area = w * h;

    debug_assert_eq!(
        nv12.len(),
        area * 3 / 2,
        "preprocess_nv12_into_slice: nv12 length mismatch"
    );
    debug_assert_eq!(
        dst.len(),
        3 * area,
        "preprocess_nv12_into_slice: dst length mismatch"
    );
    debug_assert_eq!(w & 1, 0, "preprocess_nv12_into_slice: width must be even");
    debug_assert_eq!(h & 1, 0, "preprocess_nv12_into_slice: height must be even");

    // Pre-fuse BT.601 full-range YUV→RGB coefficients with ImageNet
    // normalisation so the hot loop contains only FMAs:
    //
    //   out_R = (Y + 1.402*(V-128))                   * (1/255/std_r) - mean_r/std_r
    //         = Y*ky_r  +  V*kv_r  +  bias_r
    //
    //   out_G = (Y − 0.344*(U-128) − 0.714*(V-128))  * (1/255/std_g) - mean_g/std_g
    //         = Y*ky_g  +  U*ku_g  +  V*kv_g  +  bias_g
    //
    //   out_B = (Y + 1.772*(U-128))                   * (1/255/std_b) - mean_b/std_b
    //         = Y*ky_b  +  U*ku_b  +  bias_b
    let ky_r = 1.0_f32 / (255.0 * std[0]);
    let ky_g = 1.0_f32 / (255.0 * std[1]);
    let ky_b = 1.0_f32 / (255.0 * std[2]);
    let kv_r = 1.402_f32 / (255.0 * std[0]);
    let ku_g = -0.344_136_f32 / (255.0 * std[1]);
    let kv_g = -0.714_136_f32 / (255.0 * std[1]);
    let ku_b = 1.772_f32 / (255.0 * std[2]);
    let bias_r = -128.0 * kv_r - mean[0] / std[0];
    let bias_g = -128.0 * (ku_g + kv_g) - mean[1] / std[1];
    let bias_b = -128.0 * ku_b - mean[2] / std[2];

    let y_plane = &nv12[..area];
    let uv_plane = &nv12[area..];

    let (r_plane, rest) = dst.split_at_mut(area);
    let (g_plane, b_plane) = rest.split_at_mut(area);

    let n_threads = rayon::current_num_threads().max(1);
    let rows_per_chunk = (h / 2).div_ceil(n_threads);
    let chunk_elems = 2 * w * rows_per_chunk;

    r_plane
        .par_chunks_mut(chunk_elems)
        .zip(g_plane.par_chunks_mut(chunk_elems))
        .zip(b_plane.par_chunks_mut(chunk_elems))
        .enumerate()
        .for_each(|(ci, ((r_chunk, g_chunk), b_chunk))| {
            let row2_start = ci * rows_per_chunk;
            let actual_row_pairs = r_chunk.len() / (2 * w);

            for local_row2 in 0..actual_row_pairs {
                let row2 = row2_start + local_row2;
                let row0 = row2 * 2;
                let row1 = row0 + 1;

                let y_row0 = &y_plane[row0 * w..][..w];
                let y_row1 = &y_plane[row1 * w..][..w];
                let uv_row = &uv_plane[row2 * w..][..w];
                let (r_row0, r_row1) = r_chunk[local_row2 * 2 * w..][..2 * w].split_at_mut(w);
                let (g_row0, g_row1) = g_chunk[local_row2 * 2 * w..][..2 * w].split_at_mut(w);
                let (b_row0, b_row1) = b_chunk[local_row2 * 2 * w..][..2 * w].split_at_mut(w);

                for col2 in 0..w / 2 {
                    let col0 = col2 * 2;

                    let u = uv_row[col0] as f32;
                    let v = uv_row[col0 + 1] as f32;
                    let r_uv = v.mul_add(kv_r, bias_r);
                    let g_uv = u.mul_add(ku_g, v.mul_add(kv_g, bias_g));
                    let b_uv = u.mul_add(ku_b, bias_b);

                    let y00 = y_row0[col0] as f32;
                    let y01 = y_row0[col0 + 1] as f32;
                    let y10 = y_row1[col0] as f32;
                    let y11 = y_row1[col0 + 1] as f32;

                    r_row0[col0] = y00.mul_add(ky_r, r_uv);
                    r_row0[col0 + 1] = y01.mul_add(ky_r, r_uv);
                    r_row1[col0] = y10.mul_add(ky_r, r_uv);
                    r_row1[col0 + 1] = y11.mul_add(ky_r, r_uv);

                    g_row0[col0] = y00.mul_add(ky_g, g_uv);
                    g_row0[col0 + 1] = y01.mul_add(ky_g, g_uv);
                    g_row1[col0] = y10.mul_add(ky_g, g_uv);
                    g_row1[col0 + 1] = y11.mul_add(ky_g, g_uv);

                    b_row0[col0] = y00.mul_add(ky_b, b_uv);
                    b_row0[col0 + 1] = y01.mul_add(ky_b, b_uv);
                    b_row1[col0] = y10.mul_add(ky_b, b_uv);
                    b_row1[col0 + 1] = y11.mul_add(ky_b, b_uv);
                }
            }
        });
}
