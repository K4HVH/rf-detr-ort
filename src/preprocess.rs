use fast_image_resize as fir;
use fast_image_resize::images::Image as FirImage;
use image::DynamicImage;

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
