use crate::Detection;

/// Convert raw model outputs to pixel-space detections filtered by a confidence
/// threshold.
///
/// # Arguments
/// - `boxes`   – raw `[num_queries * 4]` tensor: (cx, cy, w, h) in [0, 1].
/// - `logits`  – raw `[num_queries * num_classes]` tensor (pre-sigmoid logits).
/// - `orig_w / orig_h` – original image dimensions for scaling back to pixels.
pub fn postprocess(
    boxes: &[f32],
    logits: &[f32],
    num_queries: usize,
    num_classes: usize,
    orig_w: u32,
    orig_h: u32,
    conf_threshold: f32,
) -> Vec<Detection> {
    assert_eq!(boxes.len(), num_queries * 4);
    assert_eq!(logits.len(), num_queries * num_classes);

    let mut detections = Vec::with_capacity(num_queries / 4);
    let wf = orig_w as f32;
    let hf = orig_h as f32;

    // Precompute the confidence threshold in logit space.
    // sigmoid(x) >= t  ⟺  x >= ln(t / (1 - t))
    // This lets us skip the expensive sigmoid()+exp() call for the ~295/300
    // queries that will never exceed the threshold.
    let logit_threshold = (conf_threshold / (1.0 - conf_threshold)).ln();

    for q in 0..num_queries {
        let lp = &logits[q * num_classes..(q + 1) * num_classes];

        // Argmax over raw logits. sigmoid() is monotone, so argmax of logits
        // equals argmax of sigmoid — we only sigmoid the winner.
        let mut best_class = 0usize;
        let mut best_logit = f32::NEG_INFINITY;
        for (i, &v) in lp.iter().enumerate() {
            if v > best_logit {
                best_logit = v;
                best_class = i;
            }
        }

        // Fast rejection in logit space — avoids exp() for non-detections.
        if best_logit < logit_threshold {
            continue;
        }
        let conf = sigmoid(best_logit);

        let bp = &boxes[q * 4..];
        let cx = bp[0];
        let cy = bp[1];
        let bw = bp[2];
        let bh = bp[3];

        let x1 = ((cx - 0.5 * bw).clamp(0.0, 1.0) * wf) as i32;
        let y1 = ((cy - 0.5 * bh).clamp(0.0, 1.0) * hf) as i32;
        let x2 = ((cx + 0.5 * bw).clamp(0.0, 1.0) * wf) as i32;
        let y2 = ((cy + 0.5 * bh).clamp(0.0, 1.0) * hf) as i32;

        detections.push(Detection {
            x: x1,
            y: y1,
            width: (x2 - x1).max(0),
            height: (y2 - y1).max(0),
            class_id: best_class,
            confidence: conf,
        });
    }

    detections
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-x).exp())
}
