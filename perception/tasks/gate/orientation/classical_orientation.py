import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from perception.registry import register_perceiver
from perception.tasks.TaskPerceiver import DetectionResult, PerceptionOutput, TaskContext, TaskPerceiver


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

L_THRESHOLD   = 60 
A_THRESHOLD   = 135 

MIN_BLOB_AREA = 50 # noise filter
TOLERANCE     = 0.04

COLOR_LEFT    = (50,  220,  50) 
COLOR_RIGHT   = (50,   50, 220) 
COLOR_GATE    = (0,   165, 255) 
COLOR_DASH_BG = (10,   10,  10) 

# POST_ASPECT_MIN = 0.15 -- might use these later
# POST_ASPECT_MAX = 4.0

# ══════════════════════════════════════════════════════════════════════════════
# Underwater enhancement
# ══════════════════════════════════════════════════════════════════════════════

def enhance_underwater(img: np.ndarray) -> np.ndarray:
    result = img.copy().astype(np.float32)

    # -- white balancing => scale each channel so the darkest pixel = 0 and brightest = 255 --
    for i in range(3):
        ch = result[:, :, i]
        lo, hi = ch.min(), ch.max()
        if hi > lo:

            # Linearly stretch channel to 0-255 range
            result[:, :, i] = (ch - lo) / (hi - lo) * 255

    # --------- CLAHE -------------
    result = result.astype(np.uint8)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    # CLAHE l channel only => brightens pixels without distorting color
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ══════════════════════════════════════════════════════════════════════════════
# Interactive gate box drawing -> Replaced by YOLO
# ══════════════════════════════════════════════════════════════════════════════

# class _DrawState:
#     def __init__(self):
#         self.drawing = False
#         self.start   = (0, 0)
#         self.rect    = None


# def interactive_gate_draw(image: np.ndarray) -> tuple:
#     """
#     Draw ONE bounding box around the full gate.
#     Returns (x, y, w, h) — same format YOLO outputs.

#     ┌─ SWAP POINT ──────────────────────────────────────────────────────────┐
#     │  Replace with YOLO detection:                                         │
#     │      results  = yolo_model(image)                                     │
#     │      x, y, w, h = results[0].boxes.xywh[gate_class_idx]              │
#     │      return (int(x), int(y), int(w), int(h))                         │
#     └───────────────────────────────────────────────────────────────────────┘
#     """
#     state = _DrawState()

#     def mouse_cb(event, x, y, flags, _param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             state.drawing = True
#             state.start   = (x, y)
#         elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
#             state.rect = (*state.start, x, y)
#         elif event == cv2.EVENT_LBUTTONUP:
#             state.drawing = False
#             state.rect    = (*state.start, x, y)

#     win = "Gate Box — drag around full gate, SPACE/ENTER to confirm | Q=quit"
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback(win, mouse_cb)

#     print("\n  Draw a box around the ENTIRE gate (both posts + crossbar)")
#     print("  Left-drag = draw  |  SPACE/ENTER = confirm  |  Q = quit\n")

#     while True:
#         display = image.copy()
#         if state.rect:
#             x1, y1, x2, y2 = state.rect
#             cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_GATE, 2)
#             cv2.putText(display, "GATE", (x1+4, y1-6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GATE, 2)
#         cv2.putText(display,
#                     "Draw box around FULL GATE  (SPACE=confirm  Q=quit)",
#                     (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
#         cv2.imshow(win, display)

#         key = cv2.waitKey(16) & 0xFF
#         if key in (13, 32) and state.rect:
#             x1, y1, x2, y2 = state.rect
#             x1, x2 = sorted([x1, x2])
#             y1, y2 = sorted([y1, y2])
#             roi = (x1, y1, max(1, x2-x1), max(1, y2-y1))
#             print(f"  → Gate box confirmed: x={x1} y={y1} w={x2-x1} h={y2-y1}")
#             cv2.destroyWindow(win)
#             return roi
#         elif key == ord('q'):
#             cv2.destroyAllWindows()
#             sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
# Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def segment_black_lab(crop: np.ndarray,
                      use_percentile: bool = True, # False for raw value thresholding
                      percentile: float = 15,
                      l_threshold: int = L_THRESHOLD) -> np.ndarray:
    lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]

    # Percentile-based thresholding (set as default)
    if use_percentile:
        dynamic_threshold = np.percentile(l_ch, percentile)
        _, mask = cv2.threshold(l_ch, dynamic_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # Fixed threshold fallback — use --l-threshold to tune
        _, mask = cv2.threshold(l_ch, l_threshold, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def segment_red_lab(crop: np.ndarray,
                    a_threshold: int = A_THRESHOLD) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    a_ch = lab[:, :, 1]
    _, mask = cv2.threshold(a_ch, a_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

# ══════════════════════════════════════════════════════════════════════════════
# Blob utilities
# ══════════════════════════════════════════════════════════════════════════════

# Standard contour detection
def largest_blob(mask: np.ndarray, min_area: int = MIN_BLOB_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None, None, None

    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    blob_mask = np.zeros_like(mask)
    cv2.drawContours(blob_mask, [best], -1, 255, -1)

    # return the mask, bounding box, and some helpful stats
    return blob_mask, (x, y, w, h), dict(
        area = cv2.contourArea(best), # area of contour
        cx   = x + w / 2.0, # x-center of bounding box
        cy   = y + h / 2.0, # y-center of bounding box
        x=x, y=y, w=w, h=h,
    )

# Same idea as above, but also accounting for aspect ratio
def largest_blob_by_aspect(mask, min_area=MIN_BLOB_AREA,
                           min_aspect=0.5, max_aspect=5.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    valid = []
    for c in contours:
        # 1) Filter by area
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        # 2) Filter by aspect ratio (width/height) using predefined bounds
        if min_aspect <= aspect <= max_aspect:
            valid.append(c)

    if not valid:
        return None, None, None

    best = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    blob_mask = np.zeros_like(mask)
    cv2.drawContours(blob_mask, [best], -1, 255, -1)

    return blob_mask, (x, y, w, h), dict(
        area=cv2.contourArea(best),
        cx=x + w / 2.0, cy=y + h / 2.0,
        x=x, y=y, w=w, h=h,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Post detection — BLACK as primary signal instead of RED
# ══════════════════════════════════════════════════════════════════════════════

def detect_posts(image: np.ndarray,
                 gate_roi: tuple,
                 l_threshold: int = L_THRESHOLD,
                 a_threshold: int = A_THRESHOLD,
                 percentile: float = 15.0):
    gx, gy, gw, gh = gate_roi

    # slice the gate to isolate the black half
    crop = image[gy:gy+gh, gx:gx+gw]

    black_mask = segment_black_lab(crop)
    red_mask = np.zeros_like(black_mask)

    divider_y = gh // 2
    mid_x = gw // 2

    # LEFT post: black panel is on TOP → search upper-left quadrant
    left_zone = np.zeros_like(black_mask)
    left_zone[:divider_y, :mid_x] = 255

    # RIGHT post: black panel is on BOTTOM → search lower-right quadrant
    right_zone = np.zeros_like(black_mask)
    right_zone[divider_y:, mid_x:] = 255

    # Find largest black blob in each zone
    _, _, left_stats  = largest_blob(cv2.bitwise_and(black_mask, left_zone))
    _, _, right_stats = largest_blob(cv2.bitwise_and(black_mask, right_zone))

    # ── Translate to full-image coordinates ───────────────────────────────────
    def translate(s):
        return {**s,
                "img_cx":    s["cx"] + gx,
                "img_cy":    s["cy"] + gy,
                "img_x":     s["x"]  + gx,
                "img_y":     s["y"]  + gy,
                "divider_y": divider_y + gy}

    if left_stats:
        left_stats  = translate(left_stats)
    if right_stats:
        right_stats = translate(right_stats)

    return crop, black_mask, red_mask, left_stats, right_stats, divider_y

# ══════════════════════════════════════════════════════════════════════════════
# Alignment math
# ══════════════════════════════════════════════════════════════════════════════

def compute_alignment(left_stats, right_stats, image_width,
                      gate_roi=None, tolerance: float = TOLERANCE):
    eps = 1e-6
    cx  = image_width / 2.0

    # ── Yaw ───────────────────────────────────────────────────────────────────
    if left_stats and right_stats:
        W_L = max(left_stats["w"],  eps)
        W_R = max(right_stats["w"], eps)

        # W_R > W_L → right black panel wider → AUV angled right → ROTATE LEFT
        # W_L > W_R → left  black panel wider → AUV angled left  → ROTATE RIGHT
        width_ratio = W_R / W_L
        yaw_signal  = width_ratio - 1.0

        if abs(yaw_signal) <= tolerance:
            cmd_yaw = "HEAD-ON (YAW OK)"
        elif yaw_signal > 0:
            cmd_yaw = "ROTATE LEFT"
        else:
            cmd_yaw = "ROTATE RIGHT"
    else:
        width_ratio = yaw_signal = None
        cmd_yaw = "ROTATE LEFT" if right_stats is None else "ROTATE RIGHT"

    # Lateral: Compare the center of the gate post to the center of the frame
    if gate_roi:
        gx, gy, gw, gh = gate_roi
        gate_mid_x = gx + gw / 2.0
    elif left_stats and right_stats:
        gate_mid_x = (left_stats["img_cx"] + right_stats["img_cx"]) / 2.0
    elif left_stats: # approximate if only the left post is found
        gate_mid_x = left_stats["img_cx"]
    elif right_stats: # approximate if only the right post is found
        gate_mid_x = right_stats["img_cx"]
    else:
        gate_mid_x = cx

    lateral    = gate_mid_x - cx
    lat_thresh = cx * tolerance

    if abs(lateral) <= lat_thresh:
        cmd_strafe = "CENTERED"
    elif lateral > 0:
        cmd_strafe = "STRAFE RIGHT"
    else:
        cmd_strafe = "STRAFE LEFT"

    both_ok = (yaw_signal is not None
               and abs(yaw_signal) <= tolerance
               and abs(lateral)    <= lat_thresh)
    status = "HEAD-ON ✓" if both_ok else "ALIGNING..."

    return dict(
        yaw_signal  = yaw_signal,
        width_ratio = width_ratio,
        lateral     = lateral,
        gate_mid_x  = gate_mid_x,
        cmd_yaw     = cmd_yaw,
        cmd_strafe  = cmd_strafe,
        status      = status,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation Stuff
# ══════════════════════════════════════════════════════════════════════════════

def annotate_image(canvas, gate_roi, left_stats, right_stats,
                   aln, divider_y, black_mask, red_mask):
    """
    Draw gate box, divider line, zone boundaries, and post detections.
    Thin boxes = raw zone blob (always shown for diagnostics).
    Thick boxes with ✓ = confirmed detections used for alignment.
    """
    out = canvas.copy()
    gx, gy, gw, gh = gate_roi
    mid_x = gw // 2

    # ── Gate box ──────────────────────────────────────────────────────────────
    cv2.rectangle(out, (gx, gy), (gx+gw, gy+gh), COLOR_GATE, 2)
    cv2.putText(out, "GATE (YOLO)", (gx+4, gy-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GATE, 1, cv2.LINE_AA)

    # ── Divider line ──────────────────────────────────────────────────────────
    dy = divider_y + gy
    cv2.line(out, (gx, dy), (gx+gw, dy), (0, 255, 255), 2)
    cv2.putText(out, "divider", (gx+4, dy-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ── Vertical midpoint ─────────────────────────────────────────────────────
    cv2.line(out, (gx+mid_x, gy), (gx+mid_x, gy+gh), (60, 60, 60), 1)

    # ── Zone labels ───────────────────────────────────────────────────────────
    cv2.putText(out, "L-black zone", (gx+4, gy+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_LEFT, 1, cv2.LINE_AA)
    cv2.putText(out, "R-black zone", (gx+mid_x+4, dy+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RIGHT, 1, cv2.LINE_AA)

    # ── Raw diagnostic blobs (thin box — always drawn) ────────────────────────
    left_zone_mask = np.zeros((gh, gw), np.uint8)
    left_zone_mask[:divider_y, :mid_x] = 255
    right_zone_mask = np.zeros((gh, gw), np.uint8)
    right_zone_mask[divider_y:, mid_x:] = 255

    _, _, raw_left  = largest_blob(cv2.bitwise_and(black_mask, left_zone_mask))
    _, _, raw_right = largest_blob(cv2.bitwise_and(black_mask, right_zone_mask))

    for raw, color, label in [
        (raw_left,  COLOR_LEFT,  "L-blk?"),
        (raw_right, COLOR_RIGHT, "R-blk?"),
    ]:
        if raw is None:
            continue
        rx = raw["x"] + gx
        ry = raw["y"] + gy
        cv2.rectangle(out, (rx, ry), (rx+raw["w"], ry+raw["h"]), color, 1)
        cv2.putText(out, f"{label} {raw['w']}×{raw['h']}px",
                    (rx+2, ry-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, color, 1, cv2.LINE_AA)

    # ── Confirmed post boxes (thick) ──────────────────────────────────────────
    for stats, color, label in [
        (left_stats,  COLOR_LEFT,  "L-POST ✓"),
        (right_stats, COLOR_RIGHT, "R-POST ✓"),
    ]:
        if stats is None:
            continue
        ix, iy, iw, ih = (stats["img_x"], stats["img_y"],
                          stats["w"],     stats["h"])
        cv2.rectangle(out, (ix, iy), (ix+iw, iy+ih), color, 3)
        cv2.putText(out, f"{label} {iw}×{ih}px",
                    (ix+2, iy-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (int(stats["img_cx"]), int(stats["img_cy"])),
                       color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

    # ── Image center vs gate center ───────────────────────────────────────────
    img_cx = canvas.shape[1] // 2
    cv2.line(out, (img_cx, 0), (img_cx, canvas.shape[0]), (40, 40, 40), 1)
    gm = int(aln["gate_mid_x"])
    cv2.line(out, (gm, gy), (gm, gy+gh), (0, 200, 255), 2)
    cv2.putText(out, "gate mid", (gm+4, gy+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    return out

def draw_dashboard(canvas: np.ndarray, aln: dict) -> None:
    """Semi-transparent navigation dashboard stamped onto canvas in-place."""
    h, w = canvas.shape[:2]
    panel_h = 170
    y0      = h - panel_h

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), COLOR_DASH_BG, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    ok_col   = (50,  220,  50)
    warn_col = (0,   120, 255)
    info_col = (0,   200, 255)
    dim_col  = (160, 160, 160)

    def put(text, y, color, scale=0.65, thickness=1):
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    status_col = ok_col if "✓" in aln["status"] else warn_col
    put(f"STATUS:     {aln['status']}",     y0+28,  status_col, 0.72, 2)
    put(f"CMD YAW:    {aln['cmd_yaw']}",    y0+56,  info_col)
    put(f"CMD STRAFE: {aln['cmd_strafe']}", y0+82,  info_col)

    if aln["yaw_signal"] is not None:
        put(f"yaw_signal={aln['yaw_signal']:+.3f}  "
            f"W_ratio={aln['width_ratio']:.3f}  "
            f"(R_blk/L_blk)",                       y0+108, dim_col, 0.47)
    else:
        put("yaw_signal=N/A  (one or both posts not detected)",
                                                       y0+108, dim_col, 0.47)

    put(f"lateral={aln['lateral']:+.1f}px   "
        f"gate_mid={aln['gate_mid_x']:.1f}px",       y0+130, dim_col, 0.47)

    # Lateral bar
    bar_y   = y0 + 28
    bar_x0  = w - 220
    bar_x1  = w - 30
    bar_mid = (bar_x0 + bar_x1) // 2
    cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), (70, 70, 70), 3)
    cv2.line(canvas, (bar_mid, bar_y-8), (bar_mid, bar_y+8), (100,100,100), 1)
    norm  = np.clip(aln["lateral"] / (w / 2.0), -1.0, 1.0)
    ind_x = int(bar_mid + norm * (bar_x1 - bar_mid))
    cv2.circle(canvas, (ind_x, bar_y), 9, info_col, -1, cv2.LINE_AA)
    put("◄ LATERAL ►", y0+14, dim_col, 0.37)


# ══════════════════════════════════════════════════════════════════════════════
# Optional Threshold Sliders for Calibration and Debugging Purposes
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_thresholds(image: np.ndarray, gate_roi: tuple):
    gx, gy, gw, gh = gate_roi
    crop = image[gy:gy+gh, gx:gx+gw]

    win = "Calibrate — L=black (primary)  A=red (divider) | SPACE=confirm Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("L (black)", win, L_THRESHOLD, 255, lambda x: None)
    cv2.createTrackbar("A (red)",   win, A_THRESHOLD, 255, lambda x: None)

    print("\n[CALIBRATE] L (black): drag up until ONLY the dark gate panels are blue")
    print("[CALIBRATE] A (red):   drag up until ONLY the red panels/divider are red")
    print("            SPACE/ENTER to confirm | Q to quit\n")

    while True:
        l_thresh = cv2.getTrackbarPos("L (black)", win)
        a_thresh = cv2.getTrackbarPos("A (red)",   win)

        lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]
        a_ch = lab[:, :, 1]

        _, black_mask = cv2.threshold(l_ch, l_thresh, 255, cv2.THRESH_BINARY_INV)
        _, red_mask   = cv2.threshold(a_ch, a_thresh, 255, cv2.THRESH_BINARY)

        overlay = crop.copy()
        overlay[black_mask == 255] = (180,  30,  30)   # dark blue = black pixels
        overlay[red_mask   == 255] = (0,    80, 255)   # bright red = red pixels

        cv2.putText(overlay, f"L={l_thresh} (black)  A={a_thresh} (red)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.imshow(win, overlay)

        key = cv2.waitKey(16) & 0xFF
        if key in (13, 32):
            cv2.destroyWindow(win)
            print(f"  → L_THRESHOLD={l_thresh}  A_THRESHOLD={a_thresh}")
            return l_thresh, a_thresh
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)


@register_perceiver(task="gate", algo="classical_orientation")
class ClassicalOrientationPerceiver(TaskPerceiver):
    """TaskPerceiver wrapper for the UR-B-Perception LAB orientation refinement.

    This algorithm refines orientation after a detector supplies a full-gate
    bounding box. Pass that box as context.metadata["gate_roi"] or
    context.metadata["box"] in (x, y, w, h) format.
    """

    def __init__(self):
        super().__init__(
            l_threshold=((0, 255), L_THRESHOLD),
            a_threshold=((0, 255), A_THRESHOLD),
            percentile=((1, 50), 15),
        )

    def predict(self, frame: np.ndarray, context: TaskContext | None = None) -> PerceptionOutput:
        context = context or TaskContext()
        gate_roi = context.metadata.get("gate_roi", context.metadata.get("box"))
        if gate_roi is None:
            return PerceptionOutput(
                result=DetectionResult(
                    task=self.task,
                    algo=self.algo,
                    confidence=0.0,
                    frame_id=context.frame_id,
                    raw={"valid": False, "error": "missing gate_roi metadata"},
                )
            )

        gate_roi = tuple(int(value) for value in gate_roi)
        l_threshold = int(context.tunables.get("l_threshold", L_THRESHOLD))
        a_threshold = int(context.tunables.get("a_threshold", A_THRESHOLD))
        percentile = float(context.tunables.get("percentile", 15))

        enhanced = enhance_underwater(frame)
        crop, black_mask, red_mask, left_stats, right_stats, divider_y = detect_posts(
            enhanced,
            gate_roi,
            l_threshold=l_threshold,
            a_threshold=a_threshold,
            percentile=percentile,
        )
        alignment = compute_alignment(left_stats, right_stats, enhanced.shape[1], gate_roi=gate_roi)
        valid = left_stats is not None and right_stats is not None
        target_x = float(alignment["gate_mid_x"] / enhanced.shape[1])

        debug_frames = {}
        if context.debug:
            annotated = annotate_image(
                enhanced,
                gate_roi,
                left_stats,
                right_stats,
                alignment,
                divider_y,
                black_mask,
                red_mask,
            )
            draw_dashboard(annotated, alignment)
            debug_frames = {
                "annotated": annotated,
                "black_mask": black_mask,
                "red_mask": red_mask,
                "crop": crop,
            }

        return PerceptionOutput(
            result=DetectionResult(
                task=self.task,
                algo=self.algo,
                target_x=target_x,
                yaw_error=alignment["yaw_signal"],
                confidence=1.0 if valid else 0.0,
                frame_id=context.frame_id,
                raw={
                    "valid": valid,
                    "alignment": alignment,
                    "gate_roi": gate_roi,
                    "left_stats": left_stats,
                    "right_stats": right_stats,
                },
            ),
            debug_frames=debug_frames,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Gate alignment — LAB black panel primary segmentation")
    
    # Required Arguments
    p.add_argument("--img",         required=True,
                   help="Path to input image")
    p.add_argument("--box", type=int, nargs=4, required=True,
               metavar=("X", "Y", "W", "H"),
               help="YOLO gate bounding box as top-left x y w h")

    # Optional Arguments for Tuning
    p.add_argument("--percentile", type=float, default=15.0,
               help="Darkest N percent of pixels treated as black (default 15.0)")
    p.add_argument("--calibrate",   action="store_true",
                   help="Open dual threshold calibration window first")
    p.add_argument("--l-threshold", type=int, default=L_THRESHOLD,
                   help=f"LAB L threshold for black (default {L_THRESHOLD})")
    p.add_argument("--a-threshold", type=int, default=A_THRESHOLD,
                   help=f"LAB A threshold for red   (default {A_THRESHOLD})")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # ── Load & enhance ────────────────────────────────────────────────────────
    raw = cv2.imread(args.img)
    if raw is None:
        sys.exit(f"[ERROR] Cannot load image: {args.img}")
    img = enhance_underwater(raw)

    # ── Step 1: Draw gate box ─────────────────────────────────────────────────
    # print("\n" + "="*54)
    # print("  STEP 1 — Draw the YOLO gate bounding box")
    # print("="*54)
    # gate_roi = interactive_gate_draw(img)
    gate_roi = tuple(args.box)

    # ── Step 2: Optional calibration ──────────────────────────────────────────
    l_thresh = args.l_threshold
    a_thresh = args.a_threshold
    if args.calibrate:
        print("\n" + "="*54)
        print("  STEP 2 — Calibrate thresholds")
        print("="*54)
        l_thresh, a_thresh = calibrate_thresholds(img, gate_roi)

    # ── Step 3: Detect posts ──────────────────────────────────────────────────
    print("\nDetecting posts via BLACK panel segmentation (primary)...")
    crop, black_mask, red_mask, left_stats, right_stats, divider_y = \
        detect_posts(img, gate_roi, l_thresh, a_thresh, percentile=args.percentile)

    if left_stats is None:
        print("  [WARN] Left post not detected  (no black blob in upper-left zone)")
    else:
        print(f"  Left  post: {left_stats['w']}×{left_stats['h']}px  "
              f"area={left_stats['area']:.0f}  "
              f"center=({left_stats['img_cx']:.0f},{left_stats['img_cy']:.0f})")

    if right_stats is None:
        print("  [WARN] Right post not detected (no black blob in lower-right zone)")
    else:
        print(f"  Right post: {right_stats['w']}×{right_stats['h']}px  "
              f"area={right_stats['area']:.0f}  "
              f"center=({right_stats['img_cx']:.0f},{right_stats['img_cy']:.0f})")

    # ── Step 4: Alignment math ────────────────────────────────────────────────
    aln = compute_alignment(left_stats, right_stats, img.shape[1],
                            gate_roi=gate_roi)

    print("\n" + "="*54)
    print("  GATE ALIGNMENT")
    print("="*54)
    print(f"  STATUS       : {aln['status']}")
    print(f"  CMD YAW      : {aln['cmd_yaw']}")
    print(f"  CMD STRAFE   : {aln['cmd_strafe']}")
    if aln["yaw_signal"] is not None:
        print(f"  yaw_signal   : {aln['yaw_signal']:+.4f}  (tol ± {TOLERANCE})")
        print(f"  width_ratio  : {aln['width_ratio']:.4f}  (R/L black panel width)")
    print(f"  lateral      : {aln['lateral']:+.1f} px  (+= gate right of centre)")
    print("="*54 + "\n")

    # ── Step 5: Visualise ─────────────────────────────────────────────────────
    canvas = annotate_image(img, gate_roi, left_stats, right_stats,
                            aln, divider_y, black_mask, red_mask)
    draw_dashboard(canvas, aln)

    # Color-coded masks for display
    gx, gy, gw, gh = gate_roi
    mid_x = gw // 2

    black_color = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
    red_color   = cv2.cvtColor(red_mask,   cv2.COLOR_GRAY2BGR)

    # Left zone = green, right zone = blue
    black_color[:divider_y, :mid_x][black_mask[:divider_y, :mid_x] == 255] = COLOR_LEFT
    black_color[divider_y:, mid_x:][black_mask[divider_y:, mid_x:] == 255] = COLOR_RIGHT
    # Remaining black pixels (outside zones) shown as grey
    grey_mask = black_mask.copy()
    grey_mask[:divider_y, :mid_x] = 0
    grey_mask[divider_y:, mid_x:] = 0
    black_color[grey_mask == 255] = (100, 100, 100)

    fig, axes = plt.subplots(1, 2, figsize=(26, 7))

    axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"Annotated  |  YAW: {aln['cmd_yaw']}   STRAFE: {aln['cmd_strafe']}",
        fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(black_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title("BLACK mask (L channel) — PRIMARY\n"
                       "Green=left zone  Blue=right zone  Grey=outside zones")
    axes[1].axvline(x=mid_x,     color="white", linewidth=1, linestyle="--")
    axes[1].axhline(y=divider_y, color="cyan",  linewidth=1, linestyle="--")
    axes[1].axis("off")

    plt.suptitle(
        f"STATUS: {aln['status']}   |   "
        f"yaw_signal = {aln['yaw_signal']:+.3f}   "
        f"lateral = {aln['lateral']:+.1f} px"
        if aln["yaw_signal"] is not None else
        f"STATUS: {aln['status']}   |   lateral = {aln['lateral']:+.1f} px",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
