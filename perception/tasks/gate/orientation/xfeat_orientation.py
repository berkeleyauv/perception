import sys
import time
import cv2
import threading
import numpy as np
import argparse
from pathlib import Path

from perception.registry import register_perceiver
from perception.tasks.TaskPerceiver import DetectionResult, PerceptionOutput, TaskContext, TaskPerceiver

# ══════════════════════════════════════════════════════════════════════════════
# Threaded Video Ingestion (Unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class FrameGrabber(threading.Thread):
    def __init__(self, index, w, h):
        super().__init__()
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        ret, self.frame = self.cap.read()
        if not ret or self.frame is None:
            sys.exit(f"[ERROR] Unable to access camera device index: {index}")
        self._lock = threading.Lock()
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.frame = frame
            time.sleep(0.005)

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

# ══════════════════════════════════════════════════════════════════════════════
# Perception Subsystem Class
# ══════════════════════════════════════════════════════════════════════════════
class GateDetector:

    def __init__(self, ref_image_path, width=640, height=480):
        self.WIDTH = width
        self.HEIGHT = height
        self.MAX_KPTS = 2048
        self.TOLERANCE = 0.08
        self.RANSAC_THR = 4.0
        self.MIN_INLIERS = 30
        
        # Load and prep reference image
        ref_frame = cv2.imread(ref_image_path)
        if ref_frame is None:
            raise FileNotFoundError(f"[ERROR] Static reference path empty: {ref_image_path}")
            
        ref_frame = cv2.resize(ref_frame, (self.WIDTH, self.HEIGHT))
        self.ref_enh = self.enhance_underwater(ref_frame)
        self.ref_h, self.ref_w = self.ref_enh.shape[:2]

        print("[INFO] Initializing PyTorch XFeat Backend Context...")
        import torch
        from perception.tasks.gate.orientation.modules.xfeat import XFeat

        self.torch = torch
        self.xfeat = XFeat(top_k=self.MAX_KPTS)

        print("[INFO] Pre-computing static reference structural maps...")
        with self.torch.no_grad():
            self.ref_precomp = self.xfeat.detectAndCompute(self.ref_enh)[0]

    def enhance_underwater(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process_frame(self, live_raw):
        """
        Takes a raw frame, processes it, and RETURNS the alignment data for Controls.
        """
        live_enh = self.enhance_underwater(live_raw)
        
        with self.torch.no_grad():
            live_features = self.xfeat.detectAndCompute(live_enh)[0]
            idx0, idx1 = self.xfeat.match(self.ref_precomp['descriptors'], live_features['descriptors'], 0.82)
            points_ref = self.ref_precomp['keypoints'][idx0].cpu().numpy()
            points_live = live_features['keypoints'][idx1].cpu().numpy()

        # Default empty state if detection fails
        alignment_data = {
            "valid": False,
            "yaw_signal": 0.0,
            "lateral_shift": 0.0,
            "warped_corners": None,
            "error_msg": "CRITICAL DETECTOR FAULT: NO KEY CORRELATIONS"
        }

        if len(points_ref) >= 10:
            H, inlier_mask = cv2.findHomography(
                points_ref, points_live,
                cv2.USAC_MAGSAC, self.RANSAC_THR, maxIters=800, confidence=0.99
            )

            if H is not None:
                inlier_mask = inlier_mask.flatten() > 0
                if inlier_mask.sum() >= self.MIN_INLIERS:
                    
                    #Calculate the metrics
                    corners_ref = np.array([[0, 0], [self.ref_w, 0], [self.ref_w, self.ref_h], [0, self.ref_h]], dtype=np.float32).reshape(-1, 1, 2)
                    warped_corners = cv2.perspectiveTransform(corners_ref, H)
                    
                    tl, tr, br, bl = warped_corners[:, 0, :]
                    h_left = np.linalg.norm(tl - bl)
                    h_right = np.linalg.norm(tr - br)
                    yaw_signal = np.log(h_left / max(h_right, 1e-6))
                    
                    gate_mid = np.mean(warped_corners[:, 0, 0])
                    lateral_shift = gate_mid - (self.WIDTH / 2.0)

                    # Populate the successful data packet to return
                    alignment_data = {
                        "valid": True,
                        "yaw_signal": yaw_signal,
                        "lateral_shift": lateral_shift,
                        "warped_corners": warped_corners,
                        "valid_points": points_live[inlier_mask],
                        "error_msg": ""
                    }
                else:
                    alignment_data["error_msg"] = "LOW INLIER CONSENSUS COUNTS"
            else:
                alignment_data["error_msg"] = "HOMOGRAPHY MATRIX DIED"

        # WE NOW RETURN THIS DICTIONARY TO WHATEVER SCRIPT CALLED THIS FUNCTION!
        return alignment_data


@register_perceiver(task="gate", algo="xfeat_orientation")
class XFeatOrientationPerceiver(TaskPerceiver):
    """Optional XFeat homography orientation perceiver from UR-B-Perception."""

    def __init__(self, ref_image_path: str | None = None, width=640, height=480):
        super().__init__()
        if ref_image_path is None:
            ref_image_path = str(Path(__file__).with_name("Gate1.png"))
        self.detector = GateDetector(ref_image_path=ref_image_path, width=width, height=height)

    def predict(self, frame: np.ndarray, context: TaskContext | None = None) -> PerceptionOutput:
        context = context or TaskContext()
        data = self.detector.process_frame(frame)

        debug_frames = {}
        if context.debug:
            annotated = frame.copy()
            if data["valid"]:
                for point in data["valid_points"]:
                    cv2.circle(annotated, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
                cv2.polylines(
                    annotated,
                    [np.int32(data["warped_corners"])],
                    True,
                    (0, 165, 255),
                    3,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    annotated,
                    data["error_msg"],
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 120, 255),
                    2,
                )
            debug_frames["annotated"] = annotated

        return PerceptionOutput(
            result=DetectionResult(
                task=self.task,
                algo=self.algo,
                target_x=(
                    float((data["lateral_shift"] + self.detector.WIDTH / 2.0) / self.detector.WIDTH)
                    if data["valid"]
                    else None
                ),
                yaw_error=float(data["yaw_signal"]) if data["valid"] else None,
                confidence=1.0 if data["valid"] else 0.0,
                frame_id=context.frame_id,
                raw=data,
            ),
            debug_frames=debug_frames,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Standalone Testing Module (Runs only if executed directly)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", help="Path to reference gate image") 
    args = p.parse_args()
    
    detector = GateDetector(ref_image_path=args.ref)
    
    grabber = FrameGrabber(index=0, w=detector.WIDTH, h=detector.HEIGHT)
    grabber.start()
    time.sleep(0.5) 
    
    cv2.namedWindow("RoboSub Alignment Dashboard", cv2.WINDOW_AUTOSIZE)
    
    while True:
        live_raw = grabber.get_frame()
        if live_raw is None: continue
            
        canvas = live_raw.copy()
        
        # ACTUALLY CALL THE FUNCTION AND GET THE RETURNED DATA
        data = detector.process_frame(live_raw)
        
        if data["valid"]:
            # Draw points and poly
            for pt in data["valid_points"]:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
            cv2.polylines(canvas, [np.int32(data["warped_corners"])], True, (0, 165, 255), 3, cv2.LINE_AA)
            
            # Print the data to terminal => what controls will see
            print(f"Controls Data -> Yaw: {data['yaw_signal']:.3f} | Strafe: {data['lateral_shift']:.1f}")
        else:
            cv2.putText(canvas, data["error_msg"], (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 120, 255), 2)
            
        cv2.imshow("RoboSub Alignment Dashboard", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabber.stop()
    cv2.destroyAllWindows()
