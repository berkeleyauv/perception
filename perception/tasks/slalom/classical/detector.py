from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class PassSide(str, Enum):
    """Which side of the red pipe the AUV should pass on."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class PipeDetection:
    """A vertical pipe candidate in image coordinates."""

    label: str
    x: int
    y: int
    width: int
    height: int
    area: float
    score: float
    aspect_ratio: float = 0.0
    fill_ratio: float = 0.0
    width_stability: float = 0.0
    vertical_alignment: float = 0.0
    contrast_strength: float = 0.0

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def bottom_y(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class SlalomEstimate:
    """High-level target for task planning or visual servoing."""

    target_x: float
    target_y: float
    yaw_error: float
    confidence: float
    red_pipe: PipeDetection | None
    left_white_pipe: PipeDetection | None
    right_white_pipe: PipeDetection | None

    @property
    def valid(self) -> bool:
        return self.red_pipe is not None and self.confidence > 0


@dataclass(frozen=True)
class WhitePoleEstimate:
    """Target estimate for proxy footage with one white pole."""

    target_x: float
    target_y: float
    yaw_error: float
    confidence: float
    white_pipe: PipeDetection | None
    target_side: PassSide | None = None
    red_pipe: PipeDetection | None = None

    @property
    def valid(self) -> bool:
        return self.white_pipe is not None and self.confidence > 0


@dataclass(frozen=True)
class DetectorConfig:
    red_low_1: tuple[int, int, int] = (0, 60, 40)
    red_high_1: tuple[int, int, int] = (15, 255, 255)
    red_low_2: tuple[int, int, int] = (160, 60, 40)
    red_high_2: tuple[int, int, int] = (179, 255, 255)
    white_low: tuple[int, int, int] = (0, 0, 120)
    white_high: tuple[int, int, int] = (179, 90, 255)
    min_area: int = 120
    min_height_ratio: float = 0.16
    min_aspect_ratio: float = 2.4
    max_aspect_width_ratio: float = 0.22
    morph_kernel_size: int = 5

    # Contrast-based segmentation (see segment_contrast). At range underwater,
    # red pigment is heavily attenuated, so a "red" pole reads as near-black
    # rather than red-hued and is indistinguishable from a shadowed white pole
    # by hue alone. Poles instead show up as vertical strips that are locally
    # much darker (red pole) or much brighter (white pole) than the
    # surrounding water, so contrast against a local background estimate
    # stands in for color.
    contrast_bg_kernel: int = 61
    dark_diff_thresh: int = 18
    bright_diff_thresh: int = 18
    # Sun-rippled water surface also reads as locally dark, producing wide,
    # blobby contours that pass the generic pole-shape filter. A real pole is
    # much narrower relative to its height than a patch of glare (observed
    # aspect ratio ~9-12 for real poles vs. ~2.5-6 for glare blobs in test
    # footage), so dark/red contrast candidates need a stricter aspect ratio
    # than the hue-based default.
    dark_min_aspect_ratio: float = 7.0
    red_pair_min_height_ratio: float = 0.22
    # Candidate-shape classification. These features are intentionally simple
    # and inspectable: true PVC pipes are thin, vertically aligned, and keep a
    # fairly stable mask width from top to bottom. Glare and shadows may be
    # high-contrast, but they usually bulge, smear, or tilt.
    min_width_stability: float = 0.35
    min_vertical_alignment: float = 0.65


class SlalomDetector:
    """Detect red/white slalom pipes and estimate the desired steering target."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def detect(
        self,
        frame_bgr: np.ndarray,
        pass_side: PassSide = PassSide.LEFT,
        method: str = "hsv",
    ) -> SlalomEstimate:
        """Detect the red/white pipe triplet and estimate a steering target.

        method="hsv" segments by color threshold. method="contrast" segments by
        local brightness contrast instead -- use it when red pigment is too
        attenuated by water for hue to separate red from white/background (see
        segment_contrast()).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty BGR image")

        height, width = frame_bgr.shape[:2]
        red_mask, white_mask = (
            self.segment_contrast(frame_bgr)
            if method == "contrast"
            else self.segment(frame_bgr)
        )
        red_min_aspect_ratio = (
            self.config.dark_min_aspect_ratio if method == "contrast" else None
        )

        red_pipes = self._find_pipe_candidates(
            red_mask,
            "red",
            width,
            height,
            min_aspect_ratio=red_min_aspect_ratio,
            contrast_frame=frame_bgr,
        )
        white_pipes = self._find_pipe_candidates(
            white_mask, "white", width, height, contrast_frame=frame_bgr
        )

        red_pipe = self._choose_red_pipe(red_pipes, width)
        left_white_pipe, right_white_pipe = self._choose_adjacent_whites(
            red_pipe, white_pipes
        )

        return self._estimate_target(
            image_width=width,
            image_height=height,
            pass_side=pass_side,
            red_pipe=red_pipe,
            left_white_pipe=left_white_pipe,
            right_white_pipe=right_white_pipe,
        )

    def detect_white_pole_target(
        self,
        frame_bgr: np.ndarray,
        target_side: PassSide | None = PassSide.LEFT,
        offset_ratio: float = 0.18,
        method: str = "hsv",
    ) -> WhitePoleEstimate:
        """Detect one white vertical pole and place a target left or right of it.

        If target_side is None, infer the target side from image position: a
        pole left of center gets a right-side target, and a pole right of
        center gets a left-side target. With only one visible white pole, this
        is a proxy for steering back toward the channel interior.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty BGR image")

        height, width = frame_bgr.shape[:2]
        red_mask, white_mask = (
            self.segment_contrast(frame_bgr)
            if method == "contrast"
            else self.segment(frame_bgr)
        )
        white_pipes = self._find_pipe_candidates(
            white_mask, "white", width, height, contrast_frame=frame_bgr
        )
        white_pipe = self._choose_white_proxy_pipe(white_pipes, width)

        if white_pipe is None:
            return WhitePoleEstimate(0.5, 0.5, 0.0, 0.0, None, target_side)

        red_pipe = None
        if method == "contrast":
            red_candidates = self._find_pipe_candidates(
                red_mask,
                "red",
                width,
                height,
                min_aspect_ratio=3.6,
                min_width_stability=0.12,
                min_vertical_alignment=0.45,
                contrast_frame=frame_bgr,
            )
            red_pipe = self._choose_paired_red_pipe(
                red_candidates, white_pipe, width, height
            )

        if red_pipe is not None:
            target_x_pixels = (white_pipe.center_x + red_pipe.center_x) / 2
            top_y = min(white_pipe.y, red_pipe.y)
            bottom_y = max(white_pipe.bottom_y, red_pipe.bottom_y)
            target_x = float(np.clip(target_x_pixels / width, 0.05, 0.95))
            target_y = float(np.clip(((top_y + bottom_y) / 2) / height, 0.05, 0.95))
            confidence = float(
                np.clip(0.5 * white_pipe.score + 0.5 * red_pipe.score, 0.0, 1.0)
            )
            return WhitePoleEstimate(
                target_x=target_x,
                target_y=target_y,
                yaw_error=target_x - 0.5,
                confidence=confidence,
                white_pipe=white_pipe,
                target_side=target_side,
                red_pipe=red_pipe,
            )

        if target_side is None:
            target_side = (
                PassSide.RIGHT if white_pipe.center_x < width / 2 else PassSide.LEFT
            )

        direction = -1 if target_side == PassSide.LEFT else 1
        # Apparent pole width grows as the AUV gets closer. Let it nudge the
        # target offset wider for close poles while keeping sane image bounds.
        base_offset = offset_ratio * width
        width_scaled_offset = 14.0 * white_pipe.width
        offset_pixels = float(
            np.clip(max(base_offset, width_scaled_offset), 0.12 * width, 0.32 * width)
        )
        target_x_pixels = white_pipe.center_x + direction * offset_pixels
        target_x = float(np.clip(target_x_pixels / width, 0.05, 0.95))
        target_y = float(np.clip(white_pipe.center_y / height, 0.05, 0.95))

        return WhitePoleEstimate(
            target_x=target_x,
            target_y=target_y,
            yaw_error=target_x - 0.5,
            confidence=white_pipe.score,
            white_pipe=white_pipe,
            target_side=target_side,
            red_pipe=None,
        )

    def segment_contrast(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment poles by local brightness contrast against the water background.

        A large-kernel median blur of the value channel stands in for the local
        water brightness (which varies with depth, sun glare, and camera angle).
        Pixels much darker than that local background are red-pole candidates;
        pixels much brighter are white-pole candidates. Returns (dark_mask, bright_mask).
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]

        kernel_size = self.config.contrast_bg_kernel | 1  # medianBlur requires odd
        background = cv2.medianBlur(value, kernel_size)
        diff = value.astype(np.int16) - background.astype(np.int16)

        dark_mask = (diff < -self.config.dark_diff_thresh).astype(np.uint8) * 255
        bright_mask = (diff > self.config.bright_diff_thresh).astype(np.uint8) * 255

        red_hue_1 = cv2.inRange(
            hsv, np.array(self.config.red_low_1), np.array(self.config.red_high_1)
        )
        red_hue_2 = cv2.inRange(
            hsv, np.array(self.config.red_low_2), np.array(self.config.red_high_2)
        )
        red_hue_mask = cv2.bitwise_or(red_hue_1, red_hue_2)
        dark_mask = cv2.bitwise_or(dark_mask, red_hue_mask)

        morph_kernel = np.ones(
            (self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8
        )
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, morph_kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, morph_kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, morph_kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, morph_kernel)

        return dark_mask, bright_mask

    def segment(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        red_1 = cv2.inRange(
            hsv, np.array(self.config.red_low_1), np.array(self.config.red_high_1)
        )
        red_2 = cv2.inRange(
            hsv, np.array(self.config.red_low_2), np.array(self.config.red_high_2)
        )
        red_mask = cv2.bitwise_or(red_1, red_2)

        white_mask = cv2.inRange(
            hsv, np.array(self.config.white_low), np.array(self.config.white_high)
        )

        kernel = np.ones(
            (self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8
        )
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        return red_mask, white_mask

    def annotate(self, frame_bgr: np.ndarray, estimate: SlalomEstimate) -> np.ndarray:
        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]

        for pipe in [
            estimate.left_white_pipe,
            estimate.red_pipe,
            estimate.right_white_pipe,
        ]:
            if pipe is None:
                continue
            color = (0, 0, 255) if pipe.label == "red" else (255, 255, 255)
            cv2.rectangle(
                annotated,
                (pipe.x, pipe.y),
                (pipe.x + pipe.width, pipe.y + pipe.height),
                color,
                2,
            )
            cv2.putText(
                annotated,
                f"{pipe.label}:{pipe.score:.2f}",
                (pipe.x, max(20, pipe.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if estimate.valid:
            target = (int(estimate.target_x * width), int(estimate.target_y * height))
            cv2.drawMarker(annotated, target, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.line(annotated, (width // 2, height), target, (0, 255, 255), 2)

        return annotated

    def annotate_white_pole(
        self, frame_bgr: np.ndarray, estimate: WhitePoleEstimate
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]

        if estimate.white_pipe is not None:
            pipe = estimate.white_pipe
            cv2.rectangle(
                annotated,
                (pipe.x, pipe.y),
                (pipe.x + pipe.width, pipe.y + pipe.height),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"white:{pipe.score:.2f}",
                (pipe.x, max(20, pipe.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if estimate.red_pipe is not None:
            pipe = estimate.red_pipe
            cv2.rectangle(
                annotated,
                (pipe.x, pipe.y),
                (pipe.x + pipe.width, pipe.y + pipe.height),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"red/dark:{pipe.score:.2f}",
                (pipe.x, max(20, pipe.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if estimate.valid:
            target = (int(estimate.target_x * width), int(estimate.target_y * height))
            cv2.circle(annotated, target, 10, (0, 255, 255), -1)
            cv2.circle(annotated, target, 14, (0, 0, 0), 2)
            cv2.line(annotated, (width // 2, height), target, (0, 255, 255), 2)

        return annotated

    def _find_pipe_candidates(
        self,
        mask: np.ndarray,
        label: str,
        image_width: int,
        image_height: int,
        min_aspect_ratio: float | None = None,
        min_width_stability: float | None = None,
        min_vertical_alignment: float | None = None,
        contrast_frame: np.ndarray | None = None,
    ) -> list[PipeDetection]:
        min_aspect_ratio = (
            self.config.min_aspect_ratio
            if min_aspect_ratio is None
            else min_aspect_ratio
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[PipeDetection] = []
        value: np.ndarray | None = None
        background: np.ndarray | None = None
        if contrast_frame is not None:
            hsv = cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2]
            background = cv2.medianBlur(value, self.config.contrast_bg_kernel | 1)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue

            aspect_ratio = height / width
            height_ratio = height / image_height
            width_ratio = width / image_width

            if aspect_ratio < min_aspect_ratio:
                continue
            if height_ratio < self.config.min_height_ratio:
                continue
            if width_ratio > self.config.max_aspect_width_ratio:
                continue

            fill_ratio, width_stability, vertical_alignment = self._shape_features(
                mask, contour, x, y, width, height
            )
            width_stability_threshold = (
                min_width_stability
                if min_width_stability is not None
                else self.config.min_width_stability if label == "red" else 0.20
            )
            vertical_alignment_threshold = (
                min_vertical_alignment
                if min_vertical_alignment is not None
                else self.config.min_vertical_alignment if label == "red" else 0.55
            )
            if width_stability < width_stability_threshold:
                continue
            if vertical_alignment < vertical_alignment_threshold:
                continue

            contrast_strength = (
                self._contrast_strength(value, background, contour, label)
                if value is not None
                else 0.0
            )

            vertical_score = min(aspect_ratio / (10.0 if label == "red" else 8.0), 1.0)
            size_score = min(height_ratio / 0.5, 1.0)
            contrast_score = min(contrast_strength / 45.0, 1.0)
            if label == "red":
                score = (
                    0.40 * vertical_score
                    + 0.22 * width_stability
                    + 0.18 * vertical_alignment
                    + 0.12 * contrast_score
                    + 0.08 * size_score
                )
            else:
                score = (
                    0.30 * vertical_score
                    + 0.20 * width_stability
                    + 0.20 * vertical_alignment
                    + 0.18 * contrast_score
                    + 0.12 * size_score
                )

            candidates.append(
                PipeDetection(
                    label,
                    x,
                    y,
                    width,
                    height,
                    float(area),
                    float(score),
                    float(aspect_ratio),
                    float(fill_ratio),
                    float(width_stability),
                    float(vertical_alignment),
                    float(contrast_strength),
                )
            )

        return sorted(
            candidates, key=lambda pipe: (pipe.score, pipe.area), reverse=True
        )

    def _shape_features(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[float, float, float]:
        roi = mask[y : y + height, x : x + width]
        row_widths = np.count_nonzero(roi, axis=1)
        nonzero_rows = row_widths[row_widths > 0]

        if nonzero_rows.size == 0:
            return 0.0, 0.0, 0.0

        fill_ratio = float(np.count_nonzero(roi) / (width * height))
        median_width = float(np.median(nonzero_rows))
        width_spread = float(
            np.percentile(nonzero_rows, 90) - np.percentile(nonzero_rows, 10)
        )
        width_stability = float(
            np.clip(1.0 - width_spread / max(median_width, 1.0), 0.0, 1.0)
        )

        vertical_alignment = self._vertical_alignment(contour)
        return fill_ratio, width_stability, vertical_alignment

    def _vertical_alignment(self, contour: np.ndarray) -> float:
        if len(contour) < 5:
            return 1.0

        vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vertical_component = abs(float(vy)) / max(float(np.hypot(vx, vy)), 1e-6)
        return float(np.clip(vertical_component, 0.0, 1.0))

    def _contrast_strength(
        self,
        value: np.ndarray | None,
        background: np.ndarray | None,
        contour: np.ndarray,
        label: str,
    ) -> float:
        if value is None or background is None:
            return 0.0

        contour_mask = np.zeros(value.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        selected = contour_mask > 0
        if not np.any(selected):
            return 0.0

        diff = value.astype(np.int16) - background.astype(np.int16)
        candidate_diff = diff[selected]
        if label == "red":
            return float(max(0.0, -np.median(candidate_diff)))
        return float(max(0.0, np.median(candidate_diff)))

    def _choose_red_pipe(
        self, red_pipes: list[PipeDetection], image_width: int
    ) -> PipeDetection | None:
        if not red_pipes:
            return None

        image_center = image_width / 2
        return max(
            red_pipes,
            key=lambda pipe: pipe.score
            - 0.25 * abs(pipe.center_x - image_center) / image_width,
        )

    def _choose_paired_red_pipe(
        self,
        red_pipes: list[PipeDetection],
        white_pipe: PipeDetection,
        image_width: int,
        image_height: int,
    ) -> PipeDetection | None:
        if not red_pipes:
            return None

        min_separation = 0.08 * image_width
        max_separation = 0.70 * image_width
        plausible = [
            pipe
            for pipe in red_pipes
            if min_separation
            <= abs(pipe.center_x - white_pipe.center_x)
            <= max_separation
            and pipe.height >= self.config.red_pair_min_height_ratio * image_height
        ]
        if not plausible:
            return None

        white_height = max(float(white_pipe.height), 1.0)
        return max(
            plausible,
            key=lambda pipe: (
                0.45 * min(pipe.height / white_height, 1.6)
                + 0.30 * min(pipe.aspect_ratio / 8.0, 1.0)
                + 0.20 * min(pipe.contrast_strength / 60.0, 1.0)
                + 0.15 * pipe.vertical_alignment
                + 0.10 * pipe.score
                - 0.15 * abs(pipe.center_x - image_width / 2) / image_width
            ),
        )

    def _choose_adjacent_whites(
        self,
        red_pipe: PipeDetection | None,
        white_pipes: list[PipeDetection],
    ) -> tuple[PipeDetection | None, PipeDetection | None]:
        if red_pipe is None:
            return None, None

        left = [pipe for pipe in white_pipes if pipe.center_x < red_pipe.center_x]
        right = [pipe for pipe in white_pipes if pipe.center_x > red_pipe.center_x]

        left_pipe = max(
            left, key=lambda pipe: (pipe.score, pipe.center_x), default=None
        )
        right_pipe = max(
            right, key=lambda pipe: (pipe.score, -pipe.center_x), default=None
        )
        return left_pipe, right_pipe

    def _choose_white_proxy_pipe(
        self,
        white_pipes: list[PipeDetection],
        image_width: int,
    ) -> PipeDetection | None:
        if not white_pipes:
            return None

        image_center = image_width / 2
        return max(
            white_pipes,
            key=lambda pipe: pipe.score
            - 0.15 * abs(pipe.center_x - image_center) / image_width,
        )

    def _estimate_target(
        self,
        image_width: int,
        image_height: int,
        pass_side: PassSide,
        red_pipe: PipeDetection | None,
        left_white_pipe: PipeDetection | None,
        right_white_pipe: PipeDetection | None,
    ) -> SlalomEstimate:
        if red_pipe is None:
            return SlalomEstimate(
                0.5, 0.5, 0.0, 0.0, None, left_white_pipe, right_white_pipe
            )

        desired_white = (
            left_white_pipe if pass_side == PassSide.LEFT else right_white_pipe
        )
        fallback_offset = 0.22 * image_width

        if desired_white is not None:
            target_x_pixels = (red_pipe.center_x + desired_white.center_x) / 2
            top_y = min(red_pipe.y, desired_white.y)
            bottom_y = max(red_pipe.bottom_y, desired_white.bottom_y)
            confidence = 0.9
        else:
            direction = -1 if pass_side == PassSide.LEFT else 1
            target_x_pixels = red_pipe.center_x + direction * fallback_offset
            top_y = red_pipe.y
            bottom_y = red_pipe.bottom_y
            confidence = 0.45

        target_x = float(np.clip(target_x_pixels / image_width, 0.05, 0.95))
        target_y = float(np.clip(((top_y + bottom_y) / 2) / image_height, 0.05, 0.95))
        yaw_error = target_x - 0.5

        return SlalomEstimate(
            target_x=target_x,
            target_y=target_y,
            yaw_error=yaw_error,
            confidence=confidence,
            red_pipe=red_pipe,
            left_white_pipe=left_white_pipe,
            right_white_pipe=right_white_pipe,
        )
