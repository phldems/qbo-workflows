"""
Check Detection Module

Detects individual checks on scanned pages before OCR processing.
Handles both single-check and multi-check scenarios.

Uses per-group outlier filtering for robust detection:
1. Connected component analysis to find all text/ink regions
2. Gap detection to identify check boundaries
3. MICR line merging to combine separated check components
4. Per-group outlier filtering to remove noise while preserving check content
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger


class CheckDetector:
    """
    Detects and isolates individual checks on scanned pages.

    Strategy:
    1. Use connected components to find all ink regions
    2. Apply gap detection with IQR to identify check boundaries
    3. Merge MICR lines back into check bodies
    4. Apply per-group outlier filtering to remove noise
    5. Return cropped check images for individual processing
    """

    def __init__(
        self,
        iqr_factor: float = 1.5,
        outlier_threshold: float = 2.5,
        expand_percent: float = 4.0,
        dpi: int = 300,
        min_check_dimension_px: int = 300,
        min_component_area_px: int = 80,
        micr_line_max_height_px: int = 450,
        micr_line_max_gap_px: int = 600,
        base_detection_confidence: float = 50.0,
        vertical_overlap_margin_px: int = 40,
    ):
        """
        Initialize check detector with all parameters configurable via Settings/.env.

        All parameters can be overridden via .env file:
        - CHECK_DETECTOR_IQR_FACTOR
        - CHECK_DETECTOR_OUTLIER_THRESHOLD
        - CHECK_DETECTOR_EXPAND_PERCENT
        - CHECK_DETECTOR_DPI
        - self.min_check_dimension_px
        - self.min_component_area_px
        - self.micr_line_max_height_px
        - self.micr_line_max_gap_px
        - self.base_detection_confidence

        Args:
            iqr_factor: IQR multiplier for gap detection (default: 1.5)
            outlier_threshold: Z-score threshold for outlier filtering (default: 2.5)
            expand_percent: Percentage to expand bounding boxes (default: 4.0)
            dpi: DPI used for image conversion (default: 300)
            min_check_dimension_px: Minimum check dimension in pixels (default: 300)
            min_component_area_px: Minimum component area to avoid noise (default: 80)
            micr_line_max_height_px: Max MICR line height in pixels (default: 450)
            micr_line_max_gap_px: Max gap to MICR line in pixels (default: 600)
            base_detection_confidence: Base confidence score (default: 50.0)
        """
        self.iqr_factor = iqr_factor
        self.outlier_threshold = outlier_threshold
        self.expand_percent = expand_percent
        self.dpi = dpi
        self.min_check_dimension_px = min_check_dimension_px
        self.min_component_area_px = min_component_area_px
        self.micr_line_max_height_px = micr_line_max_height_px
        self.micr_line_max_gap_px = micr_line_max_gap_px
        self.base_detection_confidence = base_detection_confidence
        self.vertical_overlap_margin_px = vertical_overlap_margin_px

        logger.info(
            f"Initialized CheckDetector (all params from .env): "
            f"iqr_factor={iqr_factor}, outlier_threshold={outlier_threshold}, "
            f"expand={expand_percent}%, dpi={dpi}, min_check_dim={min_check_dimension_px}px"
        )

    def detect_checks(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Detect individual checks on a page image using per-group outlier filtering.

        Args:
            image: Page image as numpy array (RGB format)

        Returns:
            List of dicts with:
                - bbox: (x, y, width, height) bounding box
                - image: Cropped check image
                - confidence: Detection confidence (0-100)
                - area: Bounding box area
                - components_used: Number of components in the check
                - outliers_removed: Number of outliers filtered out
        """
        logger.info(f"Detecting checks on image (shape: {image.shape})")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Threshold: ink -> 255, background -> 0
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        H, W = th.shape

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            th, connectivity=8
        )

        # Collect components
        components = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            cx, cy = centroids[label]

            # Skip border-touching
            if x == 0 or y == 0 or x + w == W or y + h == H:
                continue
            # Skip tiny noise
            if area < self.min_component_area_px:
                continue

            components.append((x, y, w, h, cx, cy))

        if not components:
            logger.info("No components found")
            return []

        logger.info(f"Total filtered components: {len(components)}")

        # Sort by vertical center
        components_sorted = sorted(components, key=lambda c: c[5])  # cy

        # ===== GAP DETECTION =====
        gaps = []
        for i in range(len(components_sorted) - 1):
            _, y1, w1, h1, _, _ = components_sorted[i]
            _, y2, w2, h2, _, _ = components_sorted[i + 1]
            bottom1 = y1 + h1
            top2 = y2
            gap = top2 - bottom1
            gaps.append((gap, i))

        positive_gaps = [(g, idx) for (g, idx) in gaps if g > 0]

        # Decide how many vertical splits we need
        if not positive_gaps:
            groups = [components_sorted]  # only one check
        else:
            arr = np.array([g for g, _ in positive_gaps])
            med = np.median(arr)
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            thr = med + self.iqr_factor * iqr

            # All gaps that are "large" become separators
            split_indices = sorted(idx for (g, idx) in positive_gaps if g >= thr)

            # Fallback: if nothing passes the threshold, just take the single largest gap
            if not split_indices:
                _, max_idx = max(positive_gaps, key=lambda x: x[0])
                split_indices = [max_idx]

            boundaries = [0]
            for idx in split_indices:
                boundaries.append(idx + 1)
            boundaries.append(len(components_sorted))
            boundaries = sorted(set(boundaries))

            groups = [
                components_sorted[boundaries[i] : boundaries[i + 1]]
                for i in range(len(boundaries) - 1)
            ]

        logger.info(f"Detected {len(groups)} group(s) via gap detection")

        # ===== MERGE GROUPS THAT ARE LIKELY MICR LINES =====
        # MICR lines are typically short (<1.5in tall) and appear at the bottom of checks
        # If we find a short group that's close to a taller group, merge them
        merged_groups = []
        i = 0
        while i < len(groups):
            current_group = groups[i]

            # Calculate vertical extent of current group
            if current_group:
                current_ys = [c[1] for c in current_group]  # y positions
                current_y2s = [c[1] + c[3] for c in current_group]  # y + h
                current_height = max(current_y2s) - min(current_ys)
                current_bottom = max(current_y2s)

                # Check if next group exists and might be a MICR line
                if i + 1 < len(groups):
                    next_group = groups[i + 1]
                    if next_group:
                        next_ys = [c[1] for c in next_group]
                        next_y2s = [c[1] + c[3] for c in next_group]
                        next_height = max(next_y2s) - min(next_ys)
                        next_top = min(next_ys)

                        # If next group is short and close to current (likely MICR line)
                        gap_to_next = next_top - current_bottom
                        if (
                            next_height < self.micr_line_max_height_px
                            and gap_to_next < self.micr_line_max_gap_px
                        ):
                            logger.info(
                                f"Merging Group {i+2} (MICR line candidate, height={next_height/self.dpi:.1f}in) into Group {i+1}"
                            )
                            merged_groups.append(current_group + next_group)
                            i += 2  # Skip next group as it's now merged
                            continue

            merged_groups.append(current_group)
            i += 1

        groups = merged_groups
        logger.info(f"After MICR merge: {len(groups)} group(s)")

        # ===== APPLY OUTLIER FILTERING PER GROUP =====
        results = []

        for group_idx, group in enumerate(groups):
            logger.debug(f"Group {group_idx + 1}: {len(group)} components")

            if len(group) == 0:
                continue

            # Calculate centroid statistics for THIS GROUP only
            cy_values = [c[5] for c in group]  # c[5] is cy
            cy_mean = np.mean(cy_values)
            cy_std = np.std(cy_values)

            # Filter outliers within THIS GROUP
            non_outliers = []
            outliers = []

            for comp in group:
                x, y, w, h, cx, cy = comp
                z_score = abs(cy - cy_mean) / (cy_std + 1e-10)

                if z_score <= self.outlier_threshold:
                    non_outliers.append(comp)
                else:
                    outliers.append((comp, z_score))

            logger.debug(
                f"Group {group_idx + 1}: {len(non_outliers)} non-outliers, {len(outliers)} outliers removed"
            )

            # If all filtered out, use original group
            if not non_outliers:
                logger.warning(
                    f"Group {group_idx + 1}: All components filtered! Using original group."
                )
                non_outliers = group

            # Calculate union box for this group
            xs = [c[0] for c in non_outliers]  # x
            ys = [c[1] for c in non_outliers]  # y
            x2s = [c[0] + c[2] for c in non_outliers]  # x + w
            y2s = [c[1] + c[3] for c in non_outliers]  # y + h

            x0 = int(min(xs))
            y0 = int(min(ys))
            x1 = int(max(x2s))
            y1 = int(max(y2s))
            w = x1 - x0
            h = y1 - y0

            # Apply expansion
            if self.expand_percent > 0:
                expand_w = int(w * self.expand_percent / 100)
                expand_h = int(h * self.expand_percent / 100)

                x0 = max(0, x0 - expand_w)
                y0 = max(0, y0 - expand_h)
                x1 = min(W, x1 + expand_w)
                y1 = min(H, y1 + expand_h)
                w = x1 - x0
                h = y1 - y0

            # Filter out tiny boxes (smaller than minimum dimension)
            if w < self.min_check_dimension_px:
                logger.debug(
                    f"Group {group_idx + 1}: Box too small (< 1 inch wide), skipping"
                )
                continue

            # Crop check image
            check_image = image[y0:y1, x0:x1].copy()

            # Calculate confidence (simple heuristic based on component density)
            confidence = min(
                100, self.base_detection_confidence + (len(non_outliers) / 10)
            )

            results.append(
                {
                    "bbox": (x0, y0, w, h),
                    "image": check_image,
                    "confidence": confidence,
                    "area": w * h,
                    "components_used": len(non_outliers),
                    "outliers_removed": len(outliers),
                }
            )

            logger.info(
                f"Detected check: bbox=({x0}, {y0}, {w}, {h}), "
                f"size=({w/self.dpi:.1f}x{h/self.dpi:.1f}in), "
                f"components={len(non_outliers)}, "
                f"outliers_removed={len(outliers)}"
            )

        if len(results) > 1 and self.vertical_overlap_margin_px > 0:
            self._trim_vertical_overlap(results, image)

        return results

    def count_checks(self, image: np.ndarray) -> int:
        """
        Quick count of checks on a page without extracting images.

        Args:
            image: Page image as numpy array

        Returns:
            Number of checks detected
        """
        detected = self.detect_checks(image)
        return len(detected)

    def has_multiple_checks(
        self, image: np.ndarray, min_confidence: float = 50.0
    ) -> bool:
        """
        Determine if page contains multiple checks.

        Args:
            image: Page image as numpy array
            min_confidence: Minimum confidence to count as a check

        Returns:
            True if 2+ checks detected with sufficient confidence
        """
        detected = self.detect_checks(image)

        # Filter by confidence
        confident_checks = [c for c in detected if c["confidence"] >= min_confidence]

        return len(confident_checks) >= 2

    def _trim_vertical_overlap(
        self, detections: List[Dict[str, any]], image: np.ndarray
    ) -> None:
        """
        Reduce bleed between stacked checks by trimming overlapping vertical ranges.
        """
        sorted_indices = sorted(range(len(detections)), key=lambda i: detections[i]["bbox"][1])

        for idx in range(1, len(sorted_indices)):
            prev_idx = sorted_indices[idx - 1]
            cur_idx = sorted_indices[idx]

            prev_bbox = detections[prev_idx]["bbox"]
            cur_bbox = detections[cur_idx]["bbox"]

            prev_top = prev_bbox[1]
            prev_height = prev_bbox[3]
            prev_bottom = prev_top + prev_height

            cur_x, cur_y, cur_w, cur_h = cur_bbox
            cur_bottom = cur_y + cur_h

            min_allowed_top = prev_bottom + self.vertical_overlap_margin_px
            if cur_y >= min_allowed_top:
                continue

            new_top = min_allowed_top
            if new_top >= cur_bottom:
                new_top = cur_bottom - 1
            new_height = cur_bottom - new_top

            if new_height <= 0:
                continue

            detections[cur_idx]["bbox"] = (cur_x, new_top, cur_w, new_height)
            detections[cur_idx]["image"] = image[new_top:cur_bottom, cur_x : cur_x + cur_w].copy()
            detections[cur_idx]["area"] = cur_w * new_height

            logger.info(
                f"Trimming overlap between checks: adjusted bbox of check #{cur_idx + 1} "
                f"from y={cur_y} to y={new_top}"
            )
