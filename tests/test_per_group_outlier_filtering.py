"""
Test Per-Group Outlier Filtering

Applies outlier filtering to each check group separately after gap detection.
"""

import cv2
import numpy as np
import fitz
from pathlib import Path
from typing import Iterable, List, Optional
import pytest
import sys

from src.utils.sample_files import get_multi_check_sample_pdf
from src.utils.image_utils import ensure_portrait


def detect_checks_with_per_group_outlier_filtering(
    image: np.ndarray,
    iqr_factor: float = 1.5,
    outlier_threshold: float = 2.5,
    expand_percent: float = 4.0,
) -> list:
    """
    Detect checks with per-group outlier filtering.

    Args:
        image: Input image
        iqr_factor: IQR multiplier for gap detection
        outlier_threshold: Z-score threshold for outlier filtering
        expand_percent: Percentage to expand bounding boxes

    Returns:
        List of detected check bounding boxes
    """
    image = ensure_portrait(image)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

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
        if area < 80:
            continue

        components.append((x, y, w, h, cx, cy))

    if not components:
        return []

    print(f"   Total filtered components: {len(components)}")

    # Sort by vertical center
    components_sorted = sorted(components, key=lambda c: c[5])  # cy

    # ===== GAP DETECTION (same as original CV algorithm) =====
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
        thr = med + iqr_factor * iqr

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

    print(f"   Detected {len(groups)} group(s) via gap detection")

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

                    # If next group is short (< 1.5 inches) and close to current (< 2 inches gap)
                    gap_to_next = next_top - current_bottom
                    if (
                        next_height < 450 and gap_to_next < 600
                    ):  # 1.5in and 2in at 300 DPI
                        print(
                            f"   ⚠️  Merging Group {i+2} (MICR line candidate, height={next_height/300:.1f}in) into Group {i+1}"
                        )
                        merged_groups.append(current_group + next_group)
                        i += 2  # Skip next group as it's now merged
                        continue

        merged_groups.append(current_group)
        i += 1

    groups = merged_groups
    print(f"   After MICR merge: {len(groups)} group(s)")

    # ===== APPLY OUTLIER FILTERING PER GROUP =====
    results = []

    for group_idx, group in enumerate(groups):
        print(f"\n   Group {group_idx + 1}:")
        print(f"      Components in group: {len(group)}")

        if len(group) == 0:
            continue

        # Calculate centroid statistics for THIS GROUP only
        cy_values = [c[5] for c in group]  # c[5] is cy
        cy_mean = np.mean(cy_values)
        cy_std = np.std(cy_values)

        print(f"      Centroid Y statistics: mean={cy_mean:.1f}, std={cy_std:.1f}")

        # Filter outliers within THIS GROUP
        non_outliers = []
        outliers = []

        for comp in group:
            x, y, w, h, cx, cy = comp
            z_score = abs(cy - cy_mean) / (cy_std + 1e-10)

            if z_score <= outlier_threshold:
                non_outliers.append(comp)
            else:
                outliers.append((comp, z_score))

        print(f"      Non-outliers: {len(non_outliers)}")
        print(f"      Outliers removed: {len(outliers)}")

        if outliers:
            for comp, z_score in sorted(outliers, key=lambda x: x[1], reverse=True)[:3]:
                x, y, w, h, cx, cy = comp
                print(f"         Y={y}, H={h}, Z-score={z_score:.2f}")

        # If all filtered out, use original group
        if not non_outliers:
            print(f"      WARNING: All components filtered! Using original group.")
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

        print(
            f"      Bounding box (before expansion): {w}×{h}px ({w/300:.1f}×{h/300:.1f}in)"
        )

        # Apply expansion
        if expand_percent > 0:
            expand_w = int(w * expand_percent / 100)
            expand_h = int(h * expand_percent / 100)

            x0 = max(0, x0 - expand_w)
            y0 = max(0, y0 - expand_h)
            x1 = min(W, x1 + expand_w)
            y1 = min(H, y1 + expand_h)
            w = x1 - x0
            h = y1 - y0

            print(
                f"      After {expand_percent}% expansion: {w}×{h}px ({w/300:.1f}×{h/300:.1f}in)"
            )

        # Filter out tiny boxes (smaller than 1 inch wide)
        min_dimension_px = 300  # 1 inch at 300 DPI
        if w < min_dimension_px:
            print(f"      ⚠️  Box too small (< 1 inch wide), skipping this group")
            continue

        results.append(
            {
                "bbox": (x0, y0, w, h),
                "group_id": group_idx,
                "components_used": len(non_outliers),
                "outliers_removed": len(outliers),
            }
        )

    return results


def run_per_group_outlier_filtering(
    pdf_path: str, page_nums: Optional[Iterable[int]] = None
):
    """Execute the verbose per-group outlier filtering visualization."""

    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    output_dir = Path(f"per_group_outlier_results_{pdf_name}")
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PER-GROUP OUTLIER FILTERING TEST")
    print(f"{'='*80}")
    print(f"File: {pdf_path}\n")
    print(f"📁 Saving results to: {output_dir.absolute()}\n")

    pdf_document = fitz.open(str(pdf_path))
    try:
        total_pages = len(pdf_document)
        if page_nums is None:
            page_indices = list(range(total_pages))
        else:
            page_indices = [p for p in page_nums if 0 <= p < total_pages]

        if not page_indices:
            print("No valid pages requested; exiting.")
            return

        for page_num in page_indices:
            print(f"\n{'-'*80}")
            print(f"Processing page {page_num + 1}/{total_pages}")
            print(f"{'-'*80}")

            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img_data = pix.samples
            image = np.frombuffer(img_data, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = ensure_portrait(image)

            print(
                f"📐 Page size: {image.shape[1]}x{image.shape[0]} "
                f"({image.shape[1]/300:.1f}x{image.shape[0]/300:.1f} inches)\n"
            )

            thresholds = [2.0, 2.5, 3.0]

            for threshold in thresholds:
                print(f"{'='*80}")
                print(f"OUTLIER THRESHOLD: {threshold} standard deviations")
                print(f"{'='*80}")

                detections = detect_checks_with_per_group_outlier_filtering(
                    image, iqr_factor=1.5, outlier_threshold=threshold, expand_percent=4.0
                )

                print(f"\n📊 SUMMARY:")
                print(f"   Total checks detected: {len(detections)}")

                if detections:
                    vis_image = image.copy()

                    for idx, det in enumerate(detections):
                        x, y, w, h = det["bbox"]
                        width_inches = w / 300
                        height_inches = h / 300

                        print(f"\n   Check {idx + 1}:")
                        print(
                            f"      Size: {w}×{h}px ({width_inches:.1f}×{height_inches:.1f}in)"
                        )
                        print(f"      Components used: {det['components_used']}")
                        print(f"      Outliers removed: {det['outliers_removed']}")

                        color = (
                            (0, 255, 0)
                            if len(detections) == 1
                            else ((255, 0, 0) if idx == 0 else (0, 0, 255))
                        )
                        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 8)

                        label = (
                            f"Check {idx + 1}: {width_inches:.1f}×{height_inches:.1f}in"
                        )
                        cv2.putText(
                            vis_image,
                            label,
                            (x + 10, y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            color,
                            4,
                        )

                        cropped = image[y : y + h, x : x + w]
                        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                        filename = (
                            f"page_{page_num + 1}_threshold_{threshold}_check_{idx+1}.png"
                        )
                        cv2.imwrite(str(output_dir / filename), cropped_bgr)

                    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                    vis_name = (
                        f"page_{page_num + 1}_threshold_{threshold}_detections.png"
                    )
                    cv2.imwrite(str(output_dir / vis_name), vis_bgr)

                print()
    finally:
        pdf_document.close()

    print(f"{'='*80}")
    print(f"Test complete!")
    print(f"Check outputs in: {output_dir.absolute()}")
    print(f"{'='*80}\n")


def test_per_group_outlier_filtering():
    """Pytest wrapper that uses the configured sample PDF."""

    try:
        pdf_path = str(get_multi_check_sample_pdf())
    except Exception as exc:
        pytest.skip(f"Multi-check sample PDF not configured: {exc}")

    run_per_group_outlier_filtering(pdf_path, page_nums=[0])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        if len(sys.argv) > 2:
            page_arg = sys.argv[2]
            if page_arg.lower() == "all":
                pages = None
            else:
                pages = [int(page_arg)]
        else:
            pages = None
        run_per_group_outlier_filtering(pdf_file, page_nums=pages)
    else:
        try:
            pdf_path = get_multi_check_sample_pdf()
        except Exception as exc:
            print(f"Error resolving multi-check PDF: {exc}")
            sys.exit(1)
        run_per_group_outlier_filtering(str(pdf_path), page_nums=[0])
