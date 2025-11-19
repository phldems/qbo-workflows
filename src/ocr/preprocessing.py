"""
Image preprocessing for check OCR.
Improves OCR accuracy through various image enhancement techniques.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from loguru import logger
import pytesseract
from pytesseract import Output
import math


class ImagePreprocessor:
    """Preprocesses check images for optimal OCR results."""

    def __init__(self):
        """Initialize preprocessor."""
        self.dpi_target = 300  # Target DPI for OCR

    def preprocess(
        self,
        image_path: str,
        save_preprocessed: bool = False,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to check image.

        Args:
            image_path: Path to input image
            save_preprocessed: Whether to save preprocessed image
            output_path: Where to save preprocessed image

        Returns:
            Preprocessed image as numpy array
        """
        logger.info(f"Preprocessing image: {image_path}")

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Apply preprocessing steps
        gray = self._convert_to_grayscale(img)
        deskewed = self._deskew(gray)
        denoised = self._denoise(deskewed)
        enhanced = self._enhance_contrast(denoised)
        binary = self._binarize(enhanced)
        cleaned = self._remove_noise(binary)

        # Save if requested
        if save_preprocessed:
            output = output_path or self._get_preprocessed_path(image_path)
            cv2.imwrite(str(output), cleaned)
            logger.info(f"Saved preprocessed image: {output}")

        return cleaned

    def _convert_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            img: Input color image

        Returns:
            Grayscale image
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            logger.debug("Converted to grayscale")
            return gray
        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Correct skew/rotation in image using Hough line detection.
        More robust than minAreaRect for documents with text lines.

        Args:
            img: Input grayscale image

        Returns:
            Deskewed image
        """
        # Check aspect ratio - checks are typically wider than tall
        h, w = img.shape[:2]
        aspect_ratio = w / h

        # If aspect ratio suggests wrong orientation (portrait instead of landscape)
        # This shouldn't happen for properly scanned checks
        if aspect_ratio < 1.0:
            logger.warning(
                f"Unusual aspect ratio {aspect_ratio:.2f} - check may be rotated 90 degrees"
            )

        # Use edge detection to find text lines
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            logger.debug("No lines detected for deskewing, using original image")
            return img

        # Calculate angles of detected lines (in degrees)
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (
                (theta - np.pi / 2) * 180 / np.pi
            )  # Convert to degrees from horizontal

            # Only consider near-horizontal lines (within ±20 degrees)
            if abs(angle) < 20:
                angles.append(angle)

        if not angles:
            logger.debug("No horizontal lines detected, skipping deskew")
            return img

        # Use median angle to avoid outliers
        median_angle = np.median(angles)

        # Only deskew if angle is significant (> 0.5 degrees)
        if abs(median_angle) < 0.5:
            logger.debug("Image already straight, skipping deskew")
            return img

        # Limit rotation to reasonable range (±10 degrees)
        if abs(median_angle) > 10:
            logger.warning(
                f"Detected angle {median_angle:.2f}° seems too large, limiting to ±10°"
            )
            median_angle = np.clip(median_angle, -10, 10)

        # Apply rotation
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
        return deskewed

    def _denoise(self, img: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Remove noise from image while preserving edges.

        Args:
            img: Input grayscale image
            method: Denoising method ('bilateral', 'nlmeans', 'gaussian')

        Returns:
            Denoised image
        """
        if method == "bilateral":
            # Bilateral filter preserves edges well
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
        elif method == "nlmeans":
            # Non-local means denoising (slower but better quality)
            denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        elif method == "gaussian":
            # Simple Gaussian blur
            denoised = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            raise ValueError(f"Unknown denoising method: {method}")

        logger.debug(f"Applied {method} denoising")
        return denoised

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.

        Args:
            img: Input grayscale image

        Returns:
            Contrast-enhanced image
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

        logger.debug("Enhanced contrast with CLAHE")
        return enhanced

    def _binarize(self, img: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Convert grayscale to binary (black and white).

        Args:
            img: Input grayscale image
            method: Binarization method ('adaptive', 'otsu', 'simple')

        Returns:
            Binary image
        """
        if method == "adaptive":
            # Adaptive thresholding handles uneven lighting
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            # Otsu's method automatically determines threshold
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "simple":
            # Simple fixed threshold
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unknown binarization method: {method}")

        logger.debug(f"Binarized with {method} thresholding")
        return binary

    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Remove small noise artifacts using morphological operations.

        Args:
            img: Input binary image

        Returns:
            Cleaned binary image
        """
        # Morphological closing removes small black holes
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # Optional: Opening removes small white noise
        # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        logger.debug("Removed noise with morphological operations")
        return cleaned

    def _remove_scanner_border(
        self, img: np.ndarray, border_size: int = 20
    ) -> np.ndarray:
        """
        Remove dark scanner borders/shadows from edges of image.
        Common artifact from flatbed scanners.

        Args:
            img: Input grayscale or color image
            border_size: Size of border to analyze/remove (pixels)

        Returns:
            Image with scanner borders removed
        """
        if border_size <= 0:
            return img

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img

        # Use grayscale version to evaluate border darkness, but update original image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        result = img.copy()

        # Helper to whiten border regions when the average intensity indicates a dark strip
        def _maybe_whiten(region_slice, description: str) -> None:
            region = gray[region_slice]
            if region.size == 0:
                return
            if np.mean(region) < 200:
                result[region_slice] = 255
                logger.debug(f"Removed dark border from {description} edge")

        # Top edge
        _maybe_whiten((slice(0, min(border_size, h)), slice(None)), "top")
        # Bottom edge
        _maybe_whiten((slice(max(h - border_size, 0), h), slice(None)), "bottom")
        # Left edge
        _maybe_whiten((slice(None), slice(0, min(border_size, w))), "left")
        # Right edge
        _maybe_whiten((slice(None), slice(max(w - border_size, 0), w)), "right")

        return result

    def extract_region(
        self, img: np.ndarray, x: int, y: int, width: int, height: int
    ) -> np.ndarray:
        """
        Extract a specific region from image (for targeted OCR).

        Args:
            img: Input image
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Region width
            height: Region height

        Returns:
            Cropped image region
        """
        return img[y : y + height, x : x + width]

    def detect_check_boundary(
        self, img: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the check boundary in a full page image using density-based analysis.
        Finds where the actual check content is, not just scattered pixels.

        Args:
            img: Input grayscale image

        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        height, width = img.shape[:2]

        # Method: Density-based content detection
        # Find pixels darker than threshold
        threshold = 240
        mask = img < threshold

        # Calculate density for each row and column
        # Density = percentage of non-white pixels in that row/column
        row_density = np.sum(mask, axis=1) / width
        col_density = np.sum(mask, axis=0) / height

        # Find rows/columns with significant content (>2% density for checks)
        # This filters out scattered pixels and finds actual content blocks
        min_density = 0.02
        significant_rows = np.where(row_density > min_density)[0]
        significant_cols = np.where(col_density > min_density)[0]

        if len(significant_rows) == 0 or len(significant_cols) == 0:
            logger.warning("No significant content detected for boundary detection")
            return None

        # Get bounding box of significant content
        y_min, y_max = significant_rows[0], significant_rows[-1]
        x_min, x_max = significant_cols[0], significant_cols[-1]

        content_width = x_max - x_min + 1
        content_height = y_max - y_min + 1

        # Check if this looks reasonable
        content_area = content_width * content_height
        image_area = width * height
        area_ratio = content_area / image_area

        # Check fills 10-99% of image area (checks with margins)
        if 0.10 <= area_ratio < 0.99:
            # Add padding for safety
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width - 1, x_max + padding)
            y_max = min(height - 1, y_max + padding)

            content_width = x_max - x_min + 1
            content_height = y_max - y_min + 1

            logger.info(
                f"Detected check boundary: ({x_min}, {y_min}) - "
                f"{content_width}x{content_height} pixels "
                f"({area_ratio*100:.1f}% of image)"
            )
            return (x_min, y_min, content_width, content_height)
        elif area_ratio >= 0.99:
            logger.info("Check fills entire page (>99% of image)")
            return None  # Use full image
        else:
            logger.warning(f"Content area too small ({area_ratio*100:.1f}%)")
            return None

    def crop_to_check(
        self, img: np.ndarray, strip_border: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Crop image to check boundary.

        Args:
            img: Input image (color or grayscale)
            strip_border: Whether to remove common scanner borders before detecting the check

        Returns:
            Tuple of (cropped image, boundary coordinates) or (border-cleaned image, None)
        """
        # Strip common scanner borders before attempting to find the check boundary
        if strip_border:
            img_for_boundary = self._remove_scanner_border(img)
        else:
            img_for_boundary = img

        # Convert to grayscale if needed for detection
        if len(img_for_boundary.shape) == 3:
            gray = cv2.cvtColor(img_for_boundary, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_for_boundary

        # Detect boundary
        boundary = self.detect_check_boundary(gray)

        if boundary:
            x, y, w, h = boundary
            cropped = img_for_boundary[y : y + h, x : x + w]
            logger.info(
                f"Cropped image from {img_for_boundary.shape[:2]} to {cropped.shape[:2]}"
            )
            return cropped, boundary
        else:
            logger.warning("Using full image (no check boundary detected)")
            return img_for_boundary, None

    def detect_micr_line(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect MICR line region in check image.
        Align the band tightly to the digits with a thin buffer underneath.

        Args:
            img: Input grayscale image

        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        height, width = img.shape[:2]

        # Examine the bottom portion where the MICR digits live.
        micr_band_start = int(height * 0.70)
        micr_region = img[micr_band_start:, :]
        if micr_region.size == 0:
            return None

        # Identify dark rows that correspond to the MICR characters.
        micr_mask = micr_region < 170
        row_density = np.sum(micr_mask, axis=1) / float(width)

        # Work upward from the bottom to find the digit band.
        threshold_main = 0.018
        threshold_stop = 0.010
        max_band_height = int(height * 0.14)
        bottom_idx = None

        for idx in range(len(row_density) - 1, -1, -1):
            if row_density[idx] > threshold_main:
                bottom_idx = idx
                break

        if bottom_idx is None:
            logger.warning(
                "Could not precisely detect MICR line; using estimated bottom band"
            )
            micr_height = max(int(height * 0.08), 40)
            return (0, height - micr_height, width, micr_height)

        top_idx = bottom_idx
        band_height = 0
        while (
            top_idx > 0
            and row_density[top_idx - 1] > threshold_stop
            and band_height < max_band_height
        ):
            top_idx -= 1
            band_height = bottom_idx - top_idx

        padding_above = int(height * 0.005)
        padding_below = int(height * 0.01)

        y_min = max(0, micr_band_start + top_idx - padding_above)
        y_max = min(height - 1, micr_band_start + bottom_idx + padding_below)

        micr_height = y_max - y_min + 1
        logger.info(
            f"Detected MICR region from row {y_min} to {y_max} "
            f"(height {micr_height})"
        )
        return (0, y_min, width, micr_height)

    def detect_amount_region(
        self,
        img: np.ndarray,
        payee_region: Optional[Tuple[int, int, int, int]] = None,
        micr_region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the numeric amount box adjacent to the payee line.
        Searches the right side for a rounded rectangle that sits roughly mid-way
        between the top of the check and the MICR band.

        Args:
            img: Input grayscale image
            payee_region: Optional pre-detected payee region for adjacency checks
            micr_region: Optional MICR region to define vertical band positioning

        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        height, width = img.shape[:2]

        def fallback() -> Tuple[int, int, int, int]:
            if payee_region:
                px, py, pw, ph = payee_region
                estimated_x = min(width - 1, px + pw + int(width * 0.02))
                estimated_w = min(int(width * 0.28), width - estimated_x)
                estimated_y = max(0, py - int(ph * 0.2))
                estimated_h = min(int(ph * 1.2), height - estimated_y)
                return (estimated_x, estimated_y, estimated_w, estimated_h)
            estimated_x = int(width * 0.65)
            estimated_y = int(height * 0.18)
            estimated_w = int(width * 0.28)
            estimated_h = int(height * 0.12)
            return (estimated_x, estimated_y, estimated_w, estimated_h)

        if payee_region is None:
            payee_region = self.detect_payee_region(img)

        if micr_region is None:
            micr_region = self.detect_micr_line(img)

        micr_top = micr_region[1] if micr_region else int(height * 0.82)
        target_min = int(0.40 * micr_top)
        target_max = int(0.60 * micr_top)

        payee_mid = None
        if payee_region:
            px, py, pw, ph = payee_region
            payee_mid = py + ph / 2.0
            align_window = max(int(ph * 0.6), 40)
        else:
            align_window = int(height * 0.08)

        if payee_region:
            px, py, pw, ph = payee_region
            search_x_start = max(int(width * 0.55), min(width - 1, px + int(pw * 0.60)))
        else:
            search_x_start = int(width * 0.65)
        roi = img[:, search_x_start:]
        if roi.size == 0:
            logger.warning("Amount ROI empty; using fallback region")
            return fallback()

        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blur, 45, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: List[Tuple[float, int, int, int, int]] = []
        ideal_y = (target_min + target_max) / 2.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) < 4 or len(approx) > 12:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 120 or h < 60:
                continue

            aspect = w / h if h > 0 else 0
            if not (0.6 <= aspect <= 7.0):
                continue

            area_ratio = area / float(w * h)
            if area_ratio > 0.6:
                continue

            top_global = y
            bottom_global = y + h

            candidate_x = x + search_x_start
            center_y = top_global + h / 2.0

            if payee_region:
                px, py, pw, ph = payee_region
                if candidate_x < px + int(pw * 0.55):
                    continue
                if payee_mid is not None:
                    if abs(center_y - payee_mid) > align_window:
                        continue
                    if h > align_window * 2.2:
                        continue
            else:
                if center_y < target_min or center_y > target_max:
                    continue

            edge_slice = edges[
                max(0, y - 2) : min(edges.shape[0], y + h + 2), x : x + w
            ]
            edge_strength = float(np.mean(edge_slice)) if edge_slice.size else 0.0

            score = (w + h) - abs(top_global - ideal_y) + edge_strength
            candidates.append((score, candidate_x, top_global, w, h))

        if candidates:
            candidates.sort(reverse=True, key=lambda c: c[0])
            _, cand_x, cand_y, cand_w, cand_h = candidates[0]

            padding = int(max(8, min(cand_w, cand_h) * 0.18))
            actual_x = max(0, cand_x - padding)
            actual_y = max(0, cand_y - padding)
            w = min(width - actual_x, cand_w + 2 * padding)
            h = min(height - actual_y, cand_h + 2 * padding)

            logger.info(f"Detected amount region at ({actual_x}, {actual_y}, {w}, {h})")
            return (actual_x, actual_y, w, h)

        logger.warning("Could not detect numeric amount box, using estimated region")
        return fallback()

    def detect_written_amount_region(
        self, img: np.ndarray, payee_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the written-out amount line below the payee area.
        """
        height, width = img.shape[:2]

        if payee_region is None:
            payee_region = self.detect_payee_region(img)

        if payee_region:
            px, py, pw, ph = payee_region
            search_y_start = min(height - 1, py + ph + max(10, int(0.02 * height)))
            search_y_end = min(height, search_y_start + int(ph * 1.8))
            search_x_start = max(0, px - int(0.05 * width))
            search_x_end = min(width, px + pw + int(0.25 * width))

            if search_y_start < search_y_end:
                region = img[search_y_start:search_y_end, search_x_start:search_x_end]
                if region.size > 0:
                    blurred = cv2.GaussianBlur(region, (5, 5), 0)
                    _, thresh = cv2.threshold(
                        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )

                    contours, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        all_points = np.vstack(contours)
                        x, y, w, h = cv2.boundingRect(all_points)

                        actual_x = search_x_start + x
                        actual_y = search_y_start + y

                        padding_x = int(width * 0.01)
                        padding_y = int(ph * 0.2)

                        actual_x = max(0, actual_x - padding_x)
                        actual_y = max(0, actual_y - padding_y)
                        w = min(width - actual_x, w + 2 * padding_x)
                        h = min(height - actual_y, h + 2 * padding_y)

                        logger.info(
                            f"Detected written amount region at "
                            f"({actual_x}, {actual_y}, {w}, {h})"
                        )
                        return (actual_x, actual_y, w, h)

            # Fallback relative to payee.
            estimated_x = max(0, px - int(0.02 * width))
            estimated_y = min(height - 1, py + ph + int(ph * 0.3))
            estimated_w = min(width - estimated_x, pw + int(0.25 * width))
            estimated_h = min(int(ph * 1.1), height - estimated_y)
            logger.warning("Could not detect written amount, using estimated region")
            return (estimated_x, estimated_y, estimated_w, estimated_h)

        # Generic fallback if payee is unavailable.
        estimated_x = int(width * 0.12)
        estimated_y = int(height * 0.40)
        estimated_w = int(width * 0.70)
        estimated_h = int(height * 0.10)
        logger.warning(
            "Payee region unavailable; using default written amount estimate"
        )
        return (estimated_x, estimated_y, estimated_w, estimated_h)

    def detect_payor_region(
        self, img: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the payor block (typically top-left/center of the check).
        """
        height, width = img.shape[:2]

        top_band = max(1, int(height * 0.28))
        region = img[:top_band, :]
        if region.size == 0:
            return None

        left_limit = int(width * 0.75)
        region_left = region[:, :left_limit]
        mask = (region_left < 220).astype(np.uint8)

        if cv2.countNonZero(mask) > 0:
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)

                padding_x = int(width * 0.02)
                padding_y = int(height * 0.02)

                actual_x = max(0, x - padding_x)
                actual_y = max(0, y - padding_y)
                w = min(width - actual_x, w + 2 * padding_x)
                h = min(top_band - actual_y, h + 2 * padding_y)

                logger.info(
                    f"Detected payor region at ({actual_x}, {actual_y}, {w}, {h})"
                )
                return (actual_x, actual_y, w, h)

        logger.warning("Could not detect payor block, using estimated region")
        estimated_x = int(width * 0.05)
        estimated_y = int(height * 0.05)
        estimated_w = int(width * 0.40)
        estimated_h = int(height * 0.15)
        return (estimated_x, estimated_y, estimated_w, estimated_h)

    def detect_check_number_region(
        self, img: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the printed check/reference number, usually in the upper-right corner.
        """
        height, width = img.shape[:2]

        search_y_end = max(1, int(height * 0.20))
        search_x_start = int(width * 0.55)
        region = img[:search_y_end, search_x_start:]
        if region.size == 0:
            return None

        blurred = cv2.GaussianBlur(region, (3, 3), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8), iterations=1
        )
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: List[Tuple[float, int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            actual_x = search_x_start + x

            if actual_x < int(width * 0.60):
                continue
            if w < 40 or h < 25:
                continue
            if not (1.5 <= aspect_ratio <= 10.5):
                continue
            if area < 1200:
                continue

            candidates.append((area, actual_x, y, w, h))

        if candidates:
            # Prefer candidates furthest to the right; break ties by area.
            candidates.sort(key=lambda c: (c[1] + c[3], c[0]), reverse=True)
            _, actual_x, actual_y, w, h = candidates[0]
            padding = 12

            actual_x = max(0, actual_x - padding)
            actual_y = max(0, actual_y - padding)
            w = min(width - actual_x, w + 2 * padding)
            h = min(height - actual_y, h + 2 * padding)

            logger.info(
                f"Detected check number region at ({actual_x}, {actual_y}, {w}, {h})"
            )
            return (actual_x, actual_y, w, h)

        logger.warning("Could not detect check number, using estimated region")
        estimated_x = int(width * 0.70)
        estimated_y = int(height * 0.05)
        estimated_w = int(width * 0.25)
        estimated_h = int(height * 0.10)
        return (estimated_x, estimated_y, estimated_w, estimated_h)

    def detect_payee_region(
        self, img: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the handwritten payee (Pay to the Order of) region in a check image.
        Uses the printed label as an anchor, constrained Hough search, and adaptive padding.

        Args:
            img: Input grayscale image

        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        height, width = img.shape[:2]

        def legacy_fallback() -> Tuple[int, int, int, int]:
            estimated_x = int(width * 0.15)
            estimated_y = int(height * 0.35)
            estimated_w = int(width * 0.60)
            estimated_h = int(height * 0.08)
            return (estimated_x, estimated_y, estimated_w, estimated_h)

        def locate_label() -> (
            Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]
        ):
            """
            Run a light OCR pass to find the 'Pay to the Order of' label.

            Returns:
                (label_bbox, baseline_y) where label_bbox is (x, y, w, h) in image coords.
            """
            x_start = int(width * 0.05)
            x_end = int(width * 0.55)
            y_start = int(height * 0.18)
            y_end = int(height * 0.45)
            label_region = img[y_start:y_end, x_start:x_end]
            if label_region.size == 0:
                return None, None

            try:
                data = pytesseract.image_to_data(
                    label_region, output_type=Output.DICT, config="--oem 1 --psm 6"
                )
            except pytesseract.TesseractError as e:
                logger.debug(f"Tesseract failed on label detection: {e}")
                return None, None

            lines = {}
            n = len(data.get("text", []))
            for i in range(n):
                text = data["text"][i]
                if not text or not text.strip():
                    continue
                if data["level"][i] != 5:
                    continue  # word level only
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)

            for indices in lines.values():
                words = [
                    data["text"][idx].strip()
                    for idx in indices
                    if data["text"][idx].strip()
                ]
                if not words:
                    continue
                joined = " ".join(words).lower()
                joined_compact = joined.replace(" ", "")
                if "pay" in joined and "order" in joined_compact:
                    xs = [data["left"][idx] for idx in indices]
                    ys = [data["top"][idx] for idx in indices]
                    ws = [data["width"][idx] for idx in indices]
                    hs = [data["height"][idx] for idx in indices]
                    left = min(xs)
                    top = min(ys)
                    right = max(xs[i] + ws[i] for i in range(len(xs)))
                    bottom = max(ys[i] + hs[i] for i in range(len(ys)))
                    bbox = (left + x_start, top + y_start, right - left, bottom - top)
                    baseline_y = bbox[1] + min(bbox[3] - 1, int(round(bbox[3] * 0.75)))
                    return bbox, float(baseline_y)

            return None, None

        label_bbox, baseline_y = locate_label()
        if baseline_y is None:
            baseline_y = float(int(height * 0.36))

        band_half = max(
            int(height * 0.12),
            int(label_bbox[3] * 2) if label_bbox else int(height * 0.08),
        )
        band_top = max(0, int(baseline_y) - band_half)
        band_bottom = min(height, int(baseline_y) + band_half)

        x_limit = int(width * 0.72)
        payee_roi = img[band_top:band_bottom, :x_limit]
        if payee_roi.size == 0:
            logger.warning("Payee ROI empty; falling back to legacy payee estimate")
            return legacy_fallback()

        roi_blur = cv2.GaussianBlur(payee_roi, (5, 5), 0)
        edges = cv2.Canny(roi_blur, 40, 120)

        min_line_len = max(60, int(width * 0.35))
        max_line_gap = max(10, int(width * 0.04))
        hough_threshold = max(40, int(width * 0.01))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )

        roi_thresh = cv2.adaptiveThreshold(
            roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )

        full_band = img[band_top:band_bottom, :]
        full_band_blur = cv2.GaussianBlur(full_band, (5, 5), 0)
        full_band_binary = cv2.adaptiveThreshold(
            full_band_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5,
        )

        def estimate_text_height(binary: np.ndarray, target_y: float) -> float:
            heights = []
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h < 5 or h > binary.shape[0] * 0.8:
                    continue
                actual_center = band_top + y + h / 2.0
                if abs(actual_center - target_y) <= max(12, binary.shape[0] * 0.25):
                    heights.append(h)
            if heights:
                return float(np.median(heights))
            return max(12.0, height * 0.02)

        def find_dollars_x(x_start: int) -> Optional[int]:
            right_roi = img[band_top:band_bottom, x_start:width]
            if right_roi.size == 0:
                return None
            try:
                data = pytesseract.image_to_data(
                    right_roi, output_type=Output.DICT, config="--oem 1 --psm 6"
                )
            except pytesseract.TesseractError:
                return None
            for i, text in enumerate(data.get("text", [])):
                if not text or not text.strip():
                    continue
                token = text.strip().lower()
                if "dollar" in token:
                    return x_start + data["left"][i]
            return None

        def extend_right(x_start: int, baseline: float) -> int:
            dollars_x = find_dollars_x(max(0, x_start))
            if dollars_x is not None:
                return min(width - 1, dollars_x)

            column_density = np.mean(full_band_binary > 0, axis=0)
            blank_threshold_cols = max(20, int(width * 0.10))
            blank_counter = 0
            for col in range(int(x_start), width):
                if column_density[col] < 0.02:
                    blank_counter += 1
                    if blank_counter >= blank_threshold_cols:
                        return max(int(x_start) + 5, col - blank_threshold_cols // 2)
                else:
                    blank_counter = 0
            return min(
                width - 1, int(x_start + max(blank_threshold_cols, int(width * 0.25)))
            )

        best_line = None
        if lines is not None:
            lambda_y = 1.2
            mu = 80.0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                actual_x1 = int(x1)
                actual_x2 = int(x2)
                actual_y1 = band_top + int(y1)
                actual_y2 = band_top + int(y2)

                angle = math.degrees(
                    math.atan2(actual_y2 - actual_y1, actual_x2 - actual_x1)
                )
                if abs(angle) > 7:
                    continue

                line_length = math.hypot(actual_x2 - actual_x1, actual_y2 - actual_y1)
                if line_length < min_line_len:
                    continue

                y_center = (actual_y1 + actual_y2) / 2.0

                strip_y = int(round(y_center)) - band_top
                strip_top = max(0, strip_y - 2)
                strip_bottom = min(payee_roi.shape[0], strip_y + 3)
                strip_left = max(0, min(x1, x2) - 2)
                strip_right = min(payee_roi.shape[1], max(x1, x2) + 2)
                strip = payee_roi[strip_top:strip_bottom, strip_left:strip_right]
                edge_var = float(np.var(strip) / (255.0**2)) if strip.size else 0.0

                score = (
                    line_length - lambda_y * abs(y_center - baseline_y) - mu * edge_var
                )

                if best_line is None or score > best_line["score"]:
                    best_line = {
                        "score": score,
                        "x_min": min(actual_x1, actual_x2),
                        "x_max": max(actual_x1, actual_x2),
                        "y_center": y_center,
                        "line_length": line_length,
                    }

        if best_line:
            median_height = estimate_text_height(roi_thresh, best_line["y_center"])
            pad_up = int(max(8, 1.4 * median_height))
            pad_down = int(max(6, 1.1 * median_height))

            y_top = max(0, int(round(best_line["y_center"])) - pad_up)
            y_bottom = min(height, int(round(best_line["y_center"])) + pad_down)

            x_min = max(0, int(best_line["x_min"]) - int(width * 0.02))
            x_max = extend_right(int(best_line["x_max"]), best_line["y_center"])
            if x_max <= x_min:
                logger.warning(
                    "Payee extension failed (x_max <= x_min); using legacy fallback"
                )
                return legacy_fallback()

            h = max(10, y_bottom - y_top)
            w = min(width - x_min, x_max - x_min)
            logger.info(f"Detected payee region at ({x_min}, {y_top}, {w}, {h})")
            return (x_min, y_top, w, h)

        if label_bbox:
            label_height = max(8.0, float(label_bbox[3]))
            pad_up = int(max(8, 1.4 * label_height))
            pad_down = int(max(6, 1.1 * label_height))

            baseline = baseline_y
            y_top = max(0, int(round(baseline)) - pad_up)
            y_bottom = min(height, int(round(baseline)) + pad_down)

            x_min = max(0, label_bbox[0])
            x_start = label_bbox[0] + label_bbox[2] + int(width * 0.01)
            x_max = extend_right(int(x_start), baseline)
            if x_max <= x_min:
                return legacy_fallback()

            h = max(10, y_bottom - y_top)
            w = min(width - x_min, x_max - x_min)
            logger.info(
                "Payee line not found; using label-anchored estimate "
                f"at ({x_min}, {y_top}, {w}, {h})"
            )
            return (x_min, y_top, w, h)

        logger.warning("Could not detect payee line or label; using legacy estimate")
        return legacy_fallback()

    def extract_region(
        self, img: np.ndarray, region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract a specific region from an image.

        Args:
            img: Input image
            region: Tuple of (x, y, width, height)

        Returns:
            Extracted region as numpy array
        """
        x, y, w, h = region
        height, width = img.shape[:2]

        # Ensure coordinates are within bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        return img[y : y + h, x : x + w]

    def resize_for_ocr(self, img: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """
        Resize image to target DPI for optimal OCR.

        Args:
            img: Input image
            target_dpi: Target DPI (300 recommended)

        Returns:
            Resized image
        """
        # If image is too small, upscale
        height, width = img.shape[:2]

        # Minimum size for good OCR (at 300 DPI)
        min_width = 2400  # ~8 inches at 300 DPI
        min_height = 1000  # ~3.3 inches at 300 DPI

        if width < min_width or height < min_height:
            scale = max(min_width / width, min_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            resized = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

            logger.debug(
                f"Upscaled image from {width}x{height} to {new_width}x{new_height}"
            )
            return resized

        return img

    def _get_preprocessed_path(self, original_path: str) -> Path:
        """
        Generate path for preprocessed image.

        Args:
            original_path: Original image path

        Returns:
            Path for preprocessed image
        """
        path = Path(original_path)
        return path.parent / f"{path.stem}_preprocessed{path.suffix}"

    def preprocess_for_handwriting(self, image_path: str) -> np.ndarray:
        """
        Specialized preprocessing for handwritten text regions.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image optimized for handwriting
        """
        logger.info(f"Preprocessing for handwriting: {image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to grayscale
        gray = self._convert_to_grayscale(img)

        # Deskew
        deskewed = self._deskew(gray)

        # Gentle denoising (preserve handwriting strokes)
        denoised = cv2.fastNlMeansDenoising(deskewed, None, 5, 7, 21)

        # Enhance contrast
        enhanced = self._enhance_contrast(denoised)

        # For handwriting, sometimes grayscale is better than binary
        # Only binarize if image has very clean handwriting
        # binary = self._binarize(enhanced, method='adaptive')

        # Resize for optimal OCR
        resized = self.resize_for_ocr(enhanced)

        return resized

    def preprocess_for_micr(self, image_path: str) -> np.ndarray:
        """
        Specialized preprocessing for MICR line (printed numbers at bottom).

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed MICR region
        """
        logger.info(f"Preprocessing MICR line: {image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Full preprocessing pipeline
        gray = self._convert_to_grayscale(img)
        deskewed = self._deskew(gray)

        # Detect MICR region
        micr_coords = self.detect_micr_line(deskewed)

        if micr_coords:
            x, y, w, h = micr_coords
            micr_region = self.extract_region(deskewed, x, y, w, h)
        else:
            # Fallback: use bottom 10% of image
            height = deskewed.shape[0]
            micr_region = deskewed[int(height * 0.9) :, :]

        # Aggressive preprocessing for MICR (printed text)
        denoised = self._denoise(micr_region, method="bilateral")
        enhanced = self._enhance_contrast(denoised)
        binary = self._binarize(enhanced, method="otsu")
        cleaned = self._remove_noise(binary)

        return cleaned
