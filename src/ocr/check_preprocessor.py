"""
Check Image Preprocessing Pipeline

Comprehensive preprocessing to normalize, correct geometry, and enhance
check images before detection and OCR.

Pipeline:
1. Normalization: Decode, standardize format
2. Geometry Correction: Detect edges, perspective correction, deskew
3. Illumination & Background: Flatten lighting, binarization
4. Noise Cleanup: Remove artifacts, morphological ops
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from loguru import logger


class CheckPreprocessor:
    """Preprocesses check images for optimal detection and OCR."""

    def __init__(self, dpi: int = 300):
        """
        Initialize preprocessor.

        Args:
            dpi: Target DPI for normalized output
        """
        self.dpi = dpi
        self.target_width_px = int(8.5 * dpi)  # Standard letter width
        self.target_height_px = int(11 * dpi)  # Standard letter height

    def process(self, image: np.ndarray, debug: bool = False) -> Dict:
        """
        Run full preprocessing pipeline.

        Args:
            image: Input image (RGB or grayscale)
            debug: If True, return intermediate stages

        Returns:
            Dict with:
                - 'processed': Final processed image
                - 'original': Original image
                - 'stages': Dict of intermediate stages (if debug=True)
        """
        logger.info(f"Starting preprocessing pipeline (input shape: {image.shape})")

        stages = {} if debug else None
        original = image.copy()

        # Step 1: Decode & Standardize
        rgb, gray = self._normalize_format(image)
        if debug:
            stages["01_rgb"] = rgb
            stages["02_gray"] = gray

        # Step 2: Denoise (gentle)
        denoised = self._denoise(gray)
        if debug:
            stages["03_denoised"] = denoised

        # Step 3: Geometry Correction
        corrected, transform_info = self._correct_geometry(rgb, denoised)
        if debug:
            stages["04_geometry_corrected"] = corrected
            stages["04_transform_info"] = transform_info

        # Convert corrected to grayscale
        if len(corrected.shape) == 3:
            corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        else:
            corrected_gray = corrected

        # Step 4: Illumination & Background Normalization
        normalized = self._normalize_illumination(corrected_gray)
        if debug:
            stages["05_illumination_normalized"] = normalized

        # Step 5: Adaptive Binarization
        binary = self._adaptive_binarization(normalized)
        if debug:
            stages["06_binary"] = binary

        # Step 6: Noise Cleanup
        cleaned = self._cleanup_noise(binary)
        if debug:
            stages["07_cleaned"] = cleaned

        logger.info(f"Preprocessing complete (output shape: {corrected.shape})")

        return {
            "processed": corrected,  # Return RGB/color version for OCR
            "processed_gray": corrected_gray,
            "processed_binary": cleaned,
            "original": original,
            "stages": stages,
            "transform_info": transform_info,
        }

    def _normalize_format(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1: Decode and standardize format.

        Returns:
            (rgb_image, gray_image)
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            # Grayscale -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            gray = image.copy()
        elif image.shape[2] == 4:
            # RGBA -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            # Already RGB
            rgb = image.copy()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        rgb, gray = self._ensure_portrait(rgb, gray)
        return rgb, gray

    def _ensure_portrait(
        self, rgb: np.ndarray, gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate the page 90 degrees if it's landscape so detection stays consistent.
        """
        h, w = gray.shape
        if w > h:
            logger.info("Rotating landscape page to portrait orientation")
            rgb = np.rot90(rgb, k=3)  # clockwise
            gray = np.rot90(gray, k=3)
        return rgb, gray

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        Step 2: Gentle denoising to remove sensor noise / JPEG artifacts.
        Preserves crisp edges on text and MICR.
        """
        # Bilateral filter: smooths background, keeps edges
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        return denoised

    def _correct_geometry(
        self, rgb: np.ndarray, gray: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Step 3: Geometry correction - detect edges, perspective correction, deskew.

        Returns:
            (corrected_image, transform_info)
        """
        h, w = gray.shape

        # Sub-step 3.1: Detect document edges
        quad = self._detect_document_edges(gray)

        if quad is not None:
            # Sub-step 3.2: Perspective correction
            corrected = self._perspective_correction(rgb, quad)

            # Sub-step 3.3: Deskew
            deskewed = self._deskew(corrected)

            transform_info = {
                "quadrilateral": quad,
                "perspective_corrected": True,
                "deskewed": True,
            }

            return deskewed, transform_info
        else:
            # No quadrilateral found - skip geometry correction
            logger.debug(
                "No document quadrilateral detected, skipping geometry correction"
            )

            # Still attempt deskew
            deskewed = self._deskew(rgb)

            transform_info = {
                "quadrilateral": None,
                "perspective_corrected": False,
                "deskewed": True,
            }

            return deskewed, transform_info

    def _detect_document_edges(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest quadrilateral that looks check-shaped.

        Returns:
            4-point quadrilateral or None
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Look for a quadrilateral in top 5 largest contours
        for contour in contours[:5]:
            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if it's a quadrilateral
            if len(approx) == 4:
                # Check aspect ratio (checks are ~2.5:1)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h if h > 0 else 0

                if 1.5 <= aspect_ratio <= 4.0:
                    # Good candidate!
                    return approx.reshape(4, 2)

        return None

    def _perspective_correction(
        self, image: np.ndarray, quad: np.ndarray
    ) -> np.ndarray:
        """
        Un-tilt the image using perspective transformation.

        Args:
            image: Input image
            quad: 4-point quadrilateral

        Returns:
            Warped image
        """
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(quad)

        # Compute target dimensions
        (tl, tr, br, bl) = rect

        # Width: max of top and bottom
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        # Height: max of left and right
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Destination points (rectangle)
        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        # Compute homography
        M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)

        # Warp
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order quadrilateral points: TL, TR, BR, BL.
        """
        # Initialize
        rect = np.zeros((4, 2), dtype="float32")

        # Sum: top-left has smallest sum, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        # Difference: top-right has smallest diff, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate image to horizontal using dominant text baseline.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return image

        # Find dominant angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert to degrees
            angle = (theta * 180 / np.pi) - 90

            # Only consider nearly horizontal lines (within 30 degrees)
            if -30 <= angle <= 30:
                angles.append(angle)

        if not angles:
            return image

        # Median angle
        median_angle = np.median(angles)

        # Only rotate if significant skew (> 0.5 degrees)
        if abs(median_angle) > 0.5:
            logger.debug(f"Deskewing by {median_angle:.2f} degrees")

            # Rotate
            h, w = gray.shape if len(image.shape) == 2 else image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return rotated

        return image

    def _normalize_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Step 4: Flatten uneven lighting and increase contrast.
        """
        # Estimate background illumination (large blur)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=50, sigmaY=50)

        # Subtract background (flatten illumination)
        # Use signed arithmetic to avoid clipping
        normalized = cv2.subtract(gray, background)
        normalized = cv2.add(normalized, 127)  # Re-center at mid-gray

        # CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)

        return enhanced

    def _adaptive_binarization(self, gray: np.ndarray) -> np.ndarray:
        """
        Step 5: Convert to black/white using adaptive thresholding.
        """
        # Adaptive Gaussian threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,
            C=5,
        )

        return binary

    def _cleanup_noise(self, binary: np.ndarray) -> np.ndarray:
        """
        Step 6: Remove small noise artifacts.
        """
        # Morphological opening (remove small white noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Remove small connected components
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            opened, connectivity=8
        )

        # Filter out small components (likely noise)
        min_size = 20  # pixels
        cleaned = np.zeros_like(opened)

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= min_size:
                cleaned[labels == i] = 255

        return cleaned
