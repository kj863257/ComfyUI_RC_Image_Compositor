import torch
import numpy as np
import cv2
from scipy import linalg


class RC_ColorMatch:
    """RC Color Match | RC色调匹配

    Advanced color matching node with multiple algorithms optimized for quality and performance.
    Based on Photoshop's Match Color functionality with enhanced options.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE", {
                    "tooltip": "Target image to be color matched"
                }),
                "reference": ("IMAGE", {
                    "tooltip": "Reference image to match colors from"
                }),
                "method": (["lab_statistics", "lab_histogram", "mkl_transfer", "wavelet_hybrid"], {
                    "default": "lab_statistics",
                    "tooltip": (
                        "Color matching method:\n"
                        "- lab_statistics: PS-style LAB statistical matching (fastest, most accurate)\n"
                        "- lab_histogram: LAB histogram matching (better detail preservation)\n"
                        "- mkl_transfer: Monge-Kantorovitch Linear transfer (natural, smooth)\n"
                        "- wavelet_hybrid: Wavelet + LAB hybrid (best for textures)"
                    )
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend strength between original and matched (0=original, 1=fully matched)"
                }),
                "preserve_luminosity": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve original luminosity (only match colors, not brightness)"
                }),
                "neutralize": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Neutralize gray tones in target image"
                }),
                "variance_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                    "tooltip": "Scale the variance/contrast of the match (1.0=exact match)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_colors"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Advanced color matching with Photoshop-quality algorithms"

    def match_colors(self, target, reference, method, strength, preserve_luminosity,
                    neutralize, variance_scale):

        batch_size = target.shape[0]
        ref_batch_size = reference.shape[0]
        results = []

        for i in range(batch_size):
            # Get current target frame
            target_img = (target[i].cpu().numpy() * 255).astype(np.uint8)

            # Cycle through reference frames if needed
            ref_idx = i % ref_batch_size
            ref_img = (reference[ref_idx].cpu().numpy() * 255).astype(np.uint8)

            # Handle alpha channel
            has_alpha = target_img.shape[2] == 4
            if has_alpha:
                target_rgb = target_img[:, :, :3]
                target_alpha = target_img[:, :, 3]
            else:
                target_rgb = target_img

            ref_rgb = ref_img[:, :, :3] if ref_img.shape[2] == 4 else ref_img

            # Apply neutralization if requested
            if neutralize > 0:
                target_rgb = self._neutralize_grays(target_rgb, neutralize)

            # Apply color matching based on method
            if method == "lab_statistics":
                matched = self._lab_statistics_match(target_rgb, ref_rgb, variance_scale, preserve_luminosity)
            elif method == "lab_histogram":
                matched = self._lab_histogram_match(target_rgb, ref_rgb, preserve_luminosity)
            elif method == "mkl_transfer":
                matched = self._mkl_transfer(target_rgb, ref_rgb, preserve_luminosity)
            elif method == "wavelet_hybrid":
                matched = self._wavelet_hybrid_match(target_rgb, ref_rgb, preserve_luminosity)
            else:
                matched = target_rgb

            # Blend with original based on strength
            if strength < 1.0:
                matched = target_rgb * (1 - strength) + matched * strength

            matched = np.clip(matched, 0, 255).astype(np.uint8)

            # Reassemble with alpha if needed
            if has_alpha:
                result = np.dstack([matched, target_alpha])
            else:
                result = matched

            # Convert to tensor
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            results.append(result_tensor)

        # Stack all results
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)

    def _neutralize_grays(self, img, strength):
        """Neutralize gray tones by removing color cast"""
        # Convert to float
        img_f = img.astype(np.float32) / 255.0

        # Calculate perceived brightness
        luminance = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]

        # Find near-neutral pixels (low saturation)
        max_c = np.max(img_f, axis=2)
        min_c = np.min(img_f, axis=2)
        saturation = (max_c - min_c) / (max_c + 1e-8)

        # Create neutralization mask (stronger for low saturation pixels)
        neutral_mask = 1.0 - np.clip(saturation * 3, 0, 1)
        neutral_mask = neutral_mask[:, :, np.newaxis] * strength

        # Blend towards neutral gray
        gray = np.stack([luminance, luminance, luminance], axis=2)
        result = img_f * (1 - neutral_mask) + gray * neutral_mask

        return (result * 255).astype(np.uint8)

    def _lab_statistics_match(self, target, reference, variance_scale, preserve_luminosity):
        """
        PS-style LAB statistical color matching - FASTEST and MOST ACCURATE
        Matches mean and standard deviation in LAB color space
        """
        # Convert to LAB color space (float32 for precision)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Calculate statistics
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        ref_mean = np.mean(ref_lab, axis=(0, 1))
        ref_std = np.std(ref_lab, axis=(0, 1))

        # Apply statistical matching
        matched_lab = target_lab.copy()

        for c in range(3):
            if preserve_luminosity and c == 0:
                # Skip L channel if preserving luminosity
                continue

            # Standardize target, then scale and shift to match reference
            if target_std[c] > 1e-5:
                matched_lab[:, :, c] = (target_lab[:, :, c] - target_mean[c]) / target_std[c]
                matched_lab[:, :, c] = matched_lab[:, :, c] * (ref_std[c] * variance_scale) + ref_mean[c]
            else:
                matched_lab[:, :, c] = ref_mean[c]

        # Clamp LAB values to valid ranges
        matched_lab[:, :, 0] = np.clip(matched_lab[:, :, 0], 0, 100)
        matched_lab[:, :, 1] = np.clip(matched_lab[:, :, 1], -128, 127)
        matched_lab[:, :, 2] = np.clip(matched_lab[:, :, 2], -128, 127)

        # Convert back to RGB
        matched = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return matched

    def _lab_histogram_match(self, target, reference, preserve_luminosity):
        """
        LAB histogram matching - BETTER DETAIL PRESERVATION
        Uses cumulative histogram matching for each LAB channel
        """
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)

        matched_lab = target_lab.copy()

        for c in range(3):
            if preserve_luminosity and c == 0:
                continue

            # Perform histogram matching for this channel
            matched_lab[:, :, c] = self._match_histogram_channel(
                target_lab[:, :, c],
                ref_lab[:, :, c]
            )

        matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
        return matched

    def _match_histogram_channel(self, source, reference):
        """Match histogram of a single channel using cumulative distribution"""
        # Get the set of unique pixel values and their corresponding indices/counts
        s_values, s_idx, s_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
        r_values, r_counts = np.unique(reference.ravel(), return_counts=True)

        # Calculate cumulative distribution functions (CDF)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]

        r_quantiles = np.cumsum(r_counts).astype(np.float64)
        r_quantiles /= r_quantiles[-1]

        # Interpolate linearly to find the pixel values in the reference image
        # that correspond most closely to the quantiles in the source image
        interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

        # Return the interpolated pixel values
        return interp_r_values[s_idx].reshape(source.shape).astype(np.uint8)

    def _mkl_transfer(self, target, reference, preserve_luminosity):
        """
        Monge-Kantorovitch Linear color transfer - NATURAL and SMOOTH
        Best for photographic color grading
        """
        # Convert to float and flatten to pixel arrays
        target_f = target.reshape(-1, 3).astype(np.float32)
        ref_f = reference.reshape(-1, 3).astype(np.float32)

        # Convert to LAB
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)

        # Work on AB channels only if preserving luminosity
        if preserve_luminosity:
            target_work = target_lab[:, 1:]
            ref_work = ref_lab[:, 1:]
        else:
            target_work = target_lab
            ref_work = ref_lab

        # Calculate covariance matrices
        target_mean = np.mean(target_work, axis=0)
        ref_mean = np.mean(ref_work, axis=0)

        target_centered = target_work - target_mean
        ref_centered = ref_work - ref_mean

        target_cov = np.cov(target_centered.T)
        ref_cov = np.cov(ref_centered.T)

        # Add small epsilon to avoid numerical issues
        target_cov += np.eye(target_cov.shape[0]) * 1e-5
        ref_cov += np.eye(ref_cov.shape[0]) * 1e-5

        # Calculate the transfer matrix using Cholesky decomposition
        try:
            target_chol = linalg.cholesky(target_cov, lower=True)
            ref_chol = linalg.cholesky(ref_cov, lower=True)

            # Transform: T = ref_chol @ inv(target_chol)
            transform_matrix = ref_chol @ linalg.inv(target_chol)

            # Apply transformation
            transferred = (target_centered @ transform_matrix.T) + ref_mean
        except linalg.LinAlgError:
            # Fallback to simpler variance scaling if Cholesky fails
            target_std = np.std(target_work, axis=0) + 1e-8
            ref_std = np.std(ref_work, axis=0)
            transferred = (target_work - target_mean) * (ref_std / target_std) + ref_mean

        # Reconstruct LAB
        if preserve_luminosity:
            result_lab = np.column_stack([target_lab[:, 0], transferred])
        else:
            result_lab = transferred

        # Reshape and clamp
        result_lab = result_lab.reshape(target.shape[0], target.shape[1], 3)
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128, 127)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128, 127)

        # Convert back to RGB
        matched = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return matched

    def _wavelet_hybrid_match(self, target, reference, preserve_luminosity):
        """
        Wavelet + LAB hybrid matching - BEST FOR TEXTURES
        Preserves high-frequency details while matching color
        """
        # First, match low frequency (color) using LAB statistics
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Decompose into low and high frequency using Gaussian pyramid
        target_low = cv2.GaussianBlur(target_lab, (0, 0), 5)
        ref_low = cv2.GaussianBlur(ref_lab, (0, 0), 5)

        target_high = target_lab - target_low

        # Match low frequency statistics
        ref_mean = np.mean(ref_low, axis=(0, 1))
        ref_std = np.std(ref_low, axis=(0, 1))
        target_mean = np.mean(target_low, axis=(0, 1))
        target_std = np.std(target_low, axis=(0, 1))

        matched_low = target_low.copy()
        for c in range(3):
            if preserve_luminosity and c == 0:
                continue
            if target_std[c] > 1e-5:
                matched_low[:, :, c] = (target_low[:, :, c] - target_mean[c]) / target_std[c]
                matched_low[:, :, c] = matched_low[:, :, c] * ref_std[c] + ref_mean[c]

        # Reconstruct: matched_low + target_high (preserve texture)
        result_lab = matched_low + target_high

        # Clamp LAB values
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128, 127)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128, 127)

        # Convert back to RGB
        matched = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return matched


NODE_CLASS_MAPPINGS = {
    "RC_ColorMatch": RC_ColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RC_ColorMatch": "RC Color Match",
}
