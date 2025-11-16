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

        # Convert to numpy in batch (much faster than per-frame)
        # IMPORTANT: Use .clone() to avoid modifying shared tensors in ComfyUI
        batch_size = target.shape[0]
        ref_batch_size = reference.shape[0]

        # Convert entire batch to uint8 numpy at once (creates new arrays)
        target_np = (target.clone().cpu().numpy() * 255).astype(np.uint8)
        reference_np = (reference.clone().cpu().numpy() * 255).astype(np.uint8)

        # Handle alpha channel
        has_alpha = target_np.shape[3] == 4
        if has_alpha:
            target_rgb = target_np[:, :, :, :3]
            target_alpha = target_np[:, :, :, 3:]
        else:
            target_rgb = target_np

        # Extract RGB from reference
        if reference_np.shape[3] == 4:
            reference_rgb = reference_np[:, :, :, :3]
        else:
            reference_rgb = reference_np

        # Pre-compute reference statistics (avoid recomputing for each target frame)
        ref_stats = self._precompute_reference_stats(reference_rgb, method, preserve_luminosity)

        # Process all frames
        results = []
        for i in range(batch_size):
            # Create copies to avoid modifying shared data
            target_frame = target_rgb[i].copy()
            ref_idx = i % ref_batch_size
            ref_frame = reference_rgb[ref_idx]

            # Apply neutralization if requested
            if neutralize > 0:
                target_frame = self._neutralize_grays(target_frame, neutralize)

            # Apply color matching based on method
            if method == "lab_statistics":
                matched = self._lab_statistics_match(
                    target_frame, ref_stats, variance_scale, preserve_luminosity
                )
            elif method == "lab_histogram":
                matched = self._lab_histogram_match(
                    target_frame, ref_frame, preserve_luminosity
                )
            elif method == "mkl_transfer":
                matched = self._mkl_transfer(
                    target_frame, ref_stats, preserve_luminosity
                )
            elif method == "wavelet_hybrid":
                matched = self._wavelet_hybrid_match(
                    target_frame, ref_stats, preserve_luminosity
                )
            else:
                matched = target_frame.copy()

            # Blend with original based on strength
            # Create new array instead of modifying in-place
            if strength < 1.0:
                original = target_rgb[i].astype(np.float32)
                matched_f = matched.astype(np.float32)
                matched = (original * (1 - strength) + matched_f * strength).astype(np.uint8)

            matched = np.clip(matched, 0, 255).astype(np.uint8)
            results.append(matched)

        # Stack results
        result_rgb = np.stack(results, axis=0)

        # Reassemble with alpha if needed
        if has_alpha:
            result_np = np.concatenate([result_rgb, target_alpha], axis=3)
        else:
            result_np = result_rgb

        # Convert back to torch tensor (single operation for entire batch)
        result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0)

        return (result_tensor,)

    def _precompute_reference_stats(self, reference_rgb, method, preserve_luminosity):
        """Pre-compute statistics from reference images to avoid redundant calculations"""
        stats = {}

        if method in ["lab_statistics", "wavelet_hybrid"]:
            # Compute LAB statistics for all reference frames
            ref_lab_all = []
            for i in range(reference_rgb.shape[0]):
                ref_lab = cv2.cvtColor(reference_rgb[i], cv2.COLOR_RGB2LAB).astype(np.float32)
                ref_lab_all.append(ref_lab)

            # Average statistics across all reference frames
            ref_lab_stack = np.stack(ref_lab_all, axis=0)
            stats['lab_mean'] = np.mean(ref_lab_stack, axis=(0, 1, 2))
            stats['lab_std'] = np.std(ref_lab_stack, axis=(0, 1, 2))

            if method == "wavelet_hybrid":
                # Pre-compute low frequency stats
                ref_low_all = []
                for ref_lab in ref_lab_all:
                    ref_low = cv2.GaussianBlur(ref_lab, (0, 0), 5)
                    ref_low_all.append(ref_low)
                ref_low_stack = np.stack(ref_low_all, axis=0)
                stats['low_mean'] = np.mean(ref_low_stack, axis=(0, 1, 2))
                stats['low_std'] = np.std(ref_low_stack, axis=(0, 1, 2))

        elif method == "mkl_transfer":
            # Compute covariance for all reference frames
            all_pixels = []
            for i in range(reference_rgb.shape[0]):
                ref_lab = cv2.cvtColor(reference_rgb[i], cv2.COLOR_RGB2LAB).astype(np.float32)
                pixels = ref_lab.reshape(-1, 3)

                if preserve_luminosity:
                    pixels = pixels[:, 1:]
                all_pixels.append(pixels)

            # Combine all reference pixels
            combined_pixels = np.vstack(all_pixels)
            stats['mean'] = np.mean(combined_pixels, axis=0)
            centered = combined_pixels - stats['mean']
            stats['cov'] = np.cov(centered.T)
            stats['cov'] += np.eye(stats['cov'].shape[0]) * 1e-5

            try:
                stats['chol'] = linalg.cholesky(stats['cov'], lower=True)
            except linalg.LinAlgError:
                stats['std'] = np.std(combined_pixels, axis=0) + 1e-8

        return stats

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

    def _lab_statistics_match(self, target, ref_stats, variance_scale, preserve_luminosity):
        """
        PS-style LAB statistical color matching - FASTEST and MOST ACCURATE
        Uses pre-computed reference statistics for better performance
        """
        # Convert target to LAB
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Calculate target statistics
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))

        # Apply statistical matching
        matched_lab = target_lab.copy()

        for c in range(3):
            if preserve_luminosity and c == 0:
                continue

            if target_std[c] > 1e-5:
                matched_lab[:, :, c] = (target_lab[:, :, c] - target_mean[c]) / target_std[c]
                matched_lab[:, :, c] = matched_lab[:, :, c] * (ref_stats['lab_std'][c] * variance_scale) + ref_stats['lab_mean'][c]
            else:
                matched_lab[:, :, c] = ref_stats['lab_mean'][c]

        # Clamp to OpenCV LAB valid range [0, 255]
        matched_lab = np.clip(matched_lab, 0, 255)

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

    def _mkl_transfer(self, target, ref_stats, preserve_luminosity):
        """
        Monge-Kantorovitch Linear color transfer - NATURAL and SMOOTH
        Uses pre-computed reference covariance matrix
        """
        # Convert target to LAB
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_pixels = target_lab.reshape(-1, 3)

        # Work on AB channels only if preserving luminosity
        if preserve_luminosity:
            target_work = target_pixels[:, 1:]
        else:
            target_work = target_pixels

        # Calculate target statistics
        target_mean = np.mean(target_work, axis=0)
        target_centered = target_work - target_mean

        target_cov = np.cov(target_centered.T)
        target_cov += np.eye(target_cov.shape[0]) * 1e-5

        # Apply transformation using pre-computed reference stats
        if 'chol' in ref_stats:
            try:
                target_chol = linalg.cholesky(target_cov, lower=True)
                transform_matrix = ref_stats['chol'] @ linalg.inv(target_chol)
                transferred = (target_centered @ transform_matrix.T) + ref_stats['mean']
            except linalg.LinAlgError:
                # Fallback to variance scaling
                target_std = np.std(target_work, axis=0) + 1e-8
                transferred = (target_work - target_mean) * (ref_stats['std'] / target_std) + ref_stats['mean']
        else:
            # Use fallback method
            target_std = np.std(target_work, axis=0) + 1e-8
            transferred = (target_work - target_mean) * (ref_stats['std'] / target_std) + ref_stats['mean']

        # Reconstruct LAB
        if preserve_luminosity:
            result_pixels = np.column_stack([target_pixels[:, 0], transferred])
        else:
            result_pixels = transferred

        # Reshape and clamp to OpenCV LAB range [0, 255]
        result_lab = result_pixels.reshape(target.shape[0], target.shape[1], 3)
        result_lab = np.clip(result_lab, 0, 255)

        # Convert back to RGB
        matched = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return matched

    def _wavelet_hybrid_match(self, target, ref_stats, preserve_luminosity):
        """
        Wavelet + LAB hybrid matching - BEST FOR TEXTURES
        Uses pre-computed reference low-frequency statistics
        """
        # Convert target to LAB
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Decompose into low and high frequency
        target_low = cv2.GaussianBlur(target_lab, (0, 0), 5)
        target_high = target_lab - target_low

        # Match low frequency using pre-computed stats
        target_mean = np.mean(target_low, axis=(0, 1))
        target_std = np.std(target_low, axis=(0, 1))

        matched_low = target_low.copy()
        for c in range(3):
            if preserve_luminosity and c == 0:
                continue
            if target_std[c] > 1e-5:
                matched_low[:, :, c] = (target_low[:, :, c] - target_mean[c]) / target_std[c]
                matched_low[:, :, c] = matched_low[:, :, c] * ref_stats['low_std'][c] + ref_stats['low_mean'][c]

        # Reconstruct: matched_low + target_high (preserve texture)
        result_lab = matched_low + target_high

        # Clamp to OpenCV LAB range [0, 255]
        result_lab = np.clip(result_lab, 0, 255)

        # Convert back to RGB
        matched = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return matched


NODE_CLASS_MAPPINGS = {
    "RC_ColorMatch": RC_ColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RC_ColorMatch": "RC Color Match",
}
