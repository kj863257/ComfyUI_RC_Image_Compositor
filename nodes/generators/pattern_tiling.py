import math
from typing import Tuple

import cv2
import numpy as np
import torch


class RC_PatternTiling:
    """Generate a tiled pattern image with spacing, rotation, and opacity controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern_image": ("IMAGE", {}),
                "size_mode": (["from_image", "custom"], {"default": "custom"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "pattern_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "spacing_x": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "spacing_y": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "crop_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "crop_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "crop_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_pattern"
    CATEGORY = "RC/Generate"

    @staticmethod
    def _ensure_rgba(image: np.ndarray) -> np.ndarray:
        """Ensure the pattern image has four channels."""
        if image.shape[-1] == 4:
            return image
        alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype)
        return np.concatenate([image, alpha], axis=2)

    @staticmethod
    def _transform_pattern(pattern: np.ndarray, scale: float, rotation: float) -> np.ndarray:
        """Apply scaling and rotation to the base pattern."""
        pattern = np.clip(pattern, 0.0, 1.0).astype(np.float32)
        pattern = RC_PatternTiling._ensure_rgba(pattern)

        if not math.isclose(scale, 1.0, rel_tol=1e-3):
            new_w = max(1, int(round(pattern.shape[1] * scale)))
            new_h = max(1, int(round(pattern.shape[0] * scale)))
            pattern = cv2.resize(pattern, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if not math.isclose(rotation, 0.0, abs_tol=1e-3):
            height, width = pattern.shape[:2]
            center = (width / 2.0, height / 2.0)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)

            cos_val = abs(matrix[0, 0])
            sin_val = abs(matrix[0, 1])
            new_w = int(math.ceil((height * sin_val) + (width * cos_val)))
            new_h = int(math.ceil((height * cos_val) + (width * sin_val)))

            matrix[0, 2] += (new_w / 2.0) - center[0]
            matrix[1, 2] += (new_h / 2.0) - center[1]

            pattern = cv2.warpAffine(
                pattern,
                matrix,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0, 0.0, 0.0, 0.0),
            )

        return np.clip(pattern, 0.0, 1.0)

    @staticmethod
    def _determine_output_size(
        size_mode: str,
        width: int,
        height: int,
        reference_image: torch.Tensor,
    ) -> Tuple[int, int]:
        if size_mode == "from_image" and reference_image is not None:
            ref = reference_image[0]
            return int(ref.shape[1]), int(ref.shape[0])
        return max(1, int(width)), max(1, int(height))

    @staticmethod
    def _compute_index_range(
        length: int,
        tile_size: int,
        step: int,
        offset: int,
    ) -> range:
        start = math.floor((0 - offset - tile_size) / step)
        end = math.ceil((length - offset) / step)
        return range(start, end + 1)

    @staticmethod
    def _build_tiled_canvas(
        pattern: np.ndarray,
        out_w: int,
        out_h: int,
        spacing_x: int,
        spacing_y: int,
        offset_x: int,
        offset_y: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tile_h, tile_w = pattern.shape[:2]
        if tile_h < 1 or tile_w < 1:
            zeros_rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
            zeros_alpha = np.zeros((out_h, out_w, 1), dtype=np.float32)
            return zeros_rgb, zeros_alpha

        step_x = max(1, tile_w + spacing_x)
        step_y = max(1, tile_h + spacing_y)

        pattern_alpha = pattern[..., 3:4]
        pattern_rgb_pre = pattern[..., :3] * pattern_alpha

        canvas_rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
        canvas_alpha = np.zeros((out_h, out_w, 1), dtype=np.float32)

        y_indices = RC_PatternTiling._compute_index_range(out_h, tile_h, step_y, offset_y)
        x_indices = RC_PatternTiling._compute_index_range(out_w, tile_w, step_x, offset_x)

        for iy in y_indices:
            tile_top = iy * step_y + offset_y
            tile_bottom = tile_top + tile_h

            if tile_bottom <= 0 or tile_top >= out_h:
                continue

            clip_top = max(0, tile_top)
            clip_bottom = min(out_h, tile_bottom)
            tile_y0 = clip_top - tile_top
            tile_y1 = tile_y0 + (clip_bottom - clip_top)

            for ix in x_indices:
                tile_left = ix * step_x + offset_x
                tile_right = tile_left + tile_w

                if tile_right <= 0 or tile_left >= out_w:
                    continue

                clip_left = max(0, tile_left)
                clip_right = min(out_w, tile_right)
                tile_x0 = clip_left - tile_left
                tile_x1 = tile_x0 + (clip_right - clip_left)

                tile_alpha = pattern_alpha[tile_y0:tile_y1, tile_x0:tile_x1]
                if tile_alpha.size == 0:
                    continue

                if np.all(tile_alpha <= 0.0):
                    continue

                tile_rgb_pre = pattern_rgb_pre[tile_y0:tile_y1, tile_x0:tile_x1]

                region_rgb = canvas_rgb[clip_top:clip_bottom, clip_left:clip_right]
                region_alpha = canvas_alpha[clip_top:clip_bottom, clip_left:clip_right]

                inv_alpha = 1.0 - tile_alpha
                np.multiply(region_rgb, inv_alpha, out=region_rgb)
                region_rgb += tile_rgb_pre

                np.multiply(region_alpha, inv_alpha, out=region_alpha)
                region_alpha += tile_alpha

        return canvas_rgb, canvas_alpha

    @staticmethod
    def _apply_crop(image: np.ndarray, top: int, right: int, bottom: int, left: int) -> np.ndarray:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return image

        top = int(max(0, min(top, height - 1))) if height > 1 else 0
        left = int(max(0, min(left, width - 1))) if width > 1 else 0

        bottom_trim = int(max(0, bottom))
        right_trim = int(max(0, right))

        bottom_limit = max(top + 1, height - bottom_trim)
        right_limit = max(left + 1, width - right_trim)

        bottom_limit = min(bottom_limit, height)
        right_limit = min(right_limit, width)

        return image[top:bottom_limit, left:right_limit]

    def generate_pattern(
        self,
        pattern_image,
        size_mode,
        width,
        height,
        pattern_scale,
        rotation,
        spacing_x,
        spacing_y,
        offset_x,
        offset_y,
        opacity,
        crop_top,
        crop_right,
        crop_bottom,
        crop_left,
        reference_image=None,
    ):
        pattern_tensor = pattern_image[0].detach().cpu().numpy()
        pattern_tensor = self._apply_crop(
            pattern_tensor,
            crop_top,
            crop_right,
            crop_bottom,
            crop_left,
        )

        out_w, out_h = self._determine_output_size(size_mode, width, height, reference_image)

        if pattern_tensor.size == 0 or pattern_tensor.shape[0] == 0 or pattern_tensor.shape[1] == 0:
            empty = torch.zeros(
                (1, out_h, out_w, 4),
                dtype=pattern_image.dtype,
                device=pattern_image.device,
            )
            return (empty,)

        pattern = self._transform_pattern(pattern_tensor, pattern_scale, rotation)

        canvas_rgb, canvas_alpha = self._build_tiled_canvas(
            pattern,
            out_w,
            out_h,
            spacing_x,
            spacing_y,
            offset_x,
            offset_y,
        )

        alpha = np.clip(canvas_alpha, 0.0, 1.0)

        rgb = np.zeros_like(canvas_rgb)
        non_zero = alpha > 1e-6
        np.divide(canvas_rgb, alpha, out=rgb, where=non_zero)
        if np.any(~non_zero):
            rgb[np.repeat(~non_zero, 3, axis=2)] = 0.0

        np.clip(rgb, 0.0, 1.0, out=rgb)

        rgba = np.concatenate([rgb, alpha], axis=2)
        rgba[..., 3] = np.clip(rgba[..., 3] * opacity, 0.0, 1.0)

        rgba = np.ascontiguousarray(rgba, dtype=np.float32)

        image_out = torch.from_numpy(rgba).unsqueeze(0)

        return (image_out,)


NODE_CLASS_MAPPINGS = {
    "RC_PatternTiling": RC_PatternTiling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RC_PatternTiling": "RC Pattern Tiling",
}
