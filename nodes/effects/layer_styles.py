import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from typing import Tuple, Optional
import cv2


class RC_DropShadow:
    """Drop Shadow layer style with Photoshop-compatible parameters"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_drop_shadow"
    CATEGORY = "RC/Layer Effects"
    DESCRIPTION = "Apply Drop Shadow effect to image with blur, offset, and color controls."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "distance": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Shadow distance in pixels"
                }),
                "angle": ("FLOAT", {
                    "default": 135.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Shadow angle in degrees (135° = bottom-right)"
                }),
                "size": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Shadow blur size in pixels"
                }),
                "spread": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Shadow spread (choke) in pixels"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Shadow opacity"
                }),
                "fill_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Original image fill opacity:\n"
                        "- 1.0: Normal image display\n"
                        "- 0.0: Hollow effect, shadow only\n"
                        "- 0.5: Semi-transparent image + shadow"
                    )
                }),
                "color_r": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Shadow color red component (0-255)"
                }),
                "color_g": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Shadow color green component (0-255)"
                }),
                "color_b": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Shadow color blue component (0-255)"
                }),
                "blend_mode": ([
                    "normal", "multiply", "color_burn", "linear_burn", "darken"
                ], {
                    "default": "multiply",
                    "tooltip": (
                        "Drop shadow blend mode:\n"
                        "- normal: Normal blending\n"
                        "- multiply: Multiply, classic shadow\n"
                        "- color_burn: Color burn, deep shadow\n"
                        "- linear_burn: Linear burn, soft shadow\n"
                        "- darken: Darken, gentle shadow"
                    )
                }),
                "auto_expand_canvas": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Auto expand canvas:\n"
                        "- True: Auto expand canvas when shadow exceeds bounds\n"
                        "- False: Shadow is clipped to original canvas size"
                    )
                }),
            }
        }

    def apply_drop_shadow(self, image, distance, angle, size, spread, opacity,
                         fill_opacity, color_r, color_g, color_b, blend_mode, auto_expand_canvas):
        batch_size = image.shape[0]
        results = []

        # Calculate offset based on angle and distance (same for all images)
        angle_rad = np.radians(angle)
        offset_x = int(distance * np.cos(angle_rad))
        offset_y = int(distance * np.sin(angle_rad))

        # Calculate canvas expansion (same for all images)
        canvas_expansion = int(abs(offset_x)) + int(abs(offset_y)) + int(size * 4) if auto_expand_canvas else int(size * 4)

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            h, w = img.shape[:2]
            has_alpha = img.shape[2] == 4

            # Create shadow mask from alpha channel or full opacity
            if has_alpha:
                mask = img[:, :, 3]
            else:
                mask = np.full((h, w), 255, dtype=np.uint8)

            # Apply spread (dilate mask)
            if spread > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (int(spread*2)+1, int(spread*2)+1))
                mask = cv2.dilate(mask, kernel, iterations=1)

            # Apply blur to create shadow
            if size > 0:
                # Convert to float for proper Gaussian blur
                mask_float = mask.astype(np.float32)
                blur_size = int(size * 6)  # Gaussian blur kernel size
                if blur_size % 2 == 0:
                    blur_size += 1
                mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), size)
                mask = mask_float.astype(np.uint8)

            # Create shadow image with expanded canvas
            canvas_w = w + canvas_expansion * 2
            canvas_h = h + canvas_expansion * 2
            shadow_canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

            # Position shadow on canvas
            if auto_expand_canvas:
                shadow_x = canvas_expansion + offset_x
                shadow_y = canvas_expansion + offset_y
            else:
                shadow_x = int(size * 2) + max(0, -offset_x) + offset_x
                shadow_y = int(size * 2) + max(0, -offset_y) + offset_y

            # Create mask for shadow calculation
            if auto_expand_canvas:
                # Work on expanded canvas
                shadow_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                orig_pos_x = canvas_expansion
                orig_pos_y = canvas_expansion
                shadow_mask[orig_pos_y:orig_pos_y+h, orig_pos_x:orig_pos_x+w] = mask

                # Apply spread (dilate mask) on expanded canvas
                if spread > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (int(spread*2)+1, int(spread*2)+1))
                    shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)

                # Apply blur to create shadow on expanded canvas
                if size > 0:
                    mask_float = shadow_mask.astype(np.float32)
                    blur_size = int(size * 6)
                    if blur_size % 2 == 0:
                        blur_size += 1
                    mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), size)
                    final_shadow_mask = mask_float.astype(np.uint8)
                else:
                    final_shadow_mask = shadow_mask

                # Offset the shadow mask to shadow position
                offset_shadow_mask = np.zeros_like(final_shadow_mask)
                if shadow_x >= 0 and shadow_y >= 0:
                    offset_shadow_mask[shadow_y:min(canvas_h, shadow_y+h), shadow_x:min(canvas_w, shadow_x+w)] = \
                        final_shadow_mask[orig_pos_y:min(canvas_h-shadow_y+orig_pos_y, orig_pos_y+h),
                                         orig_pos_x:min(canvas_w-shadow_x+orig_pos_x, orig_pos_x+w)]

                # Remove original shape area from shadow (avoid shadow under original image)
                original_mask_canvas = np.zeros_like(offset_shadow_mask)
                original_mask_canvas[orig_pos_y:orig_pos_y+h, orig_pos_x:orig_pos_x+w] = mask
                offset_shadow_mask = cv2.subtract(offset_shadow_mask, original_mask_canvas)

                # Apply shadow color and opacity to entire canvas
                shadow_color = np.array([color_r, color_g, color_b])
                for c in range(3):
                    shadow_canvas[:, :, c] = shadow_color[c]
                shadow_canvas[:, :, 3] = (offset_shadow_mask * opacity).astype(np.uint8)

            else:
                # Original behavior for bounded shadow
                # Apply shadow color and opacity
                shadow_color = np.array([color_r, color_g, color_b])
                for c in range(3):
                    shadow_canvas[shadow_y:shadow_y+h, shadow_x:shadow_x+w, c] = shadow_color[c]
                shadow_canvas[shadow_y:shadow_y+h, shadow_x:shadow_x+w, 3] = (mask * opacity).astype(np.uint8)

            # Position original image on canvas
            if auto_expand_canvas:
                orig_x = canvas_expansion
                orig_y = canvas_expansion
            else:
                orig_x = int(size * 2) + max(0, -offset_x)
                orig_y = int(size * 2) + max(0, -offset_y)

            # Create original image with alpha
            if not has_alpha:
                orig_rgba = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
            else:
                orig_rgba = img.copy()

            # Apply fill_opacity to original image
            orig_rgba[:, :, 3] = (orig_rgba[:, :, 3].astype(np.float32) * fill_opacity).astype(np.uint8)

            # Composite original over shadow
            result = shadow_canvas.copy()

            # Extract regions
            orig_region = result[orig_y:orig_y+h, orig_x:orig_x+w]
            shadow_alpha = orig_region[:, :, 3:4].astype(np.float32) / 255.0
            orig_alpha = orig_rgba[:, :, 3:4].astype(np.float32) / 255.0

            # Blend shadow with original
            combined_alpha = shadow_alpha + orig_alpha * (1 - shadow_alpha)

            # Apply blend mode for shadow
            shadow_rgb = orig_region[:, :, :3].astype(np.float32) / 255.0
            orig_rgb = orig_rgba[:, :, :3].astype(np.float32) / 255.0

            if blend_mode == "multiply":
                blended_rgb = shadow_rgb * orig_rgb
            elif blend_mode == "color_burn":
                blended_rgb = 1.0 - np.minimum((1.0 - shadow_rgb) / (orig_rgb + 1e-6), 1.0)
            elif blend_mode == "linear_burn":
                blended_rgb = shadow_rgb + orig_rgb - 1.0
            elif blend_mode == "darken":
                blended_rgb = np.minimum(shadow_rgb, orig_rgb)
            else:  # normal
                blended_rgb = orig_rgb

            blended_rgb = np.clip(blended_rgb, 0.0, 1.0)

            # Composite final result
            final_rgb = shadow_rgb * (1 - orig_alpha) + blended_rgb * orig_alpha
            final_rgb = np.clip(final_rgb * 255, 0, 255).astype(np.uint8)
            combined_alpha = np.clip(combined_alpha * 255, 0, 255).astype(np.uint8)

            result[orig_y:orig_y+h, orig_x:orig_x+w, :3] = final_rgb
            result[orig_y:orig_y+h, orig_x:orig_x+w, 3:4] = combined_alpha

            # Convert back to tensor for this image
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)


class RC_Stroke:
    """Stroke layer style with inside/outside/center positioning"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_stroke"
    CATEGORY = "RC/Layer Effects"
    DESCRIPTION = "Apply Stroke effect to image with customizable position, size, and color."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Stroke width in pixels"
                }),
                "position": (["outside", "inside", "center"], {
                    "default": "outside",
                    "tooltip": "Stroke position relative to shape"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Stroke opacity"
                }),
                "color_r": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Stroke color red component (0-255)"
                }),
                "color_g": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Stroke color green component (0-255)"
                }),
                "color_b": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Stroke color blue component (0-255)"
                }),
                "blend_mode": ([
                    "normal", "multiply", "screen", "overlay", "color_burn"
                ], {
                    "default": "normal",
                    "tooltip": (
                        "Stroke blend mode:\n"
                        "- normal: Normal blending, direct overlay\n"
                        "- multiply: Multiply, darkens colors\n"
                        "- screen: Screen, lightens colors\n"
                        "- overlay: Overlay, enhances contrast\n"
                        "- color_burn: Color burn, deeper shadows"
                    )
                }),
                "fill_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Original image fill opacity:\n"
                        "- 1.0: Normal image display\n"
                        "- 0.0: Hollow effect, stroke only\n"
                        "- 0.5: Semi-transparent image + stroke"
                    )
                }),
                "auto_expand_canvas": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Auto expand canvas:\n"
                        "- True: Auto expand canvas when stroke exceeds bounds\n"
                        "- False: Stroke is clipped to original canvas size"
                    )
                }),
            }
        }

    def apply_stroke(self, image, size, position, opacity, color_r, color_g, color_b, blend_mode, fill_opacity, auto_expand_canvas):
        # Get stroke size (same for all images)
        stroke_size = int(size)
        if stroke_size == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        # Create kernels (same for all images)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (stroke_size*2+1, stroke_size*2+1))
        half_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (stroke_size+1, stroke_size+1))

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            h, w = img.shape[:2]
            has_alpha = img.shape[2] == 4

            # Get alpha mask
            if has_alpha:
                mask = img[:, :, 3]
            else:
                mask = np.full((h, w), 255, dtype=np.uint8)

            # Create stroke mask based on position
            if position == "outside":
                # Dilate then subtract original
                dilated = cv2.dilate(mask, kernel, iterations=1)
                stroke_mask = cv2.subtract(dilated, mask)
                canvas_expansion = stroke_size * 2 if auto_expand_canvas else 0
            elif position == "inside":
                # Erode then subtract from original
                eroded = cv2.erode(mask, kernel, iterations=1)
                stroke_mask = cv2.subtract(mask, eroded)
                canvas_expansion = 0  # Inside stroke doesn't need expansion
            else:  # center
                # Half dilate, half erode
                dilated = cv2.dilate(mask, half_kernel, iterations=1)
                eroded = cv2.erode(mask, half_kernel, iterations=1)
                stroke_mask = cv2.subtract(dilated, eroded)
                canvas_expansion = stroke_size if auto_expand_canvas else 0

            # Create result canvas
            canvas_w = w + canvas_expansion * 2
            canvas_h = h + canvas_expansion * 2
            result = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

            # Position on canvas
            pos_x = canvas_expansion
            pos_y = canvas_expansion

            # Apply stroke
            stroke_color = np.array([color_r, color_g, color_b])

            if canvas_expansion > 0:
                # Need to create expanded stroke mask
                expanded_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                expanded_mask[pos_y:pos_y+h, pos_x:pos_x+w] = mask

                # Recalculate stroke on expanded canvas
                if position == "outside":
                    dilated = cv2.dilate(expanded_mask, kernel, iterations=1)
                    stroke_mask = cv2.subtract(dilated, expanded_mask)
                elif position == "center":
                    dilated = cv2.dilate(expanded_mask, half_kernel, iterations=1)
                    eroded = cv2.erode(expanded_mask, half_kernel, iterations=1)
                    stroke_mask = cv2.subtract(dilated, eroded)

                # Apply stroke to full canvas
                for c in range(3):
                    result[:, :, c] = stroke_color[c]
                result[:, :, 3] = (stroke_mask * opacity).astype(np.uint8)
            else:
                # Original behavior for no expansion
                for c in range(3):
                    result[pos_y:pos_y+h, pos_x:pos_x+w, c] = stroke_color[c]
                result[pos_y:pos_y+h, pos_x:pos_x+w, 3] = (stroke_mask * opacity).astype(np.uint8)

            # Composite original image
            if not has_alpha:
                orig_rgba = np.dstack([img, mask])
            else:
                orig_rgba = img.copy()

            # Apply fill_opacity to original image
            orig_rgba[:, :, 3] = (orig_rgba[:, :, 3].astype(np.float32) * fill_opacity).astype(np.uint8)

            # Blend original over stroke (or stroke over original for inside position)
            orig_alpha = orig_rgba[:, :, 3:4].astype(np.float32) / 255.0
            stroke_region = result[pos_y:pos_y+h, pos_x:pos_x+w]
            stroke_alpha = stroke_region[:, :, 3:4].astype(np.float32) / 255.0

            # For inside stroke, render stroke ABOVE original image
            if position == "inside":
                combined_alpha = orig_alpha + stroke_alpha * (1 - orig_alpha)

                # Apply blending
                stroke_rgb = stroke_region[:, :, :3].astype(np.float32) / 255.0
                orig_rgb = orig_rgba[:, :, :3].astype(np.float32) / 255.0

                # Stroke over original
                final_rgb = orig_rgb * (1 - stroke_alpha) + stroke_rgb * stroke_alpha
            else:
                # For outside and center stroke, render original ABOVE stroke
                combined_alpha = stroke_alpha + orig_alpha * (1 - stroke_alpha)

                # Apply blending
                stroke_rgb = stroke_region[:, :, :3].astype(np.float32) / 255.0
                orig_rgb = orig_rgba[:, :, :3].astype(np.float32) / 255.0

                # Original over stroke
                final_rgb = stroke_rgb * (1 - orig_alpha) + orig_rgb * orig_alpha

            final_rgb = np.clip(final_rgb * 255, 0, 255).astype(np.uint8)
            combined_alpha = np.clip(combined_alpha * 255, 0, 255).astype(np.uint8)

            result[pos_y:pos_y+h, pos_x:pos_x+w, :3] = final_rgb
            result[pos_y:pos_y+h, pos_x:pos_x+w, 3:4] = combined_alpha

            # Convert back to tensor for this image
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)


class RC_OuterGlow:
    """Outer Glow layer style with customizable color and spread"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_outer_glow"
    CATEGORY = "RC/Layer Effects"
    DESCRIPTION = "Apply Outer Glow effect with color, size, and spread controls."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Glow blur size in pixels"
                }),
                "spread": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Glow spread (choke) in pixels"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Glow opacity"
                }),
                "color_r": ("INT", {
                    "default": 255, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Glow color red component (0-255)"
                }),
                "color_g": ("INT", {
                    "default": 255, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Glow color green component (0-255)"
                }),
                "color_b": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Glow color blue component (0-255)"
                }),
                "blend_mode": ([
                    "normal", "screen", "color_dodge", "linear_dodge", "lighten"
                ], {
                    "default": "screen",
                    "tooltip": (
                        "Outer glow blend mode:\n"
                        "- normal: Normal blending\n"
                        "- screen: Screen, classic glow effect\n"
                        "- color_dodge: Color dodge, intense glow\n"
                        "- linear_dodge: Linear dodge, soft glow\n"
                        "- lighten: Lighten, gentle glow"
                    )
                }),
                "fill_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Original image fill opacity:\n"
                        "- 1.0: Normal image display\n"
                        "- 0.0: Hollow effect, glow only\n"
                        "- 0.5: Semi-transparent image + glow"
                    )
                }),
                "auto_expand_canvas": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Auto expand canvas:\n"
                        "- True: Auto expand canvas when glow exceeds bounds\n"
                        "- False: Glow is clipped to original canvas size"
                    )
                }),
            }
        }

    def apply_outer_glow(self, image, size, spread, opacity, color_r, color_g, color_b, blend_mode, fill_opacity, auto_expand_canvas):
        batch_size = image.shape[0]
        results = []

        # Calculate canvas expansion (same for all images)
        canvas_expansion = int(size * 4 + spread * 2) if auto_expand_canvas else 0

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            h, w = img.shape[:2]
            has_alpha = img.shape[2] == 4

            # Get alpha mask
            if has_alpha:
                mask = img[:, :, 3]
            else:
                mask = np.full((h, w), 255, dtype=np.uint8)

            # Apply spread (dilate mask)
            if spread > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (int(spread*2)+1, int(spread*2)+1))
                mask = cv2.dilate(mask, kernel, iterations=1)

            # Apply blur to create glow
            if size > 0:
                mask_float = mask.astype(np.float32)
                blur_size = int(size * 6)
                if blur_size % 2 == 0:
                    blur_size += 1
                mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), size)
                mask = mask_float.astype(np.uint8)

            # Create glow canvas
            canvas_w = w + canvas_expansion * 2
            canvas_h = h + canvas_expansion * 2
            glow_canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

            # Position glow on canvas
            glow_x = canvas_expansion
            glow_y = canvas_expansion

            # Create mask for glow calculation
            if canvas_expansion > 0:
                # Work on expanded canvas
                glow_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                glow_mask[glow_y:glow_y+h, glow_x:glow_x+w] = mask

                # Apply spread (dilate mask) on expanded canvas
                if spread > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (int(spread*2)+1, int(spread*2)+1))
                    glow_mask = cv2.dilate(glow_mask, kernel, iterations=1)

                # Apply blur to create glow on expanded canvas
                if size > 0:
                    mask_float = glow_mask.astype(np.float32)
                    blur_size = int(size * 6)
                    if blur_size % 2 == 0:
                        blur_size += 1
                    mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), size)
                    final_mask = mask_float.astype(np.uint8)
                else:
                    final_mask = glow_mask

                # Remove original shape area from glow (outer glow only)
                # Use original alpha channel, not processed mask
                if has_alpha:
                    original_alpha = img[:, :, 3]
                else:
                    original_alpha = np.full((h, w), 255, dtype=np.uint8)

                original_mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                original_mask_canvas[glow_y:glow_y+h, glow_x:glow_x+w] = original_alpha
                final_mask = cv2.subtract(final_mask, original_mask_canvas)

                # Apply glow color and opacity to entire canvas
                glow_color = np.array([color_r, color_g, color_b])
                for c in range(3):
                    glow_canvas[:, :, c] = glow_color[c]
                glow_canvas[:, :, 3] = (final_mask * opacity).astype(np.uint8)

            else:
                # Original behavior - work within original dimensions
                work_mask = mask.copy()

                # Apply spread (dilate mask)
                if spread > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (int(spread*2)+1, int(spread*2)+1))
                    work_mask = cv2.dilate(work_mask, kernel, iterations=1)

                # Apply blur to create glow
                if size > 0:
                    mask_float = work_mask.astype(np.float32)
                    blur_size = int(size * 6)
                    if blur_size % 2 == 0:
                        blur_size += 1
                    mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), size)
                    final_mask = mask_float.astype(np.uint8)
                else:
                    final_mask = work_mask

                # Remove original shape area from glow (outer glow only)
                # Use original alpha channel, not processed mask
                if has_alpha:
                    original_alpha = img[:, :, 3]
                else:
                    original_alpha = np.full((h, w), 255, dtype=np.uint8)

                final_mask = cv2.subtract(final_mask, original_alpha)

                # Apply glow color and opacity
                glow_color = np.array([color_r, color_g, color_b])
                for c in range(3):
                    glow_canvas[glow_y:glow_y+h, glow_x:glow_x+w, c] = glow_color[c]
                glow_canvas[glow_y:glow_y+h, glow_x:glow_x+w, 3] = (final_mask * opacity).astype(np.uint8)

            # Position original image on canvas
            orig_x = canvas_expansion
            orig_y = canvas_expansion

            # Create original image with alpha
            if not has_alpha:
                orig_rgba = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
            else:
                orig_rgba = img.copy()

            # Apply fill_opacity to original image
            orig_rgba[:, :, 3] = (orig_rgba[:, :, 3].astype(np.float32) * fill_opacity).astype(np.uint8)

            # Composite original over glow
            result = glow_canvas.copy()
            glow_region = result[orig_y:orig_y+h, orig_x:orig_x+w]

            glow_alpha = glow_region[:, :, 3:4].astype(np.float32) / 255.0
            orig_alpha = orig_rgba[:, :, 3:4].astype(np.float32) / 255.0

            combined_alpha = glow_alpha + orig_alpha * (1 - glow_alpha)

            # Apply blend mode
            glow_rgb = glow_region[:, :, :3].astype(np.float32) / 255.0
            orig_rgb = orig_rgba[:, :, :3].astype(np.float32) / 255.0

            if blend_mode == "screen":
                blended_rgb = 1.0 - (1.0 - glow_rgb) * (1.0 - orig_rgb)
            elif blend_mode == "color_dodge":
                blended_rgb = np.minimum(glow_rgb / (1.0 - orig_rgb + 1e-6), 1.0)
            elif blend_mode == "linear_dodge":
                blended_rgb = glow_rgb + orig_rgb
            elif blend_mode == "lighten":
                blended_rgb = np.maximum(glow_rgb, orig_rgb)
            else:  # normal
                blended_rgb = orig_rgb

            blended_rgb = np.clip(blended_rgb, 0.0, 1.0)

            # Composite final result
            final_rgb = glow_rgb * (1 - orig_alpha) + blended_rgb * orig_alpha
            final_rgb = np.clip(final_rgb * 255, 0, 255).astype(np.uint8)
            combined_alpha = np.clip(combined_alpha * 255, 0, 255).astype(np.uint8)

            result[orig_y:orig_y+h, orig_x:orig_x+w, :3] = final_rgb
            result[orig_y:orig_y+h, orig_x:orig_x+w, 3:4] = combined_alpha

            # Convert back to tensor for this image
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)