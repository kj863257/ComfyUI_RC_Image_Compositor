import torch
import numpy as np
import json


class RC_GradientMap:
    """RC 渐变映射 | RC Gradient Map

    High-performance gradient mapping using vectorized NumPy operations:
    - Vectorized LUT creation with np.interp()
    - Vectorized LUT application using array indexing
    - Vectorized color space conversions (RGB ↔ HSL)
    - Vectorized blend mode operations
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to apply gradient mapping"
                }),
                "gradient_data": ("STRING", {
                    "default": '{"stops": [{"position": 0.0, "color": [0, 0, 0, 255]}, {"position": 1.0, "color": [255, 255, 255, 255]}]}',
                    "tooltip": "Gradient data in JSON format"
                }),
                "blend_mode": ([
                    "normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
                    "color_dodge", "color_burn", "darken", "lighten", "difference",
                    "exclusion", "hue", "saturation", "color", "luminosity"
                ], {
                    "default": "normal",
                    "tooltip": "Blending mode for gradient mapping"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Opacity of the gradient mapping effect"
                }),
                "preserve_luminosity": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve original image luminosity when mapping"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_gradient_map"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Apply gradient mapping to image based on luminosity values"

    def apply_gradient_map(self, image, gradient_data, blend_mode, opacity, preserve_luminosity):
        """Apply gradient mapping to the input image"""

        # Convert torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        # Ensure image is in the correct format [batch, height, width, channels]
        if len(image_np.shape) == 3:
            image_np = np.expand_dims(image_np, axis=0)

        batch_size, height, width, channels = image_np.shape
        result_images = []

        for b in range(batch_size):
            img = image_np[b]

            # Apply gradient mapping
            mapped_img = self.gradient_map_image(img, gradient_data, preserve_luminosity)

            # Apply blending
            blended_img = self.blend_images(img, mapped_img, blend_mode, opacity)

            result_images.append(blended_img)

        result = np.stack(result_images, axis=0)

        # Convert back to torch tensor
        return (torch.from_numpy(result).float(),)

    def gradient_map_image(self, image, gradient_data, preserve_luminosity):
        """Apply gradient mapping to a single image"""
        try:
            gradient_settings = json.loads(gradient_data)
            stops = gradient_settings.get("stops", [])
        except:
            # Default gradient: black to white
            stops = [
                {"position": 0.0, "color": [0, 0, 0, 255]},
                {"position": 1.0, "color": [255, 255, 255, 255]}
            ]

        # Calculate luminosity using ITU-R BT.601 weights
        if image.shape[2] >= 3:
            luminosity = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            # Grayscale image
            luminosity = image[:, :, 0]

        # Create gradient lookup table (LUT)
        lut = self.create_gradient_lut(stops)

        # Map luminosity values to gradient colors
        mapped_colors = self.apply_lut(luminosity, lut)

        # Preserve original luminosity if requested
        if preserve_luminosity and image.shape[2] >= 3:
            mapped_colors = self.preserve_image_luminosity(image, mapped_colors, luminosity)

        return mapped_colors

    def create_gradient_lut(self, stops):
        """Create a 256-entry lookup table from gradient stops (vectorized)"""
        # Sort stops by position
        stops = sorted(stops, key=lambda x: x["position"])

        # Convert to arrays for vectorized operations
        positions = np.array([stop["position"] for stop in stops])
        colors = np.array([stop["color"][:3] for stop in stops], dtype=np.float32) / 255.0

        # Create position array for LUT
        lut_positions = np.linspace(0, 1, 256)

        # Vectorized interpolation
        lut = np.zeros((256, 3), dtype=np.float32)

        for i in range(3):  # For each color channel
            lut[:, i] = np.interp(lut_positions, positions, colors[:, i])

        return lut

    def apply_lut(self, luminosity, lut):
        """Apply the gradient LUT to luminosity values (vectorized)"""
        # Convert luminosity to 0-255 range
        lum_indices = np.clip(luminosity * 255, 0, 255).astype(np.int32)

        # Vectorized lookup - much faster than nested loops
        mapped = lut[lum_indices]

        return mapped

    def preserve_image_luminosity(self, original, mapped, original_luminosity):
        """Preserve the original image's luminosity while using mapped colors"""
        # Convert mapped colors to HSL
        mapped_hsl = self.rgb_to_hsl(mapped)

        # Replace luminosity with original
        mapped_hsl[:, :, 2] = original_luminosity

        # Convert back to RGB
        result = self.hsl_to_rgb(mapped_hsl)

        return result

    def rgb_to_hsl(self, rgb):
        """Convert RGB to HSL"""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val

        # Luminosity
        l = (max_val + min_val) / 2.0

        # Saturation
        s = np.zeros_like(l)
        non_zero = delta > 1e-10
        light_mask = (l <= 0.5) & non_zero
        s[light_mask] = delta[light_mask] / (max_val[light_mask] + min_val[light_mask])
        heavy_mask = (l > 0.5) & non_zero
        s[heavy_mask] = delta[heavy_mask] / (2.0 - max_val[heavy_mask] - min_val[heavy_mask])

        # Hue
        h = np.zeros_like(l)
        if np.any(non_zero):
            delta_safe = np.where(non_zero, delta, 1.0)
            idx_r = non_zero & (max_val == r)
            idx_g = non_zero & (max_val == g)
            idx_b = non_zero & (max_val == b)

            h[idx_r] = (g[idx_r] - b[idx_r]) / delta_safe[idx_r]
            h[idx_g] = 2.0 + (b[idx_g] - r[idx_g]) / delta_safe[idx_g]
            h[idx_b] = 4.0 + (r[idx_b] - g[idx_b]) / delta_safe[idx_b]

            h = (h / 6.0) % 1.0

        return np.stack([h, s, l], axis=2)

    def hsl_to_rgb(self, hsl):
        """Convert HSL to RGB"""
        h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]

        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs(((h * 6) % 2) - 1))
        m = l - c / 2

        rgb = np.zeros_like(hsl)

        # Determine RGB values based on hue
        mask1 = (h >= 0) & (h < 1/6)
        mask2 = (h >= 1/6) & (h < 2/6)
        mask3 = (h >= 2/6) & (h < 3/6)
        mask4 = (h >= 3/6) & (h < 4/6)
        mask5 = (h >= 4/6) & (h < 5/6)
        mask6 = (h >= 5/6) & (h <= 1)

        rgb[:, :, 0] = np.where(mask1, c, np.where(mask2, x, np.where(mask3, 0, np.where(mask4, 0, np.where(mask5, x, c)))))
        rgb[:, :, 1] = np.where(mask1, x, np.where(mask2, c, np.where(mask3, c, np.where(mask4, x, np.where(mask5, 0, 0)))))
        rgb[:, :, 2] = np.where(mask1, 0, np.where(mask2, 0, np.where(mask3, x, np.where(mask4, c, np.where(mask5, c, x)))))

        rgb += m[:, :, np.newaxis]

        return np.clip(rgb, 0, 1)

    def blend_images(self, base, overlay, blend_mode, opacity):
        """Blend the mapped image with the original using specified blend mode"""
        if blend_mode == "normal":
            result = overlay
        elif blend_mode == "multiply":
            result = base * overlay
        elif blend_mode == "screen":
            result = 1.0 - (1.0 - base) * (1.0 - overlay)
        elif blend_mode == "overlay":
            mask = base < 0.5
            result = np.where(mask, 2 * base * overlay, 1.0 - 2 * (1.0 - base) * (1.0 - overlay))
        elif blend_mode == "soft_light":
            mask = overlay < 0.5
            result = np.where(mask,
                            base - (1.0 - 2 * overlay) * base * (1.0 - base),
                            base + (2 * overlay - 1.0) * (np.sqrt(base) - base))
        elif blend_mode == "hard_light":
            mask = overlay < 0.5
            result = np.where(mask, 2 * base * overlay, 1.0 - 2 * (1.0 - base) * (1.0 - overlay))
        elif blend_mode == "color_dodge":
            result = np.where(overlay >= 1.0, 1.0, np.minimum(1.0, base / (1.0 - overlay + 1e-10)))
        elif blend_mode == "color_burn":
            result = np.where(overlay <= 0.0, 0.0, 1.0 - np.minimum(1.0, (1.0 - base) / (overlay + 1e-10)))
        elif blend_mode == "darken":
            result = np.minimum(base, overlay)
        elif blend_mode == "lighten":
            result = np.maximum(base, overlay)
        elif blend_mode == "difference":
            result = np.abs(base - overlay)
        elif blend_mode == "exclusion":
            result = base + overlay - 2 * base * overlay
        else:
            # For HSL blend modes (hue, saturation, color, luminosity)
            result = self.blend_hsl_modes(base, overlay, blend_mode)

        # Apply opacity
        result = base + opacity * (result - base)

        return np.clip(result, 0, 1)

    def blend_hsl_modes(self, base, overlay, blend_mode):
        """Apply HSL-based blend modes"""
        base_hsl = self.rgb_to_hsl(base)
        overlay_hsl = self.rgb_to_hsl(overlay)

        if blend_mode == "hue":
            result_hsl = np.copy(base_hsl)
            result_hsl[:, :, 0] = overlay_hsl[:, :, 0]  # Use overlay hue
        elif blend_mode == "saturation":
            result_hsl = np.copy(base_hsl)
            result_hsl[:, :, 1] = overlay_hsl[:, :, 1]  # Use overlay saturation
        elif blend_mode == "color":
            result_hsl = np.copy(base_hsl)
            result_hsl[:, :, 0] = overlay_hsl[:, :, 0]  # Use overlay hue
            result_hsl[:, :, 1] = overlay_hsl[:, :, 1]  # Use overlay saturation
        elif blend_mode == "luminosity":
            result_hsl = np.copy(base_hsl)
            result_hsl[:, :, 2] = overlay_hsl[:, :, 2]  # Use overlay luminosity
        else:
            result_hsl = overlay_hsl

        return self.hsl_to_rgb(result_hsl)