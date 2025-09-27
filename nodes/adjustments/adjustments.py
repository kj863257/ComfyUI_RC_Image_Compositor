import json
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image, ImageEnhance
import cv2


class RC_OpacityAdjust:
    """Opacity Adjustment Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_opacity"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Adjust overall image opacity, supports RGBA and RGB images."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Opacity value (0.0=fully transparent, 1.0=fully opaque)"
                }),
                "ensure_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Ensure alpha channel in output:\n"
                        "- True: Force RGBA output format\n"
                        "- False: Keep original format (RGB stays RGB)"
                    )
                }),
            }
        }

    def adjust_opacity(self, image, opacity, ensure_alpha):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img.shape[:2]
        has_alpha = img.shape[2] == 4

        if has_alpha:
            # Image already has alpha channel
            rgb = img[:, :, :3]
            alpha = img[:, :, 3].astype(np.float32)

            # Apply opacity to existing alpha
            new_alpha = (alpha * opacity).astype(np.uint8)
            result = np.dstack([rgb, new_alpha])

        else:
            # RGB image
            if ensure_alpha or opacity < 1.0:
                # Add alpha channel
                alpha = np.full((h, w), int(opacity * 255), dtype=np.uint8)
                result = np.dstack([img, alpha])
            else:
                # Keep as RGB
                result = img

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_LevelsAdjust:
    """Levels Adjustment Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_levels"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Photoshop-style levels adjustment with input/output range control."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_black": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.9, "step": 0.01,
                    "tooltip": "Input black point (0.0-0.9, raising darkens shadows)"
                }),
                "input_white": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01,
                    "tooltip": "Input white point (0.1-1.0, lowering brightens highlights)"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Gamma value (<1.0 brightens midtones, >1.0 darkens midtones)"
                }),
                "output_black": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.9, "step": 0.01,
                    "tooltip": "Output black point (lifts darkest areas)"
                }),
                "output_white": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01,
                    "tooltip": "Output white point (suppresses brightest areas)"
                }),
                "channel": (["RGB", "Red", "Green", "Blue"], {
                    "default": "RGB",
                    "tooltip": (
                        "Adjustment channel:\n"
                        "- RGB: Adjust all channels together\n"
                        "- Red/Green/Blue: Adjust individual color channels"
                    )
                }),
            }
        }


    def adjust_levels(self, image, input_black, input_white, gamma, output_black, output_white, channel):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3].astype(np.float32) / 255.0
            alpha = img[:, :, 3]
        else:
            rgb = img.astype(np.float32) / 255.0

        # Apply levels adjustment
        def apply_levels_to_channel(channel_data):
            # Input levels adjustment
            # Clamp input range
            adjusted = np.clip((channel_data - input_black) / (input_white - input_black), 0.0, 1.0)

            # Gamma correction
            adjusted = np.power(adjusted, 1.0 / gamma)

            # Output levels adjustment
            adjusted = adjusted * (output_white - output_black) + output_black

            return np.clip(adjusted, 0.0, 1.0)

        if channel == "RGB":
            # Apply to all channels
            rgb_adjusted = apply_levels_to_channel(rgb)
        else:
            # Apply to specific channel
            rgb_adjusted = rgb.copy()
            if channel == "Red":
                rgb_adjusted[:, :, 0] = apply_levels_to_channel(rgb[:, :, 0])
            elif channel == "Green":
                rgb_adjusted[:, :, 1] = apply_levels_to_channel(rgb[:, :, 1])
            elif channel == "Blue":
                rgb_adjusted[:, :, 2] = apply_levels_to_channel(rgb[:, :, 2])

        # Reassemble image
        rgb_adjusted = (rgb_adjusted * 255).astype(np.uint8)
        if has_alpha:
            result = np.dstack([rgb_adjusted, alpha])
        else:
            result = rgb_adjusted

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_BrightnessContrast:
    """Brightness/Contrast Adjustment Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_brightness_contrast"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Adjust image brightness and contrast."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Brightness adjustment (-100 to 100, negative=darker, positive=brighter)"
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Contrast adjustment (-100 to 100, negative=less contrast, positive=more contrast)"
                }),
                "method": (["PIL", "OpenCV"], {
                    "default": "OpenCV",
                    "tooltip": (
                        "Adjustment algorithm:\n"
                        "- PIL: Use PIL library algorithm\n"
                        "- OpenCV: Use OpenCV algorithm, more precise"
                    )
                }),
            }
        }

    def adjust_brightness_contrast(self, image, brightness, contrast, method):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            rgb = img

        if method == "PIL":
            # Use PIL enhancers
            pil_img = Image.fromarray(rgb, 'RGB')

            # Brightness adjustment
            if brightness != 0:
                brightness_factor = 1.0 + (brightness / 100.0)
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(brightness_factor)

            # Contrast adjustment
            if contrast != 0:
                contrast_factor = 1.0 + (contrast / 100.0)
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(contrast_factor)

            rgb_adjusted = np.array(pil_img)

        else:  # OpenCV
            # Convert to float for precise calculation
            rgb_float = rgb.astype(np.float32)

            # Apply brightness (additive)
            if brightness != 0:
                brightness_value = (brightness / 100.0) * 255.0
                rgb_float = rgb_float + brightness_value

            # Apply contrast (multiplicative)
            if contrast != 0:
                contrast_factor = 1.0 + (contrast / 100.0)
                # Apply contrast around 128 (middle gray)
                rgb_float = (rgb_float - 128) * contrast_factor + 128

            # Clamp to valid range
            rgb_adjusted = np.clip(rgb_float, 0, 255).astype(np.uint8)

        # Reassemble image
        if has_alpha:
            result = np.dstack([rgb_adjusted, alpha])
        else:
            result = rgb_adjusted

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_ColorBalance:
    """Color Balance Adjustment Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_color_balance"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Photoshop-style color balance adjustment."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cyan_red": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Cyan-Red balance (negative=cyan, positive=red)"
                }),
                "magenta_green": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Magenta-Green balance (negative=magenta, positive=green)"
                }),
                "yellow_blue": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Yellow-Blue balance (negative=yellow, positive=blue)"
                }),
                "tone_range": (["midtones", "shadows", "highlights"], {
                    "default": "midtones",
                    "tooltip": (
                        "Tone range:\n"
                        "- midtones: Midtones (main adjustment area)\n"
                        "- shadows: Shadows (dark areas)\n"
                        "- highlights: Highlights (bright areas)"
                    )
                }),
                "preserve_luminosity": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Preserve luminosity:\n"
                        "- True: Keep image brightness unchanged\n"
                        "- False: Allow brightness changes"
                    )
                }),
            }
        }

    def adjust_color_balance(self, image, cyan_red, magenta_green, yellow_blue, tone_range, preserve_luminosity):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3].astype(np.float32) / 255.0
            alpha = img[:, :, 3]
        else:
            rgb = img.astype(np.float32) / 255.0

        # Store original luminosity
        if preserve_luminosity:
            original_luminosity = np.dot(rgb, [0.299, 0.587, 0.114])

        # Create tone mask based on range
        luminosity = np.dot(rgb, [0.299, 0.587, 0.114])

        if tone_range == "shadows":
            # More effect on darker areas
            mask = 1.0 - np.power(luminosity, 0.5)
        elif tone_range == "highlights":
            # More effect on brighter areas
            mask = np.power(luminosity, 0.5)
        else:  # midtones
            # Bell curve centered on 0.5
            mask = 1.0 - np.abs(luminosity - 0.5) * 2.0
            mask = np.power(mask, 0.5)

        # Apply color balance adjustments
        rgb_adjusted = rgb.copy()

        # Convert adjustments to factors
        cyan_red_factor = cyan_red / 100.0
        magenta_green_factor = magenta_green / 100.0
        yellow_blue_factor = yellow_blue / 100.0

        # Apply color shifts
        if cyan_red_factor != 0:
            if cyan_red_factor > 0:  # More red
                rgb_adjusted[:, :, 0] += mask * cyan_red_factor * 0.3
            else:  # More cyan
                rgb_adjusted[:, :, 1] += mask * abs(cyan_red_factor) * 0.2
                rgb_adjusted[:, :, 2] += mask * abs(cyan_red_factor) * 0.2

        if magenta_green_factor != 0:
            if magenta_green_factor > 0:  # More green
                rgb_adjusted[:, :, 1] += mask * magenta_green_factor * 0.3
            else:  # More magenta
                rgb_adjusted[:, :, 0] += mask * abs(magenta_green_factor) * 0.2
                rgb_adjusted[:, :, 2] += mask * abs(magenta_green_factor) * 0.2

        if yellow_blue_factor != 0:
            if yellow_blue_factor > 0:  # More blue
                rgb_adjusted[:, :, 2] += mask * yellow_blue_factor * 0.3
            else:  # More yellow
                rgb_adjusted[:, :, 0] += mask * abs(yellow_blue_factor) * 0.2
                rgb_adjusted[:, :, 1] += mask * abs(yellow_blue_factor) * 0.2

        # Clamp values
        rgb_adjusted = np.clip(rgb_adjusted, 0.0, 1.0)

        # Restore luminosity if requested
        if preserve_luminosity:
            new_luminosity = np.dot(rgb_adjusted, [0.299, 0.587, 0.114])
            # Avoid division by zero
            scale = np.where(new_luminosity > 0.001, original_luminosity / new_luminosity, 1.0)
            scale = np.expand_dims(scale, axis=2)
            rgb_adjusted = rgb_adjusted * scale
            rgb_adjusted = np.clip(rgb_adjusted, 0.0, 1.0)

        # Convert back to uint8
        rgb_adjusted = (rgb_adjusted * 255).astype(np.uint8)

        # Reassemble image
        if has_alpha:
            result = np.dstack([rgb_adjusted, alpha])
        else:
            result = rgb_adjusted

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_ChannelMixer:
    """Channel Mixer Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix_channels"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Advanced channel mixer for custom RGB channel blending."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Red output channel
                "red_from_red": ("FLOAT", {
                    "default": 100.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Red channel contribution to red output (%)"
                }),
                "red_from_green": ("FLOAT", {
                    "default": 0.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Green channel contribution to red output (%)"
                }),
                "red_from_blue": ("FLOAT", {
                    "default": 0.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Blue channel contribution to red output (%)"
                }),
                # Green output channel
                "green_from_red": ("FLOAT", {
                    "default": 0.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Red channel contribution to green output (%)"
                }),
                "green_from_green": ("FLOAT", {
                    "default": 100.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Green channel contribution to green output (%)"
                }),
                "green_from_blue": ("FLOAT", {
                    "default": 0.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Blue channel contribution to green output (%)"
                }),
                # Blue output channel
                "blue_from_red": ("FLOAT", {
                    "default": 0.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Red channel contribution to blue output (%)"
                }),
                "blue_from_green": ("FLOAT", {
                    "default": 0.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Green channel contribution to blue output (%)"
                }),
                "blue_from_blue": ("FLOAT", {
                    "default": 100.0, "min": -200.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Blue channel contribution to blue output (%)"
                }),
                "monochrome": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Monochrome mode:\n"
                        "- True: Output grayscale image (uses only red channel settings)\n"
                        "- False: Output color image"
                    )
                }),
            }
        }


    def mix_channels(self, image, red_from_red, red_from_green, red_from_blue,
                    green_from_red, green_from_green, green_from_blue,
                    blue_from_red, blue_from_green, blue_from_blue, monochrome):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3].astype(np.float32) / 255.0
            alpha = img[:, :, 3]
        else:
            rgb = img.astype(np.float32) / 255.0

        # Extract individual channels
        r_channel = rgb[:, :, 0]
        g_channel = rgb[:, :, 1]
        b_channel = rgb[:, :, 2]

        # Calculate new channels based on mixing ratios
        new_r = (r_channel * red_from_red / 100.0 +
                g_channel * red_from_green / 100.0 +
                b_channel * red_from_blue / 100.0)

        if monochrome:
            # In monochrome mode, all channels use the red channel mixing
            new_g = new_r.copy()
            new_b = new_r.copy()
        else:
            new_g = (r_channel * green_from_red / 100.0 +
                    g_channel * green_from_green / 100.0 +
                    b_channel * green_from_blue / 100.0)

            new_b = (r_channel * blue_from_red / 100.0 +
                    g_channel * blue_from_green / 100.0 +
                    b_channel * blue_from_blue / 100.0)

        # Clamp values and recombine
        rgb_mixed = np.stack([
            np.clip(new_r, 0.0, 1.0),
            np.clip(new_g, 0.0, 1.0),
            np.clip(new_b, 0.0, 1.0)
        ], axis=2)

        # Convert back to uint8
        rgb_mixed = (rgb_mixed * 255).astype(np.uint8)

        # Reassemble image
        if has_alpha:
            result = np.dstack([rgb_mixed, alpha])
        else:
            result = rgb_mixed

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_CurvesAdjust:
    """Curves Adjustment Node"""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_curves"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Photoshop-style curves adjustment supporting RGB and individual channels."

    MAX_POINTS = 16

    @staticmethod
    def _identity_points():
        return [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    @classmethod
    def _default_curve_payload(cls):
        identity = cls._identity_points
        return {
            "channels": {
                "RGB": identity(),
                "R": identity(),
                "G": identity(),
                "B": identity(),
                "A": identity(),
            },
            "active_channel": "RGB",
            "editor": "RC_CurvesAdjust",
            "version": 1
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve_data": ("STRING", {
                    "default": json.dumps(cls._default_curve_payload()),
                    "multiline": False,
                    "tooltip": "Curve data in JSON format. Use the curve editor widget to modify points."
                }),
                "mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend factor between original (0) and adjusted image (1)."
                }),
                "apply_alpha": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply combined/alpha curves to the alpha channel when available."
                }),
            }
        }

    @staticmethod
    def _ensure_points(points):
        if not isinstance(points, list):
            return RC_CurvesAdjust._identity_points()

        processed = []
        for pt in points:
            if not isinstance(pt, dict):
                continue
            try:
                x_val = float(pt.get("x", pt.get("position", 0.0)))
                y_val = float(pt.get("y", pt.get("value", 0.0)))
            except (TypeError, ValueError):
                continue
            processed.append((max(0.0, min(1.0, x_val)), max(0.0, min(1.0, y_val))))

        if not processed:
            return RC_CurvesAdjust._identity_points()

        processed.sort(key=lambda v: v[0])

        # Ensure starting and ending points across full range
        if processed[0][0] > 0.0:
            first_y = processed[0][1]
            processed.insert(0, (0.0, first_y))
        if processed[-1][0] < 1.0:
            last_y = processed[-1][1]
            processed.append((1.0, last_y))

        unique = OrderedDict()
        for x_val, y_val in processed:
            unique[x_val] = y_val

        final_points = list(unique.items())

        if not final_points:
            return RC_CurvesAdjust._identity_points()

        first_x, first_y = final_points[0]
        if first_x != 0.0:
            final_points[0] = (0.0, first_y)

        last_x, last_y = final_points[-1]
        if last_x != 1.0:
            final_points[-1] = (1.0, last_y)

        # Remove duplicates again after enforcing endpoints
        deduped = [final_points[0]]
        for x_val, y_val in final_points[1:]:
            prev_x, _ = deduped[-1]
            if abs(x_val - prev_x) < 1e-6:
                deduped[-1] = (x_val, y_val)
            else:
                deduped.append((x_val, y_val))

        if len(deduped) > RC_CurvesAdjust.MAX_POINTS:
            keep = [deduped[0]]
            middle = deduped[1:-1]
            if middle:
                sample_indices = np.linspace(0, len(middle) - 1, RC_CurvesAdjust.MAX_POINTS - 2)
                for idx in sample_indices:
                    mapped = int(np.clip(round(idx), 0, len(middle) - 1))
                    keep.append(middle[mapped])
            keep.append(deduped[-1])
            deduped = keep

        if len(deduped) == 1:
            deduped = [(0.0, deduped[0][1]), (1.0, deduped[0][1])]

        return [{"x": float(x_val), "y": float(y_val)} for x_val, y_val in deduped]

    @staticmethod
    def _solve_tridiagonal_system(a, b, c, d):
        """
        Solve tridiagonal system Ax = d where A has diagonal b, super-diagonal c, sub-diagonal a
        Thomas algorithm implementation
        """
        n = len(d)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([d[0] / b[0]])

        # Forward elimination
        c_prime = np.zeros(n - 1, dtype=np.float64)
        d_prime = np.zeros(n, dtype=np.float64)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n - 1):
            denominator = b[i] - a[i] * c_prime[i - 1]
            if abs(denominator) < 1e-10:
                denominator = 1e-10 if denominator >= 0 else -1e-10
            c_prime[i] = c[i] / denominator
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denominator

        # Last row
        denominator = b[n - 1] - a[n - 1] * c_prime[n - 2]
        if abs(denominator) < 1e-10:
            denominator = 1e-10 if denominator >= 0 else -1e-10
        d_prime[n - 1] = (d[n - 1] - a[n - 1] * d_prime[n - 2]) / denominator

        # Back substitution
        x = np.zeros(n, dtype=np.float64)
        x[n - 1] = d_prime[n - 1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    @staticmethod
    def _calculate_natural_spline_coefficients(xs, ys):
        """
        Calculate Natural Cubic Spline coefficients
        Natural boundary conditions: second derivative = 0 at endpoints
        """
        n = len(xs)
        if n < 2:
            return None

        if n == 2:
            # Linear case
            return None

        # Convert to float64 for better numerical precision
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64)

        # Calculate h values (differences between x points)
        h = np.diff(xs)

        # Handle zero or very small differences
        h = np.maximum(h, 1e-10)

        # Set up tridiagonal system for second derivatives
        # Natural boundary conditions: S''(x0) = S''(xn) = 0
        # This means c[0] = 0 and c[n] = 0

        # Interior equations: h[i-1]*c[i-1] + 2*(h[i-1] + h[i])*c[i] + h[i]*c[i+1] = 6*delta[i]
        # where delta[i] = (y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1]

        if n == 3:
            # Special case for 3 points
            delta = (ys[2] - ys[1]) / h[1] - (ys[1] - ys[0]) / h[0]
            c1 = 6 * delta / (2 * (h[0] + h[1]))
            c = np.array([0.0, c1, 0.0], dtype=np.float64)
        else:
            # General case for n > 3
            # Set up tridiagonal system for interior points (excluding endpoints)
            interior_n = n - 2  # Number of interior points

            if interior_n <= 0:
                c = np.zeros(n, dtype=np.float64)
            else:
                a_diag = np.zeros(interior_n, dtype=np.float64)
                b_diag = np.zeros(interior_n, dtype=np.float64)
                c_diag = np.zeros(interior_n, dtype=np.float64)
                d_vec = np.zeros(interior_n, dtype=np.float64)

                for i in range(interior_n):
                    actual_i = i + 1  # Actual index in original arrays

                    if i > 0:  # Not first interior point
                        a_diag[i] = h[actual_i - 1]

                    b_diag[i] = 2 * (h[actual_i - 1] + h[actual_i])

                    if i < interior_n - 1:  # Not last interior point
                        c_diag[i] = h[actual_i]

                    # Calculate delta for right-hand side
                    delta = (ys[actual_i + 1] - ys[actual_i]) / h[actual_i] - \
                           (ys[actual_i] - ys[actual_i - 1]) / h[actual_i - 1]
                    d_vec[i] = 6 * delta

                # Solve tridiagonal system
                c_interior = RC_CurvesAdjust._solve_tridiagonal_system(a_diag, b_diag, c_diag, d_vec)

                # Assemble full c array with natural boundary conditions
                c = np.zeros(n, dtype=np.float64)
                c[1:-1] = c_interior

        # Calculate a, b, d coefficients and the c linear coefficients
        a = np.zeros(n - 1, dtype=np.float64)
        b = np.zeros(n - 1, dtype=np.float64)
        d = np.zeros(n - 1, dtype=np.float64)
        c_linear = np.zeros(n - 1, dtype=np.float64)

        for i in range(n - 1):
            d[i] = ys[i]
            a[i] = (c[i + 1] - c[i]) / (6 * h[i])
            b[i] = c[i] / 2
            c_linear[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 6

        return {
            'xs': xs.astype(np.float32),
            'ys': ys.astype(np.float32),
            'a': a.astype(np.float32),
            'b': b.astype(np.float32),
            'c': c_linear.astype(np.float32),
            'd': d.astype(np.float32),
            'h': h.astype(np.float32)
        }

    @staticmethod
    def _evaluate_natural_spline(x_eval, spline_data):
        """
        Evaluate Natural Cubic Spline at given x values
        """
        if spline_data is None:
            return x_eval  # Linear fallback

        xs = spline_data['xs']
        a = spline_data['a']
        b = spline_data['b']
        c = spline_data['c']
        d = spline_data['d']

        # Find which segment each x_eval point belongs to
        indices = np.searchsorted(xs[1:], x_eval, side='left')
        indices = np.clip(indices, 0, len(xs) - 2)

        # Calculate relative position within segment
        dx = x_eval - xs[indices]

        # Evaluate cubic polynomial: S(x) = a*(x-xi)^3 + b*(x-xi)^2 + c*(x-xi) + d
        result = (a[indices] * dx**3 +
                 b[indices] * dx**2 +
                 c[indices] * dx +
                 d[indices])

        return result

    @staticmethod
    def calculate_auto_curve_points(image_tensor):
        """
        Calculate auto curve points similar to Photoshop Auto Levels
        Based on histogram analysis of the input image
        """
        try:
            # Convert tensor to numpy array
            if len(image_tensor.shape) == 4:
                img_np = image_tensor[0].cpu().numpy()  # Remove batch dimension
            else:
                img_np = image_tensor.cpu().numpy()

            # Convert to uint8 range for histogram calculation
            img_uint8 = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)

            # Convert to grayscale using standard luminance formula
            if len(img_uint8.shape) == 3 and img_uint8.shape[2] >= 3:
                gray = np.dot(img_uint8[:,:,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                gray = img_uint8.squeeze()

            # Build histogram
            histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
            total_pixels = gray.size

            # Find meaningful black and white points (ignore extreme outliers)
            clip_percent = 0.5  # 0.5% clip on each end, like Photoshop Auto
            clip_pixels = int(total_pixels * clip_percent / 100)

            # Find black point
            cumulative = 0
            black_point = 0
            for i in range(256):
                cumulative += histogram[i]
                if cumulative >= clip_pixels:
                    black_point = i
                    break

            # Find white point
            cumulative = 0
            white_point = 255
            for i in range(255, -1, -1):
                cumulative += histogram[i]
                if cumulative >= clip_pixels:
                    white_point = i
                    break

            # Ensure valid range
            if white_point <= black_point:
                # Image is too flat, apply gentle contrast enhancement
                black_point = max(0, black_point - 10)
                white_point = min(255, white_point + 10)

            # Convert to 0-1 range
            black_point_norm = black_point / 255.0
            white_point_norm = white_point / 255.0

            # Create auto curve points
            auto_points = [
                {"x": 0.0, "y": 0.0},
                {"x": black_point_norm, "y": 0.0},
                {"x": white_point_norm, "y": 1.0},
                {"x": 1.0, "y": 1.0}
            ]

            # Add a subtle midtone adjustment if needed
            mid_point = (black_point_norm + white_point_norm) / 2
            if 0.1 < mid_point < 0.9:
                # Slight gamma adjustment for better midtones
                mid_adjust = mid_point + (0.5 - mid_point) * 0.3
                auto_points.insert(2, {"x": mid_point, "y": mid_adjust})

            return auto_points

        except Exception as e:
            print(f"Auto curve calculation error: {e}")
            # Return identity curve on error
            return [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    @staticmethod
    def _lut_from_points(points):
        pts = RC_CurvesAdjust._ensure_points(points)
        xs = np.clip(np.array([pt["x"] for pt in pts], dtype=np.float32), 0.0, 1.0)
        ys = np.clip(np.array([pt["y"] for pt in pts], dtype=np.float32), 0.0, 1.0)

        if xs.shape[0] == 1:
            value = int(np.clip(ys[0] * 255.0, 0.0, 255.0))
            return np.full(256, value, dtype=np.uint8)

        # If only two points, always generate linear LUT regardless of coordinates
        if xs.shape[0] == 2:
            # Generate linear LUT: from start point to end point
            start_y = ys[0]
            end_y = ys[1]
            linear_values = np.linspace(start_y, end_y, 256, dtype=np.float32)
            return np.clip(np.round(linear_values * 255.0), 0.0, 255.0).astype(np.uint8)

        # Use Natural Cubic Spline for 3 or more points
        spline_data = RC_CurvesAdjust._calculate_natural_spline_coefficients(xs, ys)

        sample_points = np.linspace(0.0, 1.0, 256, dtype=np.float32)

        if spline_data is None:
            # Fallback to linear interpolation
            values = np.interp(sample_points, xs, ys)
        else:
            values = RC_CurvesAdjust._evaluate_natural_spline(sample_points, spline_data)

        values = np.clip(values, 0.0, 1.0)
        return np.clip(np.round(values * 255.0), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _compose_luts(luts):
        result = np.arange(256, dtype=np.uint8)
        for lut in luts:
            if lut is None:
                continue
            lut_clipped = np.clip(lut, 0, 255).astype(np.uint8)
            result = lut_clipped[result]
        return result

    @classmethod
    def _parse_curves(cls, curve_data):
        try:
            payload = json.loads(curve_data)
        except (json.JSONDecodeError, TypeError):
            payload = cls._default_curve_payload()

        channels = payload.get("channels", {})

        parsed = {}
        for key in ("RGB", "R", "G", "B", "A"):
            channel_points = channels.get(key) or payload.get(key)
            if channel_points is None:
                parsed[key] = cls._identity_points()
            else:
                parsed[key] = cls._ensure_points(channel_points)

        return parsed

    def adjust_curves(self, image, curve_data, mix, apply_alpha):
        if mix <= 0.0:
            return (image,)

        curves = self._parse_curves(curve_data)

        combined_lut = self._lut_from_points(curves.get("RGB"))
        red_lut = self._lut_from_points(curves.get("R"))
        green_lut = self._lut_from_points(curves.get("G"))
        blue_lut = self._lut_from_points(curves.get("B"))
        alpha_lut = self._lut_from_points(curves.get("A"))

        final_red_lut = self._compose_luts([combined_lut, red_lut])
        final_green_lut = self._compose_luts([combined_lut, green_lut])
        final_blue_lut = self._compose_luts([combined_lut, blue_lut])
        final_alpha_lut = self._compose_luts([combined_lut, alpha_lut]) if apply_alpha else None

        input_tensor = image[0].cpu().numpy()
        input_tensor = np.clip(input_tensor, 0.0, 1.0)

        has_alpha = input_tensor.shape[2] == 4

        rgb = (input_tensor[:, :, :3] * 255.0).round().astype(np.uint8)
        adjusted_rgb = np.empty_like(rgb)
        adjusted_rgb[:, :, 0] = final_red_lut[rgb[:, :, 0]]
        adjusted_rgb[:, :, 1] = final_green_lut[rgb[:, :, 1]]
        adjusted_rgb[:, :, 2] = final_blue_lut[rgb[:, :, 2]]

        adjusted_rgb = adjusted_rgb.astype(np.float32) / 255.0
        blended_rgb = (1.0 - mix) * input_tensor[:, :, :3] + mix * adjusted_rgb
        blended_rgb = np.clip(blended_rgb, 0.0, 1.0)

        if has_alpha:
            alpha_channel = input_tensor[:, :, 3]
            if apply_alpha and final_alpha_lut is not None:
                alpha_u8 = (alpha_channel * 255.0).round().astype(np.uint8)
                adjusted_alpha = final_alpha_lut[alpha_u8].astype(np.float32) / 255.0
                alpha_channel = (1.0 - mix) * alpha_channel + mix * adjusted_alpha
            alpha_channel = np.clip(alpha_channel, 0.0, 1.0)
            result = np.dstack([blended_rgb, alpha_channel])
        else:
            result = blended_rgb

        result_tensor = torch.from_numpy(result.astype(np.float32)).unsqueeze(0)
        return (result_tensor,)
