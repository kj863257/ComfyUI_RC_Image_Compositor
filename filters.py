import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import colorsys


class RC_GaussianBlur:
    """Gaussian Blur Filter"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_blur"
    CATEGORY = "RC/Filters"
    DESCRIPTION = "Apply Gaussian blur with professional-grade quality control."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Blur radius in pixels"
                }),
                "method": (["PIL", "OpenCV"], {
                    "default": "OpenCV",
                    "tooltip": (
                        "Blur algorithm:\n"
                        "- PIL: Fast, general purpose\n"
                        "- OpenCV: Precise, professional grade"
                    )
                }),
                "preserve_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve alpha channel without blurring"
                }),
            }
        }

    def apply_blur(self, image, radius, method, preserve_alpha):
        if radius <= 0:
            return (image,)

        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img.shape[:2]
        has_alpha = img.shape[2] == 4

        if method == "PIL":
            # Use PIL ImageFilter
            if has_alpha:
                rgb_img = Image.fromarray(img[:, :, :3], 'RGB')
                alpha_img = Image.fromarray(img[:, :, 3], 'L')

                # Apply blur to RGB channels
                rgb_blurred = rgb_img.filter(ImageFilter.GaussianBlur(radius=radius))

                if preserve_alpha:
                    # Keep original alpha
                    result = np.dstack([np.array(rgb_blurred), np.array(alpha_img)])
                else:
                    # Blur alpha too
                    alpha_blurred = alpha_img.filter(ImageFilter.GaussianBlur(radius=radius))
                    result = np.dstack([np.array(rgb_blurred), np.array(alpha_blurred)])
            else:
                img_pil = Image.fromarray(img, 'RGB')
                result = np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=radius)))

        else:  # OpenCV
            if has_alpha:
                rgb = img[:, :, :3].copy()
                alpha = img[:, :, 3].copy()

                # Calculate kernel size (must be odd)
                kernel_size = int(radius * 6)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                if kernel_size < 3:
                    kernel_size = 3

                # Apply Gaussian blur to RGB
                rgb_blurred = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), radius)

                if preserve_alpha:
                    # Keep original alpha
                    result = np.dstack([rgb_blurred, alpha])
                else:
                    # Blur alpha too
                    alpha_blurred = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), radius)
                    result = np.dstack([rgb_blurred, alpha_blurred])
            else:
                kernel_size = int(radius * 6)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                if kernel_size < 3:
                    kernel_size = 3
                result = cv2.GaussianBlur(img, (kernel_size, kernel_size), radius)

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_Sharpen:
    """Sharpen Filter"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sharpen"
    CATEGORY = "RC/Filters"
    DESCRIPTION = "Apply sharpening effect with customizable intensity and method."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Sharpening strength (1.0 = normal, >1.0 = sharper)"
                }),
                "method": (["unsharp_mask", "high_pass", "edge_enhance"], {
                    "default": "unsharp_mask",
                    "tooltip": (
                        "Sharpening method:\n"
                        "- unsharp_mask: Classic unsharp masking\n"
                        "- high_pass: High-pass filter sharpening\n"
                        "- edge_enhance: Edge enhancement sharpening"
                    )
                }),
                "radius": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Sharpening radius (for unsharp mask)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sharpening threshold (avoid noise amplification)"
                }),
            }
        }

    def apply_sharpen(self, image, strength, method, radius, threshold):
        if strength <= 0:
            return (image,)

        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
            work_img = rgb
        else:
            work_img = img

        if method == "unsharp_mask":
            # Classic unsharp mask
            work_float = work_img.astype(np.float32)

            # Create Gaussian blur
            kernel_size = int(radius * 6)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 3:
                kernel_size = 3

            blurred = cv2.GaussianBlur(work_float, (kernel_size, kernel_size), radius)

            # Calculate mask
            mask = work_float - blurred

            # Apply threshold
            if threshold > 0:
                mask_abs = np.abs(mask)
                threshold_val = threshold * 255.0
                mask = np.where(mask_abs > threshold_val, mask, 0)

            # Apply sharpening
            sharpened = work_float + strength * mask
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        elif method == "high_pass":
            # High-pass filter sharpening
            work_float = work_img.astype(np.float32)

            # Create low-pass (Gaussian blur)
            kernel_size = int(radius * 4)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 3:
                kernel_size = 3

            low_pass = cv2.GaussianBlur(work_float, (kernel_size, kernel_size), radius)

            # High-pass = original - low-pass
            high_pass = work_float - low_pass

            # Apply sharpening
            sharpened = work_float + strength * high_pass
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        else:  # edge_enhance
            # Use PIL's edge enhance
            if len(work_img.shape) == 3:
                pil_img = Image.fromarray(work_img, 'RGB')
            else:
                pil_img = Image.fromarray(work_img, 'L')

            enhancer = ImageEnhance.Sharpness(pil_img)
            enhanced = enhancer.enhance(1.0 + strength)
            sharpened = np.array(enhanced)

        # Reassemble with alpha if needed
        if has_alpha:
            result = np.dstack([sharpened, alpha])
        else:
            result = sharpened

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_HueSaturation:
    """Hue/Saturation Adjustment"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_hue_saturation"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = "Photoshop-style Hue/Saturation adjustment with targeted color editing."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue_shift": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Hue shift in degrees (-180 to +180)"
                }),
                "saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Saturation adjustment (-100 = grayscale, +100 = double)"
                }),
                "lightness": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Lightness adjustment (-100 = black, +100 = white)"
                }),
                "colorize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Colorize mode (convert to monochromatic)"
                }),
                "colorize_hue": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Colorize hue (only in colorize mode)"
                }),
                "colorize_saturation": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Colorize saturation (only in colorize mode)"
                }),
                "target_color": (["master", "reds", "yellows", "greens", "cyans", "blues", "magentas"], {
                    "default": "master",
                    "tooltip": (
                        "Target color range:\n"
                        "- master: All colors\n"
                        "- reds: Red range\n"
                        "- yellows: Yellow range\n"
                        "- greens: Green range\n"
                        "- cyans: Cyan range\n"
                        "- blues: Blue range\n"
                        "- magentas: Magenta range"
                    )
                }),
            }
        }

    def get_color_mask(self, hue, target_color):
        """Create mask for specific color range"""
        if target_color == "master":
            return np.ones_like(hue)

        # Define color ranges in hue wheel (0-360)
        color_ranges = {
            "reds": [(345, 360), (0, 15)],      # Red: 345-360, 0-15
            "yellows": [(45, 75)],              # Yellow: 45-75
            "greens": [(75, 165)],              # Green: 75-165
            "cyans": [(165, 195)],              # Cyan: 165-195
            "blues": [(195, 285)],              # Blue: 195-285
            "magentas": [(285, 345)]            # Magenta: 285-345
        }

        mask = np.zeros_like(hue)
        ranges = color_ranges.get(target_color, [])

        for range_tuple in ranges:
            if len(range_tuple) == 2:
                start, end = range_tuple
                if start > end:  # Handle wrap-around (like reds)
                    mask |= (hue >= start) | (hue <= end)
                else:
                    mask |= (hue >= start) & (hue <= end)

        # Smooth mask edges
        feather = 30  # degrees
        for range_tuple in ranges:
            if len(range_tuple) == 2:
                start, end = range_tuple
                # Add feathering
                if start > end:  # wrap-around
                    fade_in = (hue <= (end + feather)) & (hue > end)
                    fade_out = (hue >= (start - feather)) & (hue < start)
                    mask = np.where(fade_in, (end + feather - hue) / feather, mask)
                    mask = np.where(fade_out, (hue - (start - feather)) / feather, mask)
                else:
                    fade_in = (hue >= (start - feather)) & (hue < start)
                    fade_out = (hue <= (end + feather)) & (hue > end)
                    mask = np.where(fade_in, (hue - (start - feather)) / feather, mask)
                    mask = np.where(fade_out, (end + feather - hue) / feather, mask)

        return np.clip(mask, 0, 1)

    def adjust_hue_saturation(self, image, hue_shift, saturation, lightness, colorize,
                            colorize_hue, colorize_saturation, target_color):

        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            rgb = img

        # Convert RGB to HSV for processing
        rgb_float = rgb.astype(np.float32) / 255.0
        hsv = np.zeros_like(rgb_float)

        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                h, s, v = colorsys.rgb_to_hsv(rgb_float[i, j, 0], rgb_float[i, j, 1], rgb_float[i, j, 2])
                hsv[i, j] = [h, s, v]

        # Convert hue to degrees for easier processing
        hue_degrees = hsv[:, :, 0] * 360.0

        # Create mask for target color
        if target_color != "master":
            color_mask = self.get_color_mask(hue_degrees, target_color)
        else:
            color_mask = np.ones_like(hue_degrees)

        if colorize:
            # Colorize mode: convert to grayscale then apply single color
            # Convert to lightness
            lightness_values = np.dot(rgb_float, [0.299, 0.587, 0.114])

            # Apply new hue and saturation
            new_h = colorize_hue / 360.0
            new_s = colorize_saturation / 100.0

            result_hsv = np.stack([
                np.full_like(lightness_values, new_h),
                np.full_like(lightness_values, new_s),
                lightness_values
            ], axis=2)
        else:
            # Normal HSL adjustment
            result_hsv = hsv.copy()

            # Apply hue shift
            if hue_shift != 0:
                hue_shift_norm = hue_shift / 360.0
                new_hue = (hsv[:, :, 0] + hue_shift_norm) % 1.0
                result_hsv[:, :, 0] = hsv[:, :, 0] * (1 - color_mask) + new_hue * color_mask

            # Apply saturation adjustment
            if saturation != 0:
                sat_factor = (saturation + 100.0) / 100.0
                new_sat = np.clip(hsv[:, :, 1] * sat_factor, 0, 1)
                result_hsv[:, :, 1] = hsv[:, :, 1] * (1 - color_mask) + new_sat * color_mask

            # Apply lightness adjustment
            if lightness != 0:
                light_factor = lightness / 100.0
                if light_factor > 0:
                    # Brighten
                    new_val = hsv[:, :, 2] + (1.0 - hsv[:, :, 2]) * light_factor
                else:
                    # Darken
                    new_val = hsv[:, :, 2] * (1.0 + light_factor)
                new_val = np.clip(new_val, 0, 1)
                result_hsv[:, :, 2] = hsv[:, :, 2] * (1 - color_mask) + new_val * color_mask

        # Convert back to RGB
        result_rgb = np.zeros_like(rgb_float)
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                h, s, v = result_hsv[i, j]
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                result_rgb[i, j] = [r, g, b]

        result_rgb = np.clip(result_rgb * 255, 0, 255).astype(np.uint8)

        # Reassemble with alpha if needed
        if has_alpha:
            result = np.dstack([result_rgb, alpha])
        else:
            result = result_rgb

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)