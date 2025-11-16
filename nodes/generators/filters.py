import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2


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

        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
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

            # Convert back to tensor for this image
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)


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

        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
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

            # Convert back to tensor for this image
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)


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
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Hue shift in degrees (-360 to +360)"
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

        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            has_alpha = img.shape[2] == 4

            if has_alpha:
                rgb = img[:, :, :3]
                alpha = img[:, :, 3]
            else:
                rgb = img

            # Convert RGB to HSV using OpenCV for vectorized processing
            rgb_float = rgb.astype(np.float32) / 255.0
            hsv = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2HSV).astype(np.float32)

            # OpenCV returns hue in degrees (0-360) for float32, saturation/value in 0-1
            hue = hsv[:, :, 0] / 360.0
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]

            hue_degrees = hue * 360.0

            # Create mask for target color
            if target_color != "master":
                color_mask = self.get_color_mask(hue_degrees, target_color)
            else:
                color_mask = np.ones_like(hue_degrees)

            if colorize:
                lightness_values = np.dot(rgb_float, [0.299, 0.587, 0.114]).astype(np.float32)
                result_h = np.full_like(lightness_values, colorize_hue / 360.0)
                result_s = np.full_like(lightness_values, colorize_saturation / 100.0)
                result_v = lightness_values
            else:
                result_h = hue.copy()
                result_s = sat.copy()
                result_v = val.copy()

                if hue_shift != 0:
                    hue_shift_norm = (hue_shift / 360.0) % 1.0
                    new_hue = (hue + hue_shift_norm) % 1.0
                    result_h = hue * (1 - color_mask) + new_hue * color_mask

                if saturation != 0:
                    sat_factor = (saturation + 100.0) / 100.0
                    new_sat = np.clip(sat * sat_factor, 0.0, 1.0)
                    result_s = sat * (1 - color_mask) + new_sat * color_mask

                if lightness != 0:
                    light_factor = lightness / 100.0
                    brighter = light_factor > 0
                    new_val = np.where(
                        brighter,
                        val + (1.0 - val) * light_factor,
                        val * (1.0 + light_factor)
                    )
                    new_val = np.clip(new_val, 0.0, 1.0)
                    result_v = val * (1 - color_mask) + new_val * color_mask

            hsv_adjusted = np.stack([
                np.clip(result_h * 360.0, 0.0, 360.0),
                np.clip(result_s, 0.0, 1.0),
                np.clip(result_v, 0.0, 1.0)
            ], axis=2).astype(np.float32)

            result_rgb = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
            result_rgb = np.clip(result_rgb * 255.0, 0, 255).astype(np.uint8)

            # Reassemble with alpha if needed
            if has_alpha:
                result = np.dstack([result_rgb, alpha])
            else:
                result = result_rgb

            # Convert back to tensor for this image
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)


class RC_AddNoise:
    """Add Noise Filter Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_noise"
    CATEGORY = "RC/Filters"
    DESCRIPTION = "Add customizable noise to images for texture or artistic effects."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_type": (["gaussian", "gaussian_blur", "uniform", "salt_pepper", "speckle"], {
                    "default": "gaussian",
                    "tooltip": (
                        "Noise type:\n"
                        "- gaussian: Sharp random noise with bell curve distribution\n"
                        "- gaussian_blur: Smooth Gaussian distributed noise with blur\n"
                        "- uniform: Evenly distributed random noise\n"
                        "- salt_pepper: Black and white speckles\n"
                        "- speckle: Multiplicative noise (like film grain)"
                    )
                }),
                "amount": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Noise intensity (0-100, higher values = more noise)"
                }),
                "monochromatic": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Monochromatic noise:\n"
                        "- True: Same noise pattern for all RGB channels (grayscale noise)\n"
                        "- False: Independent noise for each color channel (color noise)"
                    )
                }),
                "preserve_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep original alpha channel unchanged"
                }),
            }
        }

    def add_noise(self, image, noise_type, amount, monochromatic, preserve_alpha):
        # Early exit for zero noise
        if amount == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img = image[i].cpu().numpy()
            has_alpha = img.shape[2] == 4

            if has_alpha:
                rgb = img[:, :, :3]
                alpha = img[:, :, 3]
            else:
                rgb = img

            h, w = rgb.shape[:2]
            noise_factor = amount / 100.0

            # Pre-compute noise shape for optimization
            if monochromatic:
                noise_shape = (h, w, 1)
            else:
                noise_shape = (h, w, 3)

            # Generate noise based on type (optimized)
            if noise_type == "gaussian":
                # Vectorized sharp Gaussian noise generation
                noise = np.random.normal(0, noise_factor * 0.3, noise_shape)
                if monochromatic:
                    noise = np.broadcast_to(noise, (h, w, 3))
                result_rgb = rgb + noise

            elif noise_type == "gaussian_blur":
                # Smooth Gaussian distributed noise with blur
                noise = np.random.normal(0, noise_factor * 0.4, noise_shape)
                if monochromatic:
                    noise = np.broadcast_to(noise, (h, w, 3))

                # Apply Gaussian blur to noise for smoother appearance
                blur_kernel_size = max(3, int(noise_factor * 10))
                if blur_kernel_size % 2 == 0:
                    blur_kernel_size += 1  # Ensure odd kernel size

                # Blur each channel
                noise_blurred = np.zeros_like(noise)
                for c in range(3):
                    noise_blurred[:, :, c] = cv2.GaussianBlur(
                        noise[:, :, c],
                        (blur_kernel_size, blur_kernel_size),
                        noise_factor * 2.0
                    )

                result_rgb = rgb + noise_blurred

            elif noise_type == "uniform":
                # Vectorized uniform noise generation
                noise = np.random.uniform(-noise_factor * 0.5, noise_factor * 0.5, noise_shape)
                if monochromatic:
                    noise = np.broadcast_to(noise, (h, w, 3))
                result_rgb = rgb + noise

            elif noise_type == "salt_pepper":
                # Optimized salt and pepper noise
                result_rgb = rgb.copy()

                # Vectorized salt/pepper generation
                if monochromatic:
                    # Single mask for all channels
                    noise_probability = noise_factor * 0.01
                    noise_mask = np.random.random((h, w)) < noise_probability
                    salt_mask = np.random.random((h, w)) < 0.5

                    # Vectorized assignment using broadcasting
                    salt_pixels = noise_mask & salt_mask
                    pepper_pixels = noise_mask & ~salt_mask

                    result_rgb[salt_pixels] = 1.0    # White (salt)
                    result_rgb[pepper_pixels] = 0.0  # Black (pepper)
                else:
                    # Optimized per-channel processing
                    noise_probability = noise_factor * 0.01
                    for channel in range(3):
                        channel_noise_mask = np.random.random((h, w)) < noise_probability
                        channel_salt_mask = np.random.random((h, w)) < 0.5

                        salt_pixels = channel_noise_mask & channel_salt_mask
                        pepper_pixels = channel_noise_mask & ~channel_salt_mask

                        result_rgb[salt_pixels, channel] = 1.0
                        result_rgb[pepper_pixels, channel] = 0.0

            elif noise_type == "speckle":
                # Optimized speckle noise (multiplicative)
                noise = np.random.normal(1.0, noise_factor * 0.2, noise_shape)
                if monochromatic:
                    noise = np.broadcast_to(noise, (h, w, 3))
                result_rgb = rgb * noise

            # Single clamp operation
            result_rgb = np.clip(result_rgb, 0.0, 1.0)

            # Optimized image reassembly
            if has_alpha:
                if preserve_alpha:
                    result = np.dstack([result_rgb, alpha])
                else:
                    # Optimized alpha noise application
                    if noise_type in ["gaussian", "uniform"]:
                        alpha_noise = np.random.normal(0, noise_factor * 0.1, alpha.shape)
                        result_alpha = np.clip(alpha + alpha_noise, 0.0, 1.0)
                    else:
                        result_alpha = alpha  # Keep original for other noise types
                    result = np.dstack([result_rgb, result_alpha])
            else:
                result = result_rgb

            # Convert back to tensor (already float32) for this image
            result_tensor = torch.from_numpy(result)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)
