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