import torch
import numpy as np
import cv2


class RC_AutoColor:
    """RC 自动色彩校正 | RC Auto Color Correction

    实现 Photoshop 风格的自动色彩校正算法。
    Implement Photoshop-style auto color correction algorithms.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "algorithm": (["enhance_monochromatic", "enhance_per_channel", "find_dark_light_colors", "enhance_brightness_contrast"], {
                    "default": "enhance_monochromatic",
                    "tooltip": (
                        "Auto correction algorithm:\n"
                        "- enhance_monochromatic: Clip channels identically (Auto Contrast)\n"
                        "- enhance_per_channel: Clip channels independently (Auto Tone)\n"
                        "- find_dark_light_colors: Find dark/light colors (Auto Color)\n"
                        "- enhance_brightness_contrast: Content-aware enhancement"
                    )
                }),
                "neutralize_midtones": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adjust midtones so neutral colors map to target neutral"
                }),
                "shadow_clip": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Amount of shadow data to discard (%)"
                }),
                "highlight_clip": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Amount of highlight data to discard (%)"
                }),
                "target_shadow_r": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                }),
                "target_shadow_g": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                }),
                "target_shadow_b": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                }),
                "target_midtone_r": ("INT", {
                    "default": 128, "min": 0, "max": 255, "step": 1,
                }),
                "target_midtone_g": ("INT", {
                    "default": 128, "min": 0, "max": 255, "step": 1,
                }),
                "target_midtone_b": ("INT", {
                    "default": 128, "min": 0, "max": 255, "step": 1,
                }),
                "target_highlight_r": ("INT", {
                    "default": 255, "min": 0, "max": 255, "step": 1,
                }),
                "target_highlight_g": ("INT", {
                    "default": 255, "min": 0, "max": 255, "step": 1,
                }),
                "target_highlight_b": ("INT", {
                    "default": 255, "min": 0, "max": 255, "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "auto_color_correct"
    CATEGORY = "RC/Adjustments"

    def calculate_histogram_percentiles(self, image, shadow_clip, highlight_clip):
        """Calculate shadow and highlight points from histogram"""
        # Convert to 0-255 range for percentile calculation
        img_255 = (image * 255).astype(np.uint8)

        percentiles = []
        for channel in range(img_255.shape[2]):
            channel_data = img_255[:, :, channel]
            shadow_point = np.percentile(channel_data, shadow_clip)
            highlight_point = np.percentile(channel_data, 100 - highlight_clip)
            percentiles.append((shadow_point, highlight_point))

        return percentiles

    def enhance_monochromatic_contrast(self, image, shadow_clip, highlight_clip):
        """Auto Contrast - clip all channels identically"""
        img_255 = (image * 255).astype(np.uint8)

        # Calculate luminance
        luminance = cv2.cvtColor(img_255, cv2.COLOR_RGB2GRAY)

        # Find shadow and highlight points from luminance
        shadow_point = np.percentile(luminance, shadow_clip)
        highlight_point = np.percentile(luminance, 100 - highlight_clip)

        # Apply same adjustment to all channels
        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel] * 255
            # Stretch to use full range
            if highlight_point > shadow_point:
                adjusted = (channel_data - shadow_point) / (highlight_point - shadow_point) * 255
                adjusted = np.clip(adjusted, 0, 255)
            else:
                adjusted = channel_data
            result[:, :, channel] = adjusted / 255.0

        return result

    def enhance_per_channel_contrast(self, image, shadow_clip, highlight_clip):
        """Auto Tone - clip each channel independently"""
        percentiles = self.calculate_histogram_percentiles(image, shadow_clip, highlight_clip)

        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            shadow_point, highlight_point = percentiles[channel]
            channel_data = image[:, :, channel] * 255

            # Stretch each channel independently
            if highlight_point > shadow_point:
                adjusted = (channel_data - shadow_point) / (highlight_point - shadow_point) * 255
                adjusted = np.clip(adjusted, 0, 255)
            else:
                adjusted = channel_data
            result[:, :, channel] = adjusted / 255.0

        return result

    def find_dark_light_colors(self, image, shadow_clip, highlight_clip, target_shadow, target_highlight):
        """Auto Color - find and map dark/light colors"""
        img_255 = (image * 255).astype(np.uint8)

        # Find darkest and lightest pixels
        luminance = cv2.cvtColor(img_255, cv2.COLOR_RGB2GRAY)

        # Get percentile-based dark and light regions
        dark_threshold = np.percentile(luminance, shadow_clip * 2)  # Use 2x for more selective
        light_threshold = np.percentile(luminance, 100 - highlight_clip * 2)

        # Find average color in dark and light regions
        dark_mask = luminance <= dark_threshold
        light_mask = luminance >= light_threshold

        if np.any(dark_mask):
            dark_color = np.mean(img_255[dark_mask], axis=0)
        else:
            dark_color = np.array([0, 0, 0])

        if np.any(light_mask):
            light_color = np.mean(img_255[light_mask], axis=0)
        else:
            light_color = np.array([255, 255, 255])

        # Create mapping
        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel] * 255

            # Map dark color to target shadow, light color to target highlight
            if light_color[channel] > dark_color[channel]:
                # Linear mapping
                scale = (target_highlight[channel] - target_shadow[channel]) / (light_color[channel] - dark_color[channel])
                adjusted = target_shadow[channel] + (channel_data - dark_color[channel]) * scale
                adjusted = np.clip(adjusted, 0, 255)
            else:
                adjusted = channel_data

            result[:, :, channel] = adjusted / 255.0

        return result

    def enhance_brightness_contrast(self, image, shadow_clip, highlight_clip):
        """Content-aware brightness/contrast enhancement"""
        img_255 = (image * 255).astype(np.uint8)

        # Calculate adaptive histogram equalization per channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        result = np.zeros_like(image)
        for channel in range(min(3, image.shape[2])):  # Only process RGB
            channel_data = img_255[:, :, channel]
            enhanced = clahe.apply(channel_data)

            # Blend with original based on local contrast
            local_std = cv2.GaussianBlur(
                (channel_data.astype(np.float32) - cv2.GaussianBlur(channel_data.astype(np.float32), (21, 21), 0))**2,
                (21, 21), 0
            )**0.5

            # Higher blend ratio in low contrast areas
            blend_ratio = np.clip(1.0 - local_std / 50.0, 0.3, 1.0)

            blended = channel_data * (1 - blend_ratio) + enhanced * blend_ratio
            result[:, :, channel] = np.clip(blended, 0, 255) / 255.0

        # Copy alpha channel if present
        if image.shape[2] == 4:
            result[:, :, 3] = image[:, :, 3]

        return result

    def neutralize_midtones_func(self, image, target_midtone):
        """Neutralize midtones to target neutral color"""
        img_255 = (image * 255).astype(np.uint8)

        # Find midtone regions (around 40-60% luminance)
        luminance = cv2.cvtColor(img_255, cv2.COLOR_RGB2GRAY)
        midtone_mask = (luminance >= 102) & (luminance <= 153)  # 40-60% of 255

        if not np.any(midtone_mask):
            return image

        # Calculate average midtone color
        midtone_color = np.mean(img_255[midtone_mask], axis=0)

        # Calculate adjustment needed
        adjustment = np.array(target_midtone) - midtone_color

        # Apply adjustment with falloff from midtones
        result = image.copy()
        for channel in range(min(3, image.shape[2])):
            channel_data = image[:, :, channel] * 255

            # Create smooth adjustment mask based on distance from midtones
            dist_from_midtone = np.abs(luminance - 128) / 128.0  # 0-1 scale
            adjustment_strength = np.maximum(0, 1.0 - dist_from_midtone * 2)  # Stronger near midtones

            adjusted = channel_data + adjustment[channel] * adjustment_strength
            result[:, :, channel] = np.clip(adjusted, 0, 255) / 255.0

        return result

    def auto_color_correct(self, image, algorithm, neutralize_midtones,
                          shadow_clip, highlight_clip,
                          target_shadow_r, target_shadow_g, target_shadow_b,
                          target_midtone_r, target_midtone_g, target_midtone_b,
                          target_highlight_r, target_highlight_g, target_highlight_b):

        # Convert to numpy
        img = image[0].cpu().numpy()

        target_shadow = [target_shadow_r, target_shadow_g, target_shadow_b]
        target_midtone = [target_midtone_r, target_midtone_g, target_midtone_b]
        target_highlight = [target_highlight_r, target_highlight_g, target_highlight_b]

        # Apply selected algorithm
        if algorithm == "enhance_monochromatic":
            result = self.enhance_monochromatic_contrast(img, shadow_clip, highlight_clip)
        elif algorithm == "enhance_per_channel":
            result = self.enhance_per_channel_contrast(img, shadow_clip, highlight_clip)
        elif algorithm == "find_dark_light_colors":
            result = self.find_dark_light_colors(img, shadow_clip, highlight_clip, target_shadow, target_highlight)
        elif algorithm == "enhance_brightness_contrast":
            result = self.enhance_brightness_contrast(img, shadow_clip, highlight_clip)
        else:
            result = img

        # Apply midtone neutralization if enabled
        if neutralize_midtones:
            result = self.neutralize_midtones_func(result, target_midtone)

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32)).unsqueeze(0)
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "RC_AutoColor": RC_AutoColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RC_AutoColor": "RC Auto Color Correction",
}