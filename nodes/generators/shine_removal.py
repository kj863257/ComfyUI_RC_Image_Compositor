import torch
import numpy as np
import cv2


class RC_ShineRemoval:
    """Advanced Shine/Oil Removal Node - Professional shine reduction using frequency separation"""
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "shine_mask")
    FUNCTION = "remove_shine"
    CATEGORY = "RC/Filters"
    DESCRIPTION = "Professional shine and oil removal tool using high/low frequency separation technique."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shine_detection_threshold": ("FLOAT", {
                    "default": 0.55, "min": 0.1, "max": 0.95, "step": 0.01,
                    "tooltip": "Brightness threshold for detecting shine areas (lower = more sensitive, try 0.5-0.6 for portraits)"
                }),
                "shine_removal_strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Overall strength of shine removal effect (0=no effect, 1=maximum removal)"
                }),
                "high_frequency_reduction": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Reduce high frequency detail in shine areas (removes texture glare)"
                }),
                "low_frequency_smoothing": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Smooth low frequency layer in shine areas (color/tone correction)"
                }),
                "saturation_boost": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Restore color saturation lost in shiny areas (1.0=no boost, try 1.5-2.0)"
                }),
                "smoothing_radius": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Frequency separation blur radius (larger = smoother transition)"
                }),
                "feather_edges": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Feather the shine mask edges for natural blending (in pixels)"
                }),
                "skin_tone_detection": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only process detected skin tone areas (turn OFF if no effect, ON for portraits only)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Invert protection mask (enabled by default: mask bright=protect, mask dark=process)"
                }),
            },
            "optional": {
                "protection_mask": ("MASK", {
                    "tooltip": "Protection mask: bright areas (1.0) will be protected from processing, dark areas (0.0) will be processed"
                }),
            }
        }

    def detect_shine_areas(self, rgb_img, threshold, detect_skin):
        """Detect shiny/oily areas with improved sensitivity"""
        img_float = rgb_img.astype(np.float32) / 255.0

        # Convert to LAB for better luminosity detection
        lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0] / 100.0  # Normalize to 0-1

        # Primary detection: high luminosity with softer threshold
        lum_mask = np.clip((l_channel - threshold) / (0.9 - threshold + 0.001), 0, 1)

        # Apply power function to enhance bright areas
        lum_mask = np.power(lum_mask, 0.7)  # Makes detection more aggressive

        # Secondary detection: local variance (oil creates brightness peaks)
        l_uint8 = (l_channel * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(l_uint8, (15, 15), 0)
        variance = np.abs(l_uint8.astype(np.float32) - blur.astype(np.float32)) / 255.0
        variance_mask = np.clip(variance * 3.0, 0, 1)

        # Combine: bright areas OR bright spots
        shine_mask = np.maximum(lum_mask, variance_mask * 0.6)

        # Optional: Limit to skin tones
        if detect_skin:
            skin_mask = self.detect_skin_tone_mask(rgb_img)
            shine_mask = shine_mask * skin_mask

        # Smooth the mask
        shine_mask_uint8 = (shine_mask * 255).astype(np.uint8)

        # Morphological operations to connect nearby shine areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shine_mask_uint8 = cv2.morphologyEx(shine_mask_uint8, cv2.MORPH_CLOSE, kernel)

        # Smooth further for natural transitions
        shine_mask = cv2.GaussianBlur(shine_mask_uint8, (11, 11), 3) / 255.0

        return shine_mask

    def detect_skin_tone_mask(self, rgb_img):
        """Detect skin tone areas using HSV and YCrCb color spaces"""
        img_float = rgb_img.astype(np.float32) / 255.0

        # Method 1: HSV detection with wider ranges
        hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)

        # Very wide skin tone ranges
        lower_skin = np.array([0, 0.1, 0.15])
        upper_skin = np.array([35, 0.9, 1.0])
        mask_hsv = cv2.inRange(hsv, lower_skin, upper_skin) / 255.0

        # Method 2: YCrCb detection with wider ranges
        ycrcb = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        lower_ycrcb = np.array([0, 128, 73])
        upper_ycrcb = np.array([255, 178, 133])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb) / 255.0

        # Combine both methods
        skin_mask = np.maximum(mask_hsv, mask_ycrcb)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin_mask_uint8 = (skin_mask * 255).astype(np.uint8)
        skin_mask_uint8 = cv2.morphologyEx(skin_mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin_mask_uint8 = cv2.morphologyEx(skin_mask_uint8, cv2.MORPH_OPEN, kernel)

        # Smooth for natural blending
        skin_mask = cv2.GaussianBlur(skin_mask_uint8, (15, 15), 5) / 255.0

        return skin_mask

    def frequency_based_removal(self, rgb_img, shine_mask, smoothing_radius,
                               high_freq_reduction, low_freq_smoothing, sat_boost):
        """High/Low frequency separation method with saturation restoration"""
        img_float = rgb_img.astype(np.float32)

        # Step 1: Frequency separation
        kernel_size = int(smoothing_radius * 2) * 2 + 1
        low_freq = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), smoothing_radius)
        high_freq = img_float - low_freq + 128.0

        # Step 2: Smooth low frequency in shine areas
        low_freq_smoothed = cv2.GaussianBlur(low_freq, (kernel_size * 2 + 1, kernel_size * 2 + 1),
                                             smoothing_radius * 1.5)
        low_freq_processed = low_freq * (1 - shine_mask[:, :, np.newaxis] * low_freq_smoothing) + \
                            low_freq_smoothed * (shine_mask[:, :, np.newaxis] * low_freq_smoothing)

        # Step 3: Reduce high frequency in shine areas
        high_freq_neutral = np.full_like(high_freq, 128.0)
        high_freq_processed = high_freq * (1 - shine_mask[:, :, np.newaxis] * high_freq_reduction) + \
                             high_freq_neutral * (shine_mask[:, :, np.newaxis] * high_freq_reduction)

        # Step 4: Recombine
        result = low_freq_processed + high_freq_processed - 128.0
        result = np.clip(result, 0, 255)

        # Step 5: Restore saturation in processed areas
        if sat_boost > 1.0:
            result_float = result.astype(np.float32) / 255.0
            hsv = cv2.cvtColor(result_float, cv2.COLOR_RGB2HSV)
            saturation_multiplier = 1.0 + (sat_boost - 1.0) * shine_mask
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_multiplier, 0, 1)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            result = (result * 255).astype(np.float32)

        return result

    def remove_shine(self, image, shine_detection_threshold, shine_removal_strength,
                    high_frequency_reduction, low_frequency_smoothing, saturation_boost,
                    smoothing_radius, feather_edges, skin_tone_detection, invert_mask,
                    protection_mask=None):

        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            rgb = img

        # Detect shine areas
        shine_mask = self.detect_shine_areas(rgb, shine_detection_threshold, skin_tone_detection)

        # Apply user-provided protection mask if available
        if protection_mask is not None:
            user_mask = protection_mask[0].cpu().numpy().astype(np.float32)
            if user_mask.shape != (rgb.shape[0], rgb.shape[1]):
                user_mask = cv2.resize(user_mask, (rgb.shape[1], rgb.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

            # Apply invert option
            if invert_mask:
                # mask=1 means PROTECT (don't process), so multiply by (1-mask)
                shine_mask = shine_mask * (1.0 - user_mask)
            else:
                # mask=1 means PROCESS, multiply directly
                shine_mask *= user_mask

        # Feather the shine mask edges
        if feather_edges > 0:
            blur_size = int(feather_edges) * 2 + 1
            shine_mask = cv2.GaussianBlur(shine_mask, (blur_size, blur_size), feather_edges / 3)

        # Apply frequency-based shine removal
        processed = self.frequency_based_removal(
            rgb, shine_mask, smoothing_radius,
            high_frequency_reduction, low_frequency_smoothing, saturation_boost
        )

        processed = np.clip(processed, 0, 255).astype(np.uint8)

        # Blend original and processed based on overall strength
        shine_mask_3d = np.expand_dims(shine_mask * shine_removal_strength, axis=2)
        result_rgb = (rgb.astype(np.float32) * (1 - shine_mask_3d) +
                     processed.astype(np.float32) * shine_mask_3d)
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

        # Reassemble with alpha if needed
        if has_alpha:
            result = np.dstack([result_rgb, alpha])
        else:
            result = result_rgb

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)

        # Return shine mask for debugging
        shine_mask_tensor = torch.from_numpy(shine_mask).unsqueeze(0)

        return (result_tensor, shine_mask_tensor)
