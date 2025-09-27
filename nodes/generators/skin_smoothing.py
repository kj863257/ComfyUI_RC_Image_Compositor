import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2


class RC_HighLowFrequencySkinSmoothing:
    """High/Low Frequency Skin Smoothing Node - Photoshop-style frequency separation technique"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_skin_smoothing"
    CATEGORY = "RC/Filters"
    DESCRIPTION = "Photoshop-style high/low frequency skin smoothing with professional-grade quality control."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_frequency_radius": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Low frequency separation radius - determines the size of skin texture features to smooth"
                }),
                "high_frequency_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "High frequency detail strength (0=maximum smoothing, 1=original detail, >1=enhanced detail)"
                }),
                "skin_tone_protection": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Skin tone protection level (0=affect all areas, 1=only affect detected skin tones)"
                }),
                "preserve_fine_details": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve fine details like eyelashes, eyebrows and hair"
                }),
                "edge_protection": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Edge protection strength (0=no protection, 1=maximum edge preservation)"
                }),
                "method": (["frequency_separation", "surface_blur", "selective_gaussian"], {
                    "default": "frequency_separation",
                    "tooltip": (
                        "Smoothing method:\n"
                        "- frequency_separation: Classic PS frequency separation\n"
                        "- surface_blur: Surface blur for texture preservation\n"
                        "- selective_gaussian: Selective Gaussian blur"
                    )
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask to limit smoothing to specific areas (1.0=smooth, 0.0=preserve)"
                }),
            }
        }

    def detect_skin_tone_mask(self, rgb_img):
        """Detect skin tone areas using HSV color space analysis"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(rgb_img.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)

        # Define skin tone ranges in HSV
        # These ranges cover various skin tones
        lower_skin1 = np.array([0, 0.23, 0.35])      # Lighter skin tones
        upper_skin1 = np.array([25, 0.68, 0.95])

        lower_skin2 = np.array([5, 0.4, 0.2])        # Medium skin tones
        upper_skin2 = np.array([17, 0.98, 0.95])

        lower_skin3 = np.array([8, 0.15, 0.15])      # Darker skin tones
        upper_skin3 = np.array([20, 0.8, 0.85])

        # Create masks for each range
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask3 = cv2.inRange(hsv, lower_skin3, upper_skin3)

        # Combine all masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        skin_mask = cv2.bitwise_or(skin_mask, mask3)

        # Smooth the mask to reduce harsh transitions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # Apply Gaussian blur for soft edges
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 5)

        return skin_mask.astype(np.float32) / 255.0

    def create_edge_mask(self, rgb_img, edge_protection):
        """Create edge protection mask using Canny edge detection"""
        if edge_protection <= 0:
            return np.ones((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.float32)

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to protect surrounding areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Create protection mask (1 = no smoothing, 0 = full smoothing)
        edge_mask = edges.astype(np.float32) / 255.0
        edge_mask = edge_mask * edge_protection

        # Invert so edges are protected (high values = preserve, low values = smooth)
        protection_mask = 1.0 - edge_mask

        return protection_mask

    def frequency_separation_method(self, rgb_img, low_freq_radius, high_freq_strength):
        """Classic Photoshop frequency separation technique"""
        # Convert to float for processing
        img_float = rgb_img.astype(np.float32)

        # Step 1: Create low frequency layer (skin texture/color)
        # Use Gaussian blur to separate low frequencies
        kernel_size = int(low_freq_radius * 2) * 2 + 1  # Ensure odd kernel size
        low_freq = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), low_freq_radius)

        # Step 2: Create high frequency layer (fine details)
        # High freq = Original - Low freq + 128 (in 8-bit equivalent)
        high_freq = img_float - low_freq + 128.0

        # Step 3: Smooth the low frequency layer
        # Apply additional smoothing to the low frequency layer
        smooth_kernel_size = int(low_freq_radius * 1.5) * 2 + 1
        low_freq_smoothed = cv2.GaussianBlur(low_freq, (smooth_kernel_size, smooth_kernel_size), low_freq_radius * 0.8)

        # Step 4: Reconstruct with adjustable high frequency strength
        # Result = Smoothed Low Freq + (High Freq - 128) * strength
        high_freq_adjusted = (high_freq - 128.0) * high_freq_strength
        result = low_freq_smoothed + high_freq_adjusted

        return np.clip(result, 0, 255)

    def surface_blur_method(self, rgb_img, radius, threshold=15):
        """Surface blur method for texture-preserving smoothing"""
        img_float = rgb_img.astype(np.float32)

        # Approximate surface blur using multiple bilateral filters
        result = img_float.copy()

        # Apply multiple passes of bilateral filter for surface blur effect
        for i in range(3):
            current_radius = max(3, int(radius * (0.5 + i * 0.25)))
            sigma_color = threshold * (1 + i * 0.5)
            sigma_space = current_radius

            result = cv2.bilateralFilter(
                result.astype(np.uint8),
                current_radius * 2 + 1,
                sigma_color,
                sigma_space
            ).astype(np.float32)

        return result

    def selective_gaussian_method(self, rgb_img, radius, threshold=20):
        """Selective Gaussian blur - only blur similar colors"""
        img_float = rgb_img.astype(np.float32)

        # Create a selective blur by comparing pixel differences
        h, w, c = img_float.shape
        result = img_float.copy()

        # Apply Gaussian blur
        kernel_size = int(radius * 2) * 2 + 1
        blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), radius)

        # Create selection mask based on color difference
        diff = np.abs(img_float - blurred)
        diff_magnitude = np.sqrt(np.sum(diff ** 2, axis=2, keepdims=True))

        # Smooth transition based on threshold
        selection_mask = np.exp(-diff_magnitude / threshold)

        # Blend original and blurred based on selection
        result = img_float * (1 - selection_mask) + blurred * selection_mask

        return result

    def apply_skin_smoothing(self, image, low_frequency_radius, high_frequency_strength,
                           skin_tone_protection, preserve_fine_details, edge_protection,
                           method, mask=None):

        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            rgb = img

        # Process the RGB channels
        if method == "frequency_separation":
            processed = self.frequency_separation_method(rgb, low_frequency_radius, high_frequency_strength)
        elif method == "surface_blur":
            processed = self.surface_blur_method(rgb, low_frequency_radius)
        elif method == "selective_gaussian":
            processed = self.selective_gaussian_method(rgb, low_frequency_radius)
        else:
            processed = rgb.astype(np.float32)

        processed = np.clip(processed, 0, 255).astype(np.uint8)

        # Create combination mask
        final_mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

        # Apply skin tone protection
        if skin_tone_protection > 0:
            skin_mask = self.detect_skin_tone_mask(rgb)
            final_mask *= (skin_mask * skin_tone_protection + (1 - skin_tone_protection))

        # Apply edge protection
        if edge_protection > 0 and preserve_fine_details:
            edge_mask = self.create_edge_mask(rgb, edge_protection)
            final_mask *= edge_mask

        # Apply user-provided mask if available
        if mask is not None:
            # ComfyUI MASK type is (batch, height, width) in range [0, 1]
            user_mask = mask[0].cpu().numpy().astype(np.float32)
            # Ensure the mask has the same dimensions as the image
            if user_mask.shape != (rgb.shape[0], rgb.shape[1]):
                user_mask = cv2.resize(user_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            final_mask *= user_mask

        # Expand mask to match RGB channels
        final_mask_3d = np.expand_dims(final_mask, axis=2)

        # Blend original and processed based on final mask
        result_rgb = (rgb.astype(np.float32) * (1 - final_mask_3d) +
                     processed.astype(np.float32) * final_mask_3d)
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

        # Reassemble with alpha if needed
        if has_alpha:
            result = np.dstack([result_rgb, alpha])
        else:
            result = result_rgb

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)