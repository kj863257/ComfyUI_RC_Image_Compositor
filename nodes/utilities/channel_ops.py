import torch
import numpy as np
from PIL import Image
import cv2


class RC_ChannelExtractor:
    """RC 通道提取器 | RC Channel Extractor

    提取RGB通道并转换为灰度图或蒙版，常用于抠图工作流。
    Extract RGB channels and convert to grayscale or mask, commonly used for matting workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["red", "green", "blue", "alpha", "luminance"], {
                    "default": "green",
                    "tooltip": "Channel to extract:\n- red: Red channel\n- green: Green channel (often best for green screen)\n- blue: Blue channel\n- alpha: Alpha channel (if available)\n- luminance: Luminance (0.299*R + 0.587*G + 0.114*B)"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the extracted channel"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "extract_channel"
    CATEGORY = "RC/Channel"
    DESCRIPTION = "Extract RGB channel or luminance as grayscale image and mask"

    def extract_channel(self, image, channel, invert):
        batch_size = image.shape[0]
        image_results = []
        mask_results = []

        for i in range(batch_size):
            # Convert to numpy for each image in the batch
            img_np = image[i].cpu().numpy()  # [H, W, C]

            if channel == "red":
                extracted = img_np[:, :, 0]
            elif channel == "green":
                extracted = img_np[:, :, 1]
            elif channel == "blue":
                extracted = img_np[:, :, 2]
            elif channel == "alpha":
                if img_np.shape[2] >= 4:
                    extracted = img_np[:, :, 3]
                else:
                    # No alpha channel, return full opacity
                    extracted = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
            elif channel == "luminance":
                # Standard luminance formula
                extracted = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

            # Invert if requested
            if invert:
                extracted = 1.0 - extracted

            # Clamp values
            extracted = np.clip(extracted, 0.0, 1.0)

            # Create grayscale image (RGB with same value in all channels)
            gray_rgb = np.stack([extracted, extracted, extracted], axis=2)

            # Convert to tensors for this image
            image_tensor = torch.from_numpy(gray_rgb)
            mask_tensor = torch.from_numpy(extracted)

            image_results.append(image_tensor)
            mask_results.append(mask_tensor)

        # Stack all results to create batch tensors
        batch_image = torch.stack(image_results, dim=0)
        batch_mask = torch.stack(mask_results, dim=0)

        return (batch_image, batch_mask)


class RC_MaskApply:
    """RC 蒙版应用器 | RC Mask Applicator

    使用蒙版来控制图像的透明度，实现抠图效果。
    Use mask to control image transparency, achieving matting effects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_mode": (["alpha", "multiply"], {
                    "default": "alpha",
                    "tooltip": "Mask application mode:\n- alpha: Use mask as alpha channel\n- multiply: Multiply image by mask (darken)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Threshold for mask (values below this become transparent)"
                }),
                "feather": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Feather/smooth the mask edges"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"
    CATEGORY = "RC/Channel"
    DESCRIPTION = "Apply mask to image for matting/transparency effects"

    def apply_mask(self, image, mask, mask_mode, threshold, feather):
        batch_size = image.shape[0]
        mask_batch_size = mask.shape[0]
        output_batch_size = max(batch_size, mask_batch_size)
        results = []

        for i in range(output_batch_size):
            # Cycle through images and masks if needed
            img_idx = i % batch_size
            mask_idx = i % mask_batch_size

            # Convert to numpy for each image in the batch
            img_np = image[img_idx].cpu().numpy()  # [H, W, 3]
            mask_np = mask[mask_idx].cpu().numpy().astype(np.float32)  # [H, W]

            # Apply threshold
            if threshold > 0:
                mask_np = np.where(mask_np >= threshold, mask_np, 0.0)
                # Rescale values above threshold to 0-1 range
                if mask_np.max() > threshold:
                    mask_np = np.where(mask_np > threshold,
                                      (mask_np - threshold) / (1.0 - threshold),
                                      0.0)

            # Apply feathering (simple gaussian-like smoothing)
            if feather > 0:
                kernel_size = max(3, int(feather * min(mask_np.shape) * 0.2))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                sigma = max(kernel_size * 0.15, 1e-3)
                mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

            mask_np = np.clip(mask_np, 0.0, 1.0)

            if mask_mode == "alpha":
                # Create RGBA image
                alpha = mask_np[:, :, np.newaxis]  # [H, W, 1]
                result = np.concatenate([img_np, alpha], axis=2)  # [H, W, 4]
            else:  # multiply
                # Multiply RGB by mask
                mask_3d = mask_np[:, :, np.newaxis]  # [H, W, 1]
                result = img_np * mask_3d  # Broadcast multiply

            result = np.clip(result, 0.0, 1.0)
            result_tensor = torch.from_numpy(result)
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)
