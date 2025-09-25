import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple


class RC_CanvasPadding:
    """画布填充节点 | Canvas Padding Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_padding"
    CATEGORY = "RC/Utilities"
    DESCRIPTION = "调整画布大小，添加填充边距 | Adjust canvas size by adding padding margins."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {
                    "default": 50, "min": 0, "max": 2048, "step": 1,
                    "tooltip": "顶部填充像素数 | Top padding in pixels"
                }),
                "bottom": ("INT", {
                    "default": 50, "min": 0, "max": 2048, "step": 1,
                    "tooltip": "底部填充像素数 | Bottom padding in pixels"
                }),
                "left": ("INT", {
                    "default": 50, "min": 0, "max": 2048, "step": 1,
                    "tooltip": "左侧填充像素数 | Left padding in pixels"
                }),
                "right": ("INT", {
                    "default": 50, "min": 0, "max": 2048, "step": 1,
                    "tooltip": "右侧填充像素数 | Right padding in pixels"
                }),
                "fill_mode": (["color", "edge", "mirror", "transparent"], {
                    "default": "color",
                    "tooltip": (
                        "填充模式：\n"
                        "- color：使用纯色填充\n"
                        "- edge：延伸边缘像素\n"
                        "- mirror：镜像边缘内容\n"
                        "- transparent：透明填充（仅RGBA）\n\n"
                        "Fill mode:\n"
                        "- color: Fill with solid color\n"
                        "- edge: Extend edge pixels\n"
                        "- mirror: Mirror edge content\n"
                        "- transparent: Transparent fill (RGBA only)"
                    )
                }),
                "fill_color_r": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "填充颜色红色分量 | Fill color red component"
                }),
                "fill_color_g": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "填充颜色绿色分量 | Fill color green component"
                }),
                "fill_color_b": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "填充颜色蓝色分量 | Fill color blue component"
                }),
            }
        }

    def apply_padding(self, image, top, bottom, left, right, fill_mode,
                     fill_color_r, fill_color_g, fill_color_b):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img.shape[:2]
        has_alpha = img.shape[2] == 4

        # Calculate new dimensions
        new_h = h + top + bottom
        new_w = w + left + right

        if fill_mode == "color":
            # Fill with solid color
            if has_alpha:
                fill_color = [fill_color_r * 255, fill_color_g * 255, fill_color_b * 255, 255]
                padded = np.full((new_h, new_w, 4), fill_color, dtype=np.uint8)
            else:
                fill_color = [fill_color_r * 255, fill_color_g * 255, fill_color_b * 255]
                padded = np.full((new_h, new_w, 3), fill_color, dtype=np.uint8)

            # Place original image
            padded[top:top+h, left:left+w] = img

        elif fill_mode == "transparent":
            # Transparent fill (only for RGBA)
            if has_alpha:
                padded = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                padded[top:top+h, left:left+w] = img
            else:
                # Convert to RGBA and add transparent padding
                padded = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                padded[top:top+h, left:left+w, :3] = img
                padded[top:top+h, left:left+w, 3] = 255  # Original area opaque

        else:
            # Use PIL for edge and mirror modes
            if has_alpha:
                pil_img = Image.fromarray(img, 'RGBA')
            else:
                pil_img = Image.fromarray(img, 'RGB')

            if fill_mode == "edge":
                # Extend edge pixels
                padded_pil = ImageOps.expand(pil_img, (left, top, right, bottom),
                                           fill=None)  # PIL will extend edges
            else:  # mirror
                # Create mirrored padding manually
                if has_alpha:
                    padded = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                else:
                    padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)

                # Place original image
                padded[top:top+h, left:left+w] = img

                # Mirror padding
                if top > 0:
                    mirror_h = min(top, h)
                    padded[top-mirror_h:top, left:left+w] = np.flip(img[:mirror_h], axis=0)

                if bottom > 0:
                    mirror_h = min(bottom, h)
                    padded[top+h:top+h+mirror_h, left:left+w] = np.flip(img[-mirror_h:], axis=0)

                if left > 0:
                    mirror_w = min(left, w)
                    padded[top:top+h, left-mirror_w:left] = np.flip(padded[top:top+h, left:left+mirror_w], axis=1)

                if right > 0:
                    mirror_w = min(right, w)
                    padded[top:top+h, left+w:left+w+mirror_w] = np.flip(padded[top:top+h, left+w-mirror_w:left+w], axis=1)

                # Convert to result tensor
                result_tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).unsqueeze(0)
                return (result_tensor,)

            if fill_mode == "edge":
                padded = np.array(padded_pil)

        # Convert back to tensor
        result_tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_ImageScale:
    """图像缩放节点 | Image Scale Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_image"
    CATEGORY = "RC/Utilities"
    DESCRIPTION = "多种方式缩放图像尺寸 | Scale image with multiple resizing methods."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_method": (["percentage", "dimensions", "fit_width", "fit_height", "fit_longest", "fit_shortest"], {
                    "default": "percentage",
                    "tooltip": (
                        "缩放方法：\n"
                        "- percentage：按百分比缩放\n"
                        "- dimensions：指定确切尺寸\n"
                        "- fit_width：适配宽度（保持比例）\n"
                        "- fit_height：适配高度（保持比例）\n"
                        "- fit_longest：适配长边（保持比例）\n"
                        "- fit_shortest：适配短边（保持比例）\n\n"
                        "Scale method:\n"
                        "- percentage: Scale by percentage\n"
                        "- dimensions: Exact dimensions\n"
                        "- fit_width: Fit to width (keep ratio)\n"
                        "- fit_height: Fit to height (keep ratio)\n"
                        "- fit_longest: Fit to longest side (keep ratio)\n"
                        "- fit_shortest: Fit to shortest side (keep ratio)"
                    )
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "缩放因子（用于百分比模式）| Scale factor (for percentage mode)"
                }),
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "目标宽度（用于尺寸/适配模式）| Target width (for dimensions/fit modes)"
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "目标高度（用于尺寸/适配模式）| Target height (for dimensions/fit modes)"
                }),
                "resampling": (["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"], {
                    "default": "LANCZOS",
                    "tooltip": (
                        "重采样算法：\n"
                        "- LANCZOS：最高质量（慢）\n"
                        "- BICUBIC：高质量\n"
                        "- BILINEAR：中等质量\n"
                        "- NEAREST：最快（低质量）\n\n"
                        "Resampling algorithm:\n"
                        "- LANCZOS: Highest quality (slow)\n"
                        "- BICUBIC: High quality\n"
                        "- BILINEAR: Medium quality\n"
                        "- NEAREST: Fastest (low quality)"
                    )
                }),
                "keep_aspect_ratio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "保持宽高比（用于尺寸模式）| Keep aspect ratio (for dimensions mode)"
                }),
            }
        }

    def scale_image(self, image, scale_method, scale_factor, width, height,
                   resampling, keep_aspect_ratio):
        # Convert to numpy and PIL
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img.shape[:2]
        has_alpha = img.shape[2] == 4

        # Convert to PIL
        if has_alpha:
            pil_img = Image.fromarray(img, 'RGBA')
        else:
            pil_img = Image.fromarray(img, 'RGB')

        # Get resampling filter
        resample_map = {
            "LANCZOS": Image.LANCZOS,
            "BICUBIC": Image.BICUBIC,
            "BILINEAR": Image.BILINEAR,
            "NEAREST": Image.NEAREST,
        }
        resample_filter = resample_map[resampling]

        # Calculate target dimensions based on scale method
        if scale_method == "percentage":
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)

        elif scale_method == "dimensions":
            if keep_aspect_ratio:
                # Calculate scale factor to fit within target dimensions
                scale_w = width / w
                scale_h = height / h
                scale = min(scale_w, scale_h)
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                new_w = width
                new_h = height

        elif scale_method == "fit_width":
            scale = width / w
            new_w = width
            new_h = int(h * scale)

        elif scale_method == "fit_height":
            scale = height / h
            new_w = int(w * scale)
            new_h = height

        elif scale_method == "fit_longest":
            if w > h:  # Width is longer
                scale = width / w
                new_w = width
                new_h = int(h * scale)
            else:  # Height is longer
                scale = height / h
                new_w = int(w * scale)
                new_h = height

        else:  # fit_shortest
            if w < h:  # Width is shorter
                scale = width / w
                new_w = width
                new_h = int(h * scale)
            else:  # Height is shorter
                scale = height / h
                new_w = int(w * scale)
                new_h = height

        # Ensure minimum size
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # Resize image
        resized_pil = pil_img.resize((new_w, new_h), resample_filter)
        resized = np.array(resized_pil)

        # Convert back to tensor
        result_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_ImageCrop:
    """图像裁剪节点 | Image Crop Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "RC/Utilities"
    DESCRIPTION = "裁剪图像到指定区域 | Crop image to specified region."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_method": (["manual", "center", "aspect_ratio"], {
                    "default": "manual",
                    "tooltip": (
                        "裁剪方法：\n"
                        "- manual：手动指定坐标\n"
                        "- center：居中裁剪到指定尺寸\n"
                        "- aspect_ratio：按比例裁剪（居中）\n\n"
                        "Crop method:\n"
                        "- manual: Manual coordinates\n"
                        "- center: Center crop to size\n"
                        "- aspect_ratio: Crop by ratio (centered)"
                    )
                }),
                "x": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "裁剪起始X坐标（手动模式）| Crop start X coordinate (manual mode)"
                }),
                "y": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "裁剪起始Y坐标（手动模式）| Crop start Y coordinate (manual mode)"
                }),
                "width": ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1,
                    "tooltip": "裁剪宽度 | Crop width"
                }),
                "height": ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1,
                    "tooltip": "裁剪高度 | Crop height"
                }),
                "aspect_width": ("FLOAT", {
                    "default": 16.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "目标宽高比的宽度部分（比例模式）| Width part of aspect ratio (ratio mode)"
                }),
                "aspect_height": ("FLOAT", {
                    "default": 9.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "目标宽高比的高度部分（比例模式）| Height part of aspect ratio (ratio mode)"
                }),
            }
        }

    def crop_image(self, image, crop_method, x, y, width, height, aspect_width, aspect_height):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_h, img_w = img.shape[:2]

        if crop_method == "manual":
            # Manual crop with coordinates
            x1 = max(0, min(x, img_w))
            y1 = max(0, min(y, img_h))
            x2 = max(x1 + 1, min(x + width, img_w))
            y2 = max(y1 + 1, min(y + height, img_h))

            cropped = img[y1:y2, x1:x2]

        elif crop_method == "center":
            # Center crop to specified dimensions
            crop_w = min(width, img_w)
            crop_h = min(height, img_h)

            center_x = img_w // 2
            center_y = img_h // 2

            x1 = max(0, center_x - crop_w // 2)
            y1 = max(0, center_y - crop_h // 2)
            x2 = min(img_w, x1 + crop_w)
            y2 = min(img_h, y1 + crop_h)

            cropped = img[y1:y2, x1:x2]

        else:  # aspect_ratio
            # Crop by aspect ratio (centered)
            target_aspect = aspect_width / aspect_height
            img_aspect = img_w / img_h

            if img_aspect > target_aspect:
                # Image is wider, crop width
                crop_h = img_h
                crop_w = int(crop_h * target_aspect)
            else:
                # Image is taller, crop height
                crop_w = img_w
                crop_h = int(crop_w / target_aspect)

            center_x = img_w // 2
            center_y = img_h // 2

            x1 = max(0, center_x - crop_w // 2)
            y1 = max(0, center_y - crop_h // 2)
            x2 = min(img_w, x1 + crop_w)
            y2 = min(img_h, y1 + crop_h)

            cropped = img[y1:y2, x1:x2]

        # Convert back to tensor
        result_tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


class RC_CanvasResize:
    """画布调整节点 | Canvas Resize Node"""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_canvas"
    CATEGORY = "RC/Utilities"
    DESCRIPTION = "调整画布尺寸和图像位置 | Resize canvas and reposition image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "new_width": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "新画布宽度 | New canvas width"
                }),
                "new_height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "新画布高度 | New canvas height"
                }),
                "anchor": (["center", "top_left", "top_center", "top_right",
                          "middle_left", "middle_right", "bottom_left",
                          "bottom_center", "bottom_right"], {
                    "default": "center",
                    "tooltip": (
                        "图像在新画布中的锚点位置：\n"
                        "- center：居中\n"
                        "- top_left：左上角\n"
                        "- top_center：顶部居中\n"
                        "- top_right：右上角\n"
                        "- middle_left：左侧居中\n"
                        "- middle_right：右侧居中\n"
                        "- bottom_left：左下角\n"
                        "- bottom_center：底部居中\n"
                        "- bottom_right：右下角\n\n"
                        "Image anchor position in new canvas:\n"
                        "- center: Center\n"
                        "- top_left: Top-left corner\n"
                        "- top_center: Top center\n"
                        "- top_right: Top-right corner\n"
                        "- middle_left: Middle left\n"
                        "- middle_right: Middle right\n"
                        "- bottom_left: Bottom-left corner\n"
                        "- bottom_center: Bottom center\n"
                        "- bottom_right: Bottom-right corner"
                    )
                }),
                "x_offset": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 1,
                    "tooltip": "X轴额外偏移（像素）| Additional X offset (pixels)"
                }),
                "y_offset": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 1,
                    "tooltip": "Y轴额外偏移（像素）| Additional Y offset (pixels)"
                }),
                "background_color_r": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "背景颜色红色分量 | Background color red component"
                }),
                "background_color_g": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "背景颜色绿色分量 | Background color green component"
                }),
                "background_color_b": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "背景颜色蓝色分量 | Background color blue component"
                }),
            }
        }

    def resize_canvas(self, image, new_width, new_height, anchor, x_offset, y_offset,
                     background_color_r, background_color_g, background_color_b):
        # Convert to numpy
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_h, img_w = img.shape[:2]
        has_alpha = img.shape[2] == 4

        # Create new canvas
        if has_alpha:
            bg_color = [background_color_r * 255, background_color_g * 255,
                       background_color_b * 255, 255]
            canvas = np.full((new_height, new_width, 4), bg_color, dtype=np.uint8)
        else:
            bg_color = [background_color_r * 255, background_color_g * 255,
                       background_color_b * 255]
            canvas = np.full((new_height, new_width, 3), bg_color, dtype=np.uint8)

        # Calculate position based on anchor
        anchor_positions = {
            "center": (new_width // 2 - img_w // 2, new_height // 2 - img_h // 2),
            "top_left": (0, 0),
            "top_center": (new_width // 2 - img_w // 2, 0),
            "top_right": (new_width - img_w, 0),
            "middle_left": (0, new_height // 2 - img_h // 2),
            "middle_right": (new_width - img_w, new_height // 2 - img_h // 2),
            "bottom_left": (0, new_height - img_h),
            "bottom_center": (new_width // 2 - img_w // 2, new_height - img_h),
            "bottom_right": (new_width - img_w, new_height - img_h),
        }

        pos_x, pos_y = anchor_positions[anchor]
        pos_x += x_offset
        pos_y += y_offset

        # Place image on canvas
        # Calculate valid region
        start_x = max(0, pos_x)
        start_y = max(0, pos_y)
        end_x = min(new_width, pos_x + img_w)
        end_y = min(new_height, pos_y + img_h)

        if start_x < end_x and start_y < end_y:
            # Calculate source region
            src_start_x = max(0, -pos_x)
            src_start_y = max(0, -pos_y)
            src_end_x = src_start_x + (end_x - start_x)
            src_end_y = src_start_y + (end_y - start_y)

            # Copy image region to canvas
            canvas[start_y:end_y, start_x:end_x] = img[src_start_y:src_end_y, src_start_x:src_end_x]

        # Convert back to tensor
        result_tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)