import torch
import numpy as np
from PIL import Image, ImageOps
import os
import folder_paths
import colorsys
from typing import Optional


def rgb_to_hsl_batch(rgb):
    """Convert RGB batch to HSL batch"""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    l = np.zeros_like(r)

    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            h[i, j], l[i, j], s[i, j] = colorsys.rgb_to_hls(r[i, j], g[i, j], b[i, j])

    return np.stack([h, s, l], axis=2)


def hsl_to_rgb_batch(hsl):
    """Convert HSL batch to RGB batch"""
    h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    for i in range(hsl.shape[0]):
        for j in range(hsl.shape[1]):
            r[i, j], g[i, j], b[i, j] = colorsys.hls_to_rgb(h[i, j], l[i, j], s[i, j])

    return np.stack([r, g, b], axis=2)

class RC_ImageCompositor:
    """RC 图像合成器：Photoshop 风格混合模式、精确定位和灵活缩放 | RC Image Compositor: Photoshop-style blend modes, precise positioning, and flexible scaling."""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "RC/Image"
    DESCRIPTION = "基础图像合成器，支持 Photoshop 兼容混合模式、百分比定位、缩放、旋转和透明度 | Base compositor with Photoshop-compatible blend modes, positioning, scaling, rotation, and opacity."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "overlay": ("IMAGE",),
                "x_percent": ("INT", {
                    "default": 100, "min": 0, "max": 100, "step": 1,
                    "tooltip": "水平百分比位置（0=左，50=居中，100=右）| Horizontal percentage (0=left, 50=center, 100=right)"
                }),
                "y_percent": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "垂直百分比位置（0=上，50=居中，100=下）| Vertical percentage (0=top, 50=center, 100=bottom)"
                }),
                "x_align": (["from_left", "from_right"], {
                    "default": "from_right",
                    "tooltip": (
                        "水平对齐方式：\n"
                        "- from_left：从左侧计算偏移\n"
                        "- from_right：从右侧计算偏移（可实现紧贴右侧效果）\n\n"
                        "Horizontal alignment:\n"
                        "- from_left: Calculate offset from left\n"
                        "- from_right: Calculate offset from right (enables tight right alignment)"
                    )
                }),
                "y_align": (["from_top", "from_bottom"], {
                    "default": "from_top",
                    "tooltip": (
                        "垂直对齐方式：\n"
                        "- from_top：从顶部计算偏移\n"
                        "- from_bottom：从底部计算偏移（可实现紧贴底部效果）\n\n"
                        "Vertical alignment:\n"
                        "- from_top: Calculate offset from top\n"
                        "- from_bottom: Calculate offset from bottom (enables tight bottom alignment)"
                    )
                }),
                "x_offset": ("INT", {
                    "default": 50, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "水平偏移像素（正数，配合对齐方式使用）| Horizontal offset pixels (positive, use with alignment)"
                }),
                "y_offset": ("INT", {
                    "default": 50, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "垂直偏移像素（正数，配合对齐方式使用）| Vertical offset pixels (positive, use with alignment)"
                }),
                "scale_mode": (["relative_to_overlay", "relative_to_background_width", "relative_to_background_height"], {
                    "default": "relative_to_background_width",
                    "tooltip": (
                        "缩放参考模式：\n"
                        "- relative_to_overlay：按贴图原始尺寸缩放\n"
                        "- relative_to_background_width：贴图宽度 = scale × 背景宽度\n"
                        "- relative_to_background_height：贴图高度 = scale × 背景高度\n\n"
                        "Scaling mode:\n"
                        "- relative_to_overlay: scale by overlay's size\n"
                        "- relative_to_background_width: width = scale × bg width\n"
                        "- relative_to_background_height: height = scale × bg height"
                    )
                }),
                "scale": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": "缩放因子，参考基准由 'scale_mode' 决定 | Scale factor, reference depends on 'scale_mode'"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "旋转角度（度）| Rotation angle in degrees"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "整体不透明度（0=完全透明，1=完全不透明）| Opacity (0=transparent, 1=opaque)"
                }),
                \"blend_mode\": ([
                    \"normal\",           # 正常
                    \"darken\",           # 变暗
                    \"multiply\",         # 正片叠底
                    \"color_burn\",       # 颜色加深
                    \"linear_burn\",      # 线性加深
                    \"lighten\",          # 变亮
                    \"screen\",           # 滤色
                    \"color_dodge\",      # 颜色减淡
                    \"linear_dodge\",     # 线性减淡（添加）
                    \"overlay\",          # 叠加
                    \"soft_light\",       # 柔光
                    \"hard_light\",       # 强光
                    \"vivid_light\",      # 亮光
                    \"linear_light\",     # 线性光
                    \"pin_light\",        # 点光
                    \"hard_mix\",         # 实色混合
                    \"difference\",       # 差值
                    \"exclusion\",        # 排除
                    \"subtract\",         # 减去
                    \"divide\",           # 划分
                    \"hue\",              # 色相
                    \"saturation\",       # 饱和度
                    \"color\",            # 颜色
                    \"luminosity\"        # 明度
                ], {
                    \"default\": \"normal\",
                    \"tooltip\": (
                        \"Photoshop 完整混合模式（中英对照）| Complete Photoshop Blend Modes:\\n\"
                        \"normal: 正常 - 直接覆盖 | Normal - Direct overlay\\n\"
                        \"darken: 变暗 - 选择较暗像素 | Darken - Select darker pixels\\n\"
                        \"multiply: 正片叠底 - 颜色相乘变暗 | Multiply - Colors multiply to darken\\n\"
                        \"color_burn: 颜色加深 - 增加对比度变暗 | Color Burn - Darken with increased contrast\\n\"
                        \"linear_burn: 线性加深 - 线性方式变暗 | Linear Burn - Linear darkening\\n\"
                        \"lighten: 变亮 - 选择较亮像素 | Lighten - Select brighter pixels\\n\"
                        \"screen: 滤色 - 反向相乘变亮 | Screen - Inverse multiply to lighten\\n\"
                        \"color_dodge: 颜色减淡 - 减少对比度变亮 | Color Dodge - Lighten with reduced contrast\\n\"
                        \"linear_dodge: 线性减淡 - 直接相加变亮 | Linear Dodge - Direct addition to lighten\\n\"
                        \"overlay: 叠加 - 结合正片叠底和滤色 | Overlay - Combines multiply and screen\\n\"
                        \"soft_light: 柔光 - 柔和的对比增强 | Soft Light - Gentle contrast enhancement\\n\"
                        \"hard_light: 强光 - 强烈的对比增强 | Hard Light - Strong contrast enhancement\\n\"
                        \"vivid_light: 亮光 - 极端对比效果 | Vivid Light - Extreme contrast effect\\n\"
                        \"linear_light: 线性光 - 线性对比调整 | Linear Light - Linear contrast adjustment\\n\"
                        \"pin_light: 点光 - 替换颜色根据亮度 | Pin Light - Replace colors based on brightness\\n\"
                        \"hard_mix: 实色混合 - 产生纯色结果 | Hard Mix - Creates solid color results\\n\"
                        \"difference: 差值 - 颜色差的绝对值 | Difference - Absolute difference of colors\\n\"
                        \"exclusion: 排除 - 柔和的差值效果 | Exclusion - Softer difference effect\\n\"
                        \"subtract: 减去 - 直接颜色相减 | Subtract - Direct color subtraction\\n\"
                        \"divide: 划分 - 颜色除法运算 | Divide - Color division operation\\n\"
                        \"hue: 色相 - 仅改变色相，保持饱和度和明度 | Hue - Change only hue, keep saturation & lightness\\n\"
                        \"saturation: 饱和度 - 仅改变饱和度 | Saturation - Change only saturation\\n\"
                        \"color: 颜色 - 改变色相和饱和度，保持明度 | Color - Change hue & saturation, keep lightness\\n\"
                        \"luminosity: 明度 - 仅改变明度，保持色相和饱和度 | Luminosity - Change only lightness\"
                    )
                }),
                "flip_h": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "水平翻转贴图 | Flip horizontally"
                }),
                "flip_v": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "垂直翻转贴图 | Flip vertically"
                }),
            }
        }

    def composite(self, background, overlay,
                  x_percent, y_percent,
                  x_align, y_align,
                  x_offset, y_offset,
                  scale_mode, scale,
                  rotation, opacity,
                  blend_mode, flip_h, flip_v):
        # Convert to [H, W, C] uint8 numpy
        bg = (background[0].cpu().numpy() * 255).astype(np.uint8)
        fg = (overlay[0].cpu().numpy() * 255).astype(np.uint8)

        bg_h, bg_w = bg.shape[:2]
        fg_h_orig, fg_w_orig = fg.shape[:2]

        # === 1. Compute target size ===
        if scale_mode == "relative_to_overlay":
            new_w = int(fg_w_orig * scale)
            new_h = int(fg_h_orig * scale)
        elif scale_mode == "relative_to_background_width":
            target_width = int(bg_w * scale)
            scale_factor = target_width / fg_w_orig if fg_w_orig > 0 else 1.0
            new_w = target_width
            new_h = int(fg_h_orig * scale_factor)
        elif scale_mode == "relative_to_background_height":
            target_height = int(bg_h * scale)
            scale_factor = target_height / fg_h_orig if fg_h_orig > 0 else 1.0
            new_h = target_height
            new_w = int(fg_w_orig * scale_factor)
        else:
            new_w, new_h = fg_w_orig, fg_h_orig

        if new_w <= 0 or new_h <= 0:
            return (background,)

        # === 2. Resize ===
        if fg.shape[2] == 4:
            rgb = Image.fromarray(fg[:, :, :3], 'RGB')
            a = Image.fromarray(fg[:, :, 3], 'L')
            rgb = rgb.resize((new_w, new_h), Image.LANCZOS)
            a = a.resize((new_w, new_h), Image.LANCZOS)
            fg_resized = np.dstack([np.array(rgb), np.array(a)])
        else:
            fg_resized = np.array(Image.fromarray(fg, 'RGB').resize((new_w, new_h), Image.LANCZOS))

        # === 3. Flip ===
        if flip_h:
            fg_resized = np.fliplr(fg_resized)
        if flip_v:
            fg_resized = np.flipud(fg_resized)

        # === 4. Rotate ===
        if abs(rotation) > 1e-6:
            pil_img = Image.fromarray(fg_resized)
            rotated = pil_img.rotate(rotation, resample=Image.BICUBIC, expand=True)
            fg_resized = np.array(rotated)

        # === 5. Position with improved alignment system ===
        fg_h, fg_w = fg_resized.shape[:2]

        # Calculate base position from percentage
        x_base = int((bg_w - fg_w) * x_percent / 100)
        y_base = int((bg_h - fg_h) * y_percent / 100)

        # Apply offset based on alignment direction
        if x_align == "from_left":
            x = x_base + x_offset
        else:  # from_right
            x = x_base - x_offset

        if y_align == "from_top":
            y = y_base + y_offset
        else:  # from_bottom
            y = y_base - y_offset

        # === 6. Composite ===
        result = bg.copy()
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + fg_w), min(bg_h, y + fg_h)

        if x1 >= x2 or y1 >= y2:
            pass
        else:
            fg_x1 = max(0, -x)
            fg_y1 = max(0, -y)
            fg_roi = fg_resized[fg_y1:fg_y1 + (y2 - y1), fg_x1:fg_x1 + (x2 - x1)]
            bg_roi = result[y1:y2, x1:x2]

            if fg_roi.shape[:2] != bg_roi.shape[:2]:
                pass
            else:
                if fg_roi.shape[2] == 4:
                    rgb = fg_roi[:, :, :3].astype(np.float32) / 255.0
                    alpha = (fg_roi[:, :, 3:4].astype(np.float32) / 255.0) * opacity
                else:
                    rgb = fg_roi.astype(np.float32) / 255.0
                    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), opacity, dtype=np.float32)

                bg_f = bg_roi.astype(np.float32) / 255.0

                # Apply blend mode
                blended = self.apply_blend_mode(bg_f, rgb, blend_mode)

                # Clamp blend results to [0, 1]
                blended = np.clip(blended, 0.0, 1.0)

                out = bg_f * (1 - alpha) + blended * alpha
                out = np.clip(out, 0, 1) * 255
                out = out.astype(np.uint8)

                if result.shape[2] == 4:
                    result[y1:y2, x1:x2, :3] = out
                else:
                    result[y1:y2, x1:x2] = out

        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)

    def apply_blend_mode(self, bg_f, fg_f, blend_mode):
        """Apply blend mode with full Photoshop compatibility"""
        if blend_mode == "darken":
            return np.minimum(bg_f, fg_f)
        elif blend_mode == "multiply":
            return bg_f * fg_f
        elif blend_mode == "color_burn":
            return 1.0 - np.minimum((1.0 - bg_f) / (fg_f + 1e-6), 1.0)
        elif blend_mode == "linear_burn":
            return bg_f + fg_f - 1.0
        elif blend_mode == "lighten":
            return np.maximum(bg_f, fg_f)
        elif blend_mode == "screen":
            return 1.0 - (1.0 - bg_f) * (1.0 - fg_f)
        elif blend_mode == "color_dodge":
            return np.minimum(bg_f / (1.0 - fg_f + 1e-6), 1.0)
        elif blend_mode == "linear_dodge":
            return bg_f + fg_f
        elif blend_mode == "overlay":
            mask = bg_f >= 0.5
            result = np.zeros_like(bg_f)
            result[mask] = 1.0 - 2.0 * (1.0 - bg_f[mask]) * (1.0 - fg_f[mask])
            result[~mask] = 2.0 * bg_f[~mask] * fg_f[~mask]
            return result
        elif blend_mode == "soft_light":
            mask = fg_f <= 0.5
            result = np.zeros_like(bg_f)
            result[mask] = bg_f[mask] - (1.0 - 2.0 * fg_f[mask]) * bg_f[mask] * (1.0 - bg_f[mask])
            result[~mask] = bg_f[~mask] + (2.0 * fg_f[~mask] - 1.0) * (np.sqrt(bg_f[~mask]) - bg_f[~mask])
            return result
        elif blend_mode == "hard_light":
            mask = fg_f <= 0.5
            result = np.zeros_like(bg_f)
            result[mask] = 2.0 * bg_f[mask] * fg_f[mask]
            result[~mask] = 1.0 - 2.0 * (1.0 - bg_f[~mask]) * (1.0 - fg_f[~mask])
            return result
        elif blend_mode == "vivid_light":
            mask = fg_f <= 0.5
            result = np.zeros_like(bg_f)
            result[mask] = 1.0 - np.minimum((1.0 - bg_f[mask]) / (2.0 * fg_f[mask] + 1e-6), 1.0)
            result[~mask] = np.minimum(bg_f[~mask] / (2.0 * (1.0 - fg_f[~mask]) + 1e-6), 1.0)
            return result
        elif blend_mode == "linear_light":
            return bg_f + 2.0 * fg_f - 1.0
        elif blend_mode == "pin_light":
            mask = fg_f <= 0.5
            result = np.zeros_like(bg_f)
            result[mask] = np.minimum(bg_f[mask], 2.0 * fg_f[mask])
            result[~mask] = np.maximum(bg_f[~mask], 2.0 * fg_f[~mask] - 1.0)
            return result
        elif blend_mode == "hard_mix":
            vivid = self.apply_blend_mode(bg_f, fg_f, "vivid_light")
            return (vivid >= 0.5).astype(np.float32)
        elif blend_mode == "difference":
            return np.abs(bg_f - fg_f)
        elif blend_mode == "exclusion":
            return bg_f + fg_f - 2.0 * bg_f * fg_f
        elif blend_mode == "subtract":
            return np.clip(bg_f - fg_f, 0.0, 1.0)
        elif blend_mode == "divide":
            return np.minimum(bg_f / (fg_f + 1e-6), 1.0)
        elif blend_mode in ["hue", "saturation", "color", "luminosity"]:
            return self.apply_hsl_blend_mode(bg_f, fg_f, blend_mode)
        else:  # normal
            return fg_f

    def apply_hsl_blend_mode(self, bg_f, fg_f, blend_mode):
        """Apply HSL-based blend modes"""
        # Convert to HSL
        bg_hsl = rgb_to_hsl_batch(bg_f)
        fg_hsl = rgb_to_hsl_batch(fg_f)

        if blend_mode == "hue":
            # Use hue from overlay, saturation and lightness from background
            result_hsl = bg_hsl.copy()
            result_hsl[:, :, 0] = fg_hsl[:, :, 0]  # Use overlay hue
        elif blend_mode == "saturation":
            # Use saturation from overlay, hue and lightness from background
            result_hsl = bg_hsl.copy()
            result_hsl[:, :, 1] = fg_hsl[:, :, 1]  # Use overlay saturation
        elif blend_mode == "color":
            # Use hue and saturation from overlay, lightness from background
            result_hsl = bg_hsl.copy()
            result_hsl[:, :, 0] = fg_hsl[:, :, 0]  # Use overlay hue
            result_hsl[:, :, 1] = fg_hsl[:, :, 1]  # Use overlay saturation
        elif blend_mode == "luminosity":
            # Use lightness from overlay, hue and saturation from background
            result_hsl = bg_hsl.copy()
            result_hsl[:, :, 2] = fg_hsl[:, :, 2]  # Use overlay lightness
        else:
            result_hsl = fg_hsl

        # Convert back to RGB
        return hsl_to_rgb_batch(result_hsl)


class RC_LoadImageWithAlpha:
    """RC 透明图像加载器 | RC Transparent Image Loader"""

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "RGBA_mode": ("BOOLEAN", {"default": True, "tooltip": "强制输出 RGBA。若图像无 alpha，将添加全不透明通道 | Force RGBA output. If no alpha, add opaque channel."})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "RC/Image"
    DESCRIPTION = "加载图像并保留透明通道（PNG）。输出 IMAGE (RGBA) 和 MASK（alpha）| Load image preserving alpha channel (PNG). Outputs IMAGE (RGBA) and MASK (alpha)."

    def load_image(self, image: str, RGBA_mode: bool = True):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)

        # 转为 RGBA（保留 alpha）
        if img.mode in ("RGBA", "LA"):
            # 已有 alpha
            pass
        elif img.mode == "RGB":
            if RGBA_mode:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
        else:
            img = img.convert("RGBA")

        # 转为 numpy [H, W, C]
        img_np = np.array(img).astype(np.float32) / 255.0

        # 确保是 [H, W, 4] if RGBA_mode
        if RGBA_mode and img_np.shape[2] == 3:
            alpha = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.float32)
            img_np = np.concatenate([img_np, alpha], axis=2)

        # 输出 IMAGE: [1, H, W, C]
        image_tensor = torch.from_numpy(img_np)[None,]

        # 输出 MASK: [H, W] —— alpha 通道（用于后续 inpaint 等）
        if img_np.shape[2] == 4:
            mask = 1.0 - img_np[:, :, 3]  # 注意：MASK 是 1-alpha（ComfyUI 惯例：白色=遮罩）
        else:
            mask = torch.zeros((img_np.shape[0], img_np.shape[1]), dtype=torch.float32)

        mask_tensor = torch.from_numpy(mask)[None,]  # [1, H, W]

        return (image_tensor, mask_tensor)

    @classmethod
    def IS_CHANGED(s, image, RGBA_mode=True):
        image_path = folder_paths.get_annotated_filepath(image)
        m = os.path.getmtime(image_path)
        return m

    @classmethod
    def VALIDATE_INPUTS(s, image, RGBA_mode=True):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
