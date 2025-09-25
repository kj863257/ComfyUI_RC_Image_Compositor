import torch
import numpy as np
from PIL import Image

class RC_Image_Compositor:
    """RC Image Compositor: Photoshop-style blend modes, precise positioning, and flexible scaling."""
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "RC/Image"
    DESCRIPTION = "Composite overlay with Photoshop-compatible blend modes, percentage positioning, scaling, rotation, and opacity."

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
                "x_offset": ("INT", {
                    "default": -50, "min": -4096, "max": 4096, "step": 1,
                    "tooltip": "在百分比基础上额外偏移的像素（可负）| Pixel offset on top of percentage (can be negative)"
                }),
                "y_offset": ("INT", {
                    "default": 50, "min": -4096, "max": 4096, "step": 1,
                    "tooltip": "同上 | Same as above"
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
                "blend_mode": ([
                    "normal",           # 正常
                    "darken",           # 变暗
                    "multiply",         # 正片叠底
                    "color_burn",       # 颜色加深
                    "linear_burn",      # 线性加深
                    "lighten",          # 变亮
                    "screen",           # 滤色
                    "color_dodge",      # 颜色减淡
                    "linear_dodge",     # 线性减淡（添加）
                    "overlay",          # 叠加
                    "soft_light",       # 柔光
                    "hard_light",       # 强光
                    "difference",       # 差值
                    "exclusion"         # 排除
                ], {
                    "default": "normal",
                    "tooltip": (
                        "Photoshop 混合模式（中英对照）| Photoshop Blend Modes:\n"
                        "- normal: 正常 | Normal\n"
                        "- darken: 变暗 | Darken\n"
                        "- multiply: 正片叠底 | Multiply\n"
                        "- color_burn: 颜色加深 | Color Burn\n"
                        "- linear_burn: 线性加深 | Linear Burn\n"
                        "- lighten: 变亮 | Lighten\n"
                        "- screen: 滤色 | Screen\n"
                        "- color_dodge: 颜色减淡 | Color Dodge\n"
                        "- linear_dodge: 线性减淡（添加）| Linear Dodge (Add)\n"
                        "- overlay: 叠加 | Overlay\n"
                        "- soft_light: 柔光 | Soft Light\n"
                        "- hard_light: 强光 | Hard Light\n"
                        "- difference: 差值 | Difference\n"
                        "- exclusion: 排除 | Exclusion"
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

        # === 5. Position ===
        fg_h, fg_w = fg_resized.shape[:2]
        x_base = int((bg_w - fg_w) * x_percent / 100)
        y_base = int((bg_h - fg_h) * y_percent / 100)
        x = x_base + x_offset
        y = y_base + y_offset

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

                # --- Photoshop-compatible blend modes ---
                if blend_mode == "darken":
                    blended = np.minimum(bg_f, rgb)
                elif blend_mode == "multiply":
                    blended = bg_f * rgb
                elif blend_mode == "color_burn":
                    # Avoid division by zero
                    blended = 1.0 - np.minimum((1.0 - bg_f) / (rgb + 1e-6), 1.0)
                elif blend_mode == "linear_burn":
                    blended = bg_f + rgb - 1.0
                elif blend_mode == "lighten":
                    blended = np.maximum(bg_f, rgb)
                elif blend_mode == "screen":
                    blended = 1.0 - (1.0 - bg_f) * (1.0 - rgb)
                elif blend_mode == "color_dodge":
                    blended = np.minimum(bg_f / (1.0 - rgb + 1e-6), 1.0)
                elif blend_mode == "linear_dodge":
                    blended = bg_f + rgb
                elif blend_mode == "overlay":
                    mask = bg_f >= 0.5
                    blended = np.zeros_like(bg_f)
                    blended[mask] = 1.0 - 2.0 * (1.0 - bg_f[mask]) * (1.0 - rgb[mask])
                    blended[~mask] = 2.0 * bg_f[~mask] * rgb[~mask]
                elif blend_mode == "soft_light":
                    mask = rgb <= 0.5
                    blended = np.zeros_like(bg_f)
                    blended[mask] = bg_f[mask] - (1.0 - 2.0 * rgb[mask]) * bg_f[mask] * (1.0 - bg_f[mask])
                    blended[~mask] = bg_f[~mask] + (2.0 * rgb[~mask] - 1.0) * (np.sqrt(bg_f[~mask]) - bg_f[~mask])
                elif blend_mode == "hard_light":
                    mask = rgb <= 0.5
                    blended = np.zeros_like(bg_f)
                    blended[mask] = 2.0 * bg_f[mask] * rgb[mask]
                    blended[~mask] = 1.0 - 2.0 * (1.0 - bg_f[~mask]) * (1.0 - rgb[~mask])
                elif blend_mode == "difference":
                    blended = np.abs(bg_f - rgb)
                elif blend_mode == "exclusion":
                    blended = bg_f + rgb - 2.0 * bg_f * rgb
                else:  # normal
                    blended = rgb

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
