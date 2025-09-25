# Import core nodes
from .compositor import RC_ImageCompositor, RC_LoadImageWithAlpha

# Import layer style effects
from .layer_styles import RC_DropShadow, RC_Stroke, RC_OuterGlow

# Import filters and adjustments
from .filters import RC_GaussianBlur, RC_Sharpen, RC_HueSaturation

# Import utility nodes
from .utilities import RC_CanvasPadding, RC_ImageScale, RC_ImageCrop, RC_CanvasResize

# Import adjustment nodes
from .adjustments import RC_OpacityAdjust, RC_LevelsAdjust, RC_BrightnessContrast, RC_ColorBalance, RC_ChannelMixer

# Node class mappings - used by ComfyUI to register nodes
NODE_CLASS_MAPPINGS = {
    # Core compositor nodes
    "RC_ImageCompositor": RC_ImageCompositor,
    "RC_LoadImageWithAlpha": RC_LoadImageWithAlpha,

    # Layer style effects
    "RC_DropShadow": RC_DropShadow,
    "RC_Stroke": RC_Stroke,
    "RC_OuterGlow": RC_OuterGlow,

    # Filters and adjustments
    "RC_GaussianBlur": RC_GaussianBlur,
    "RC_Sharpen": RC_Sharpen,
    "RC_HueSaturation": RC_HueSaturation,

    # Utility nodes
    "RC_CanvasPadding": RC_CanvasPadding,
    "RC_ImageScale": RC_ImageScale,
    "RC_ImageCrop": RC_ImageCrop,
    "RC_CanvasResize": RC_CanvasResize,

    # Adjustment nodes
    "RC_OpacityAdjust": RC_OpacityAdjust,
    "RC_LevelsAdjust": RC_LevelsAdjust,
    "RC_BrightnessContrast": RC_BrightnessContrast,
    "RC_ColorBalance": RC_ColorBalance,
    "RC_ChannelMixer": RC_ChannelMixer,
}

# Display name mappings - shown in ComfyUI interface (中英双语)
NODE_DISPLAY_NAME_MAPPINGS = {
    # Core compositor nodes
    "RC_ImageCompositor": "RC 图像合成器 (完整版) | RC Image Compositor (Complete)",
    "RC_LoadImageWithAlpha": "RC 加载透明图像 | RC Load Image (Alpha)",

    # Layer style effects
    "RC_DropShadow": "RC 投影效果 | RC Drop Shadow",
    "RC_Stroke": "RC 描边效果 | RC Stroke",
    "RC_OuterGlow": "RC 外发光效果 | RC Outer Glow",

    # Filters and adjustments
    "RC_GaussianBlur": "RC 高斯模糊 | RC Gaussian Blur",
    "RC_Sharpen": "RC 锐化滤镜 | RC Sharpen",
    "RC_HueSaturation": "RC 色相/饱和度 | RC Hue/Saturation",

    # Utility nodes
    "RC_CanvasPadding": "RC 画布填充 | RC Canvas Padding",
    "RC_ImageScale": "RC 图像缩放 | RC Image Scale",
    "RC_ImageCrop": "RC 图像裁剪 | RC Image Crop",
    "RC_CanvasResize": "RC 画布调整 | RC Canvas Resize",

    # Adjustment nodes
    "RC_OpacityAdjust": "RC 透明度调整 | RC Opacity Adjust",
    "RC_LevelsAdjust": "RC 色阶调整 | RC Levels",
    "RC_BrightnessContrast": "RC 亮度/对比度 | RC Brightness/Contrast",
    "RC_ColorBalance": "RC 色彩平衡 | RC Color Balance",
    "RC_ChannelMixer": "RC 通道混合器 | RC Channel Mixer",
}

# Plugin metadata
__version__ = "2.0.0"
__description__ = "Professional Photoshop-style layer effects and compositing for ComfyUI | 专业的 Photoshop 风格图层效果和合成工具"
__author__ = "RC Studio"

# Inform user about the plugin capabilities
print(f"\n🎨 RC Image Compositor v{__version__} loaded successfully!")
print("   ✨ Complete 24 Photoshop Blend Modes + Layer Effects")
print("   🔧 Utilities: Canvas Padding, Image Scale, Crop, Canvas Resize")
print("   🎚️  Adjustments: Opacity, Levels, Brightness/Contrast, Color Balance, Channel Mixer")
print("   🌐 Full bilingual support (中英双语)")
print("   📁 Modular architecture for professional workflows")
print("   🎯 Professional Photoshop-grade effects\n")
