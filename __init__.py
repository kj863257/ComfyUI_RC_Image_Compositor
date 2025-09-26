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

# Import channel operations
from .channel_ops import RC_ChannelExtractor, RC_MaskApply

# Import gradient generator
from .gradient_generator import RC_GradientGenerator

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

    # Channel operations
    "RC_ChannelExtractor": RC_ChannelExtractor,
    "RC_MaskApply": RC_MaskApply,
    
    # Gradient operations
    "RC_GradientGenerator": RC_GradientGenerator,
}

# Display name mappings - shown in ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    # Core compositor nodes
    "RC_ImageCompositor": "RC Image Compositor (Complete)",
    "RC_LoadImageWithAlpha": "RC Load Image (Alpha)",

    # Layer style effects
    "RC_DropShadow": "RC Drop Shadow",
    "RC_Stroke": "RC Stroke",
    "RC_OuterGlow": "RC Outer Glow",

    # Filters and adjustments
    "RC_GaussianBlur": "RC Gaussian Blur",
    "RC_Sharpen": "RC Sharpen",
    "RC_HueSaturation": "RC Hue/Saturation",

    # Utility nodes
    "RC_CanvasPadding": "RC Canvas Padding",
    "RC_ImageScale": "RC Image Scale",
    "RC_ImageCrop": "RC Image Crop",
    "RC_CanvasResize": "RC Canvas Resize",

    # Adjustment nodes
    "RC_OpacityAdjust": "RC Opacity Adjust",
    "RC_LevelsAdjust": "RC Levels",
    "RC_BrightnessContrast": "RC Brightness/Contrast",
    "RC_ColorBalance": "RC Color Balance",
    "RC_ChannelMixer": "RC Channel Mixer",

    # Channel operations
    "RC_ChannelExtractor": "RC Channel Extractor",
    "RC_MaskApply": "RC Mask Apply",
    
    # Gradient operations
    "RC_GradientGenerator": "RC Gradient Generator",
}

WEB_DIRECTORY = "./js"

# Plugin metadata
__version__ = "2.0.0"
__description__ = "Professional Photoshop-style layer effects and compositing for ComfyUI"
__author__ = "RC Studio"

# Inform user about the plugin capabilities
print(f"\n🎨 RC Image Compositor v{__version__} loaded successfully!")
print("   ✨ Complete 24 Photoshop Blend Modes + Layer Effects")
print("   🔧 Utilities: Canvas Padding, Image Scale, Crop, Canvas Resize")
print("   🎚️  Adjustments: Opacity, Levels, Brightness/Contrast, Color Balance, Channel Mixer")
print("   🌐 Full bilingual support")
print("   📁 Modular architecture for professional workflows")
print("   🎯 Professional Photoshop-grade effects\n")
