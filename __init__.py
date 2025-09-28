# Import core nodes
from .nodes.core.compositor import RC_ImageCompositor, RC_LoadImageWithAlpha

# Import layer style effects
from .nodes.effects.layer_styles import RC_DropShadow, RC_Stroke, RC_OuterGlow

# Import filters and adjustments
from .nodes.generators.filters import RC_GaussianBlur, RC_Sharpen, RC_HueSaturation, RC_AddNoise

# Import utility nodes
from .nodes.utilities.utilities import RC_CanvasPadding, RC_ImageScale, RC_ImageCrop, RC_CanvasResize

# Import adjustment nodes
from .nodes.adjustments.adjustments import (
    RC_OpacityAdjust,
    RC_LevelsAdjust,
    RC_BrightnessContrast,
    RC_ColorBalance,
    RC_ChannelMixer,
    RC_CurvesAdjust,
    RC_Threshold,
    RC_Vibrance,
)

# Import channel operations
from .nodes.utilities.channel_ops import RC_ChannelExtractor, RC_MaskApply

# Import gradient generator
from .nodes.generators.gradient_generator import RC_GradientGenerator

# Import auto color correction
from .nodes.adjustments.auto_color import RC_AutoColor

# Import skin smoothing
from .nodes.generators.skin_smoothing import RC_HighLowFrequencySkinSmoothing

# Import gradient map
from .nodes.adjustments.gradient_map import RC_GradientMap

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
    "RC_AddNoise": RC_AddNoise,

    # Utility nodes
    "RC_CanvasPadding": RC_CanvasPadding,
    "RC_ImageScale": RC_ImageScale,
    "RC_ImageCrop": RC_ImageCrop,
    "RC_CanvasResize": RC_CanvasResize,

    # Adjustment nodes
    "RC_OpacityAdjust": RC_OpacityAdjust,
    "RC_LevelsAdjust": RC_LevelsAdjust,
    "RC_CurvesAdjust": RC_CurvesAdjust,
    "RC_BrightnessContrast": RC_BrightnessContrast,
    "RC_ColorBalance": RC_ColorBalance,
    "RC_ChannelMixer": RC_ChannelMixer,
    "RC_GradientMap": RC_GradientMap,
    "RC_Threshold": RC_Threshold,
    "RC_Vibrance": RC_Vibrance,

    # Channel operations
    "RC_ChannelExtractor": RC_ChannelExtractor,
    "RC_MaskApply": RC_MaskApply,
    
    # Gradient operations
    "RC_GradientGenerator": RC_GradientGenerator,

    # Auto color correction
    "RC_AutoColor": RC_AutoColor,

    # Skin smoothing
    "RC_HighLowFrequencySkinSmoothing": RC_HighLowFrequencySkinSmoothing,
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
    "RC_AddNoise": "RC Add Noise",

    # Utility nodes
    "RC_CanvasPadding": "RC Canvas Padding",
    "RC_ImageScale": "RC Image Scale",
    "RC_ImageCrop": "RC Image Crop",
    "RC_CanvasResize": "RC Canvas Resize",

    # Adjustment nodes
    "RC_OpacityAdjust": "RC Opacity Adjust",
    "RC_LevelsAdjust": "RC Levels",
    "RC_CurvesAdjust": "RC Curves",
    "RC_BrightnessContrast": "RC Brightness/Contrast",
    "RC_ColorBalance": "RC Color Balance",
    "RC_ChannelMixer": "RC Channel Mixer",
    "RC_GradientMap": "RC Gradient Map",
    "RC_Threshold": "RC Threshold",
    "RC_Vibrance": "RC Vibrance",

    # Channel operations
    "RC_ChannelExtractor": "RC Channel Extractor",
    "RC_MaskApply": "RC Mask Apply",
    
    # Gradient operations
    "RC_GradientGenerator": "RC Gradient Generator",

    # Auto color correction
    "RC_AutoColor": "RC Auto Color Correction",

    # Skin smoothing
    "RC_HighLowFrequencySkinSmoothing": "RC High/Low Frequency Skin Smoothing",
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
print("   🎚️  Adjustments: Opacity, Levels, Curves, Brightness/Contrast, Color Balance, Channel Mixer")
print("   🌐 Full bilingual support")
print("   📁 Modular architecture for professional workflows")
print("   🎯 Professional Photoshop-grade effects\n")
