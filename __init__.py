from .rc_image_compository import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "RC_Image_Compositor": RC_Image_Compositor,
    "RC_LoadImageWithAlpha": LoadImageWithAlpha
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RC_Image_Compositor": "RC 图像合成器 (Photoshop 混合模式)",
    "RC_LoadImageWithAlpha": "RC 加载图像（带透明通道）"
}
