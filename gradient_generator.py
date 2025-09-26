import torch
import numpy as np
import json
from PIL import Image


class RC_GradientGenerator:
    """RC 渐变生成器 | RC Gradient Generator

    创建渐变图片，支持透明度和多色渐变。
    Create gradient images with support for transparency and multi-color gradients.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gradient_data": ("STRING", {
                    "default": json.dumps({
                        "stops": [
                            {"position": 0.0, "color": [0, 0, 0, 255]},
                            {"position": 1.0, "color": [255, 255, 255, 255]}
                        ]
                    }),
                    "multiline": False,
                    "tooltip": "Gradient color stops data (JSON format)"
                }),
                "gradient_type": (["linear", "radial", "angular", "reflected"], {
                    "default": "linear",
                    "tooltip": "Gradient type:\n- linear: Linear gradient\n- radial: Radial gradient from center\n- angular: Angular/conical gradient\n- reflected: Reflected linear gradient"
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Gradient angle in degrees (for linear and reflected types)"
                }),
                "size_mode": (["from_image", "custom"], {
                    "default": "custom",
                    "tooltip": "Size mode:\n- from_image: Use reference image size\n- custom: Specify custom width and height"
                }),
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Width in pixels (used in custom mode)"
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Height in pixels (used in custom mode)"
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Horizontal center position (0-1, for radial and angular)"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Vertical center position (0-1, for radial and angular)"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Gradient scale/zoom factor"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference image for size (used when size_mode is from_image)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "generate_gradient"
    CATEGORY = "RC/Generate"
    DESCRIPTION = "Generate gradient images with transparency support and multiple color stops"

    def parse_gradient_data(self, gradient_data_str):
        """Parse gradient data from JSON string"""
        try:
            data = json.loads(gradient_data_str)
            stops = data.get("stops", [])
            
            # Sort stops by position
            stops.sort(key=lambda x: x["position"])
            
            # Ensure we have at least 2 stops
            if len(stops) < 2:
                stops = [
                    {"position": 0.0, "color": [0, 0, 0, 255]},
                    {"position": 1.0, "color": [255, 255, 255, 255]}
                ]
            
            return stops
        except:
            # Default gradient if parsing fails
            return [
                {"position": 0.0, "color": [0, 0, 0, 255]},
                {"position": 1.0, "color": [255, 255, 255, 255]}
            ]

    def interpolate_color(self, stops, position):
        """Interpolate color at given position based on gradient stops"""
        position = np.clip(position, 0.0, 1.0)
        
        # Find surrounding stops
        left_stop = stops[0]
        right_stop = stops[-1]
        
        for i in range(len(stops) - 1):
            if stops[i]["position"] <= position <= stops[i + 1]["position"]:
                left_stop = stops[i]
                right_stop = stops[i + 1]
                break
        
        # Calculate interpolation factor
        if right_stop["position"] == left_stop["position"]:
            t = 0.0
        else:
            t = (position - left_stop["position"]) / (right_stop["position"] - left_stop["position"])
        
        # Interpolate RGBA
        left_color = np.array(left_stop["color"], dtype=np.float32) / 255.0
        right_color = np.array(right_stop["color"], dtype=np.float32) / 255.0
        
        color = left_color * (1 - t) + right_color * t
        return color

    def generate_gradient(self, gradient_data, gradient_type, angle, size_mode, 
                         width, height, center_x, center_y, scale, reference_image=None):
        
        # Determine output size
        if size_mode == "from_image" and reference_image is not None:
            ref_shape = reference_image.shape
            height = ref_shape[1]
            width = ref_shape[2]
        
        # Parse gradient stops
        stops = self.parse_gradient_data(gradient_data)
        
        # Create coordinate grids
        y_coords = np.linspace(0, 1, height)
        x_coords = np.linspace(0, 1, width)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Calculate gradient positions based on type
        if gradient_type == "linear":
            # Convert angle to radians
            rad = np.radians(angle)
            
            # Rotate coordinates
            x_rot = (x_grid - 0.5) * np.cos(rad) - (y_grid - 0.5) * np.sin(rad)
            y_rot = (x_grid - 0.5) * np.sin(rad) + (y_grid - 0.5) * np.cos(rad)
            
            # Calculate position along gradient line
            positions = (x_rot / scale + 0.5)
            
        elif gradient_type == "reflected":
            # Similar to linear but reflected
            rad = np.radians(angle)
            x_rot = (x_grid - 0.5) * np.cos(rad) - (y_grid - 0.5) * np.sin(rad)
            positions = np.abs(x_rot / scale) * 2
            
        elif gradient_type == "radial":
            # Distance from center
            dx = (x_grid - center_x) / scale
            dy = (y_grid - center_y) / scale
            positions = np.sqrt(dx**2 + dy**2) * 2
            
        elif gradient_type == "angular":
            # Angle around center
            dx = x_grid - center_x
            dy = y_grid - center_y
            positions = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
            # Apply rotation
            positions = (positions + angle / 360.0) % 1.0
        
        # Clip positions to valid range
        positions = np.clip(positions, 0.0, 1.0)
        
        # Create output image (RGBA)
        result = np.zeros((height, width, 4), dtype=np.float32)
        
        # Apply gradient colors
        for i in range(height):
            for j in range(width):
                result[i, j] = self.interpolate_color(stops, positions[i, j])
        
        # Extract RGB and alpha
        rgb = result[:, :, :3]
        alpha = result[:, :, 3]
        
        # Create RGBA image
        rgba = np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=2)
        
        # Convert to tensors
        image_out = torch.from_numpy(rgba)[None, ...]  # [1, H, W, 4]
        mask_out = torch.from_numpy(alpha)[None, ...]   # [1, H, W]
        
        return (image_out, mask_out)


NODE_CLASS_MAPPINGS = {
    "RC_GradientGenerator": RC_GradientGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RC_GradientGenerator": "RC Gradient Generator",
}