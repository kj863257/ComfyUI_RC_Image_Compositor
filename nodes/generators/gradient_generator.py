import torch
import numpy as np
import json


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
                }),
                "gradient_type": (["linear", "radial", "angular", "reflected"], {
                    "default": "linear",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                }),
                "size_mode": (["from_image", "custom"], {
                    "default": "custom",
                }),
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8,
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                }),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_gradient"
    CATEGORY = "RC/Generate"

    def parse_gradient_data(self, gradient_data_str):
        """Parse gradient data from JSON string"""
        try:
            data = json.loads(gradient_data_str)
            stops = data.get("stops", [])
            stops.sort(key=lambda x: x["position"])
            
            if len(stops) < 2:
                stops = [
                    {"position": 0.0, "color": [0, 0, 0, 255]},
                    {"position": 1.0, "color": [255, 255, 255, 255]}
                ]
            
            return stops
        except:
            return [
                {"position": 0.0, "color": [0, 0, 0, 255]},
                {"position": 1.0, "color": [255, 255, 255, 255]}
            ]

    def create_gradient_lut(self, stops, size=1024):
        """Create lookup table for gradient colors"""
        positions = np.linspace(0.0, 1.0, size, dtype=np.float32)
        stop_positions = np.array([stop["position"] for stop in stops], dtype=np.float32)
        stop_positions = np.clip(stop_positions, 0.0, 1.0)
        stop_colors = np.array([stop["color"] for stop in stops], dtype=np.float32) / 255.0

        unique_positions, unique_indices = np.unique(stop_positions, return_index=True)
        stop_colors = stop_colors[unique_indices]

        if unique_positions.size == 1:
            return np.tile(stop_colors[0], (size, 1))

        lut = np.empty((size, 4), dtype=np.float32)
        for channel in range(4):
            lut[:, channel] = np.interp(positions, unique_positions, stop_colors[:, channel])

        return lut

    def generate_gradient(self, gradient_data, gradient_type, angle, size_mode,
                         width, height, center_x, center_y, scale, reference_image=None):

        # Validate and fix size_mode
        if size_mode not in ["from_image", "custom"]:
            size_mode = "custom"

        # Determine output size
        if size_mode == "from_image" and reference_image is not None:
            ref_shape = reference_image.shape
            height = ref_shape[1]
            width = ref_shape[2]
        
        # Parse gradient data
        stops = self.parse_gradient_data(gradient_data)
        
        # Create gradient LUT for efficient color lookup
        lut = self.create_gradient_lut(stops, size=1024)
        
        # Create coordinate grids - 直接在目标尺寸生成，避免缩放
        y_coords = np.linspace(0, 1, height, dtype=np.float32)
        x_coords = np.linspace(0, 1, width, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Calculate gradient positions based on type
        if gradient_type == "linear":
            rad = np.radians(angle)
            x_rot = (x_grid - 0.5) * np.cos(rad) - (y_grid - 0.5) * np.sin(rad)
            positions = (x_rot / scale + 0.5)
            
        elif gradient_type == "reflected":
            rad = np.radians(angle)
            x_rot = (x_grid - 0.5) * np.cos(rad) - (y_grid - 0.5) * np.sin(rad)
            positions = np.abs(x_rot / scale) * 2
            
        elif gradient_type == "radial":
            dx = (x_grid - center_x) / scale
            dy = (y_grid - center_y) / scale
            positions = np.sqrt(dx**2 + dy**2) * 2
            
        elif gradient_type == "angular":
            dx = x_grid - center_x
            dy = y_grid - center_y
            positions = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
            positions = (positions + angle / 360.0) % 1.0
        
        # Clip positions
        positions = np.clip(positions, 0.0, 1.0)
        
        # Use LUT for efficient color mapping
        indices = (positions * (len(lut) - 1)).astype(np.int32)
        result = lut[indices]
        
        # Extract RGB and alpha
        rgb = result[:, :, :3]
        alpha = result[:, :, 3]
        
        # Create RGBA image
        rgba = np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=2)
        
        # Convert to tensor
        image_out = torch.from_numpy(rgba).unsqueeze(0)  # [1, H, W, 4]

        return (image_out,)


NODE_CLASS_MAPPINGS = {
    "RC_GradientGenerator": RC_GradientGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RC_GradientGenerator": "RC Gradient Generator",
}
