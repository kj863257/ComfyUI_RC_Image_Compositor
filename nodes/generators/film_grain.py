import torch
import numpy as np


class RC_FilmGrain:
    """Realistic film grain effect simulating analog film characteristics"""
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "grain_map")
    FUNCTION = "apply_grain"
    CATEGORY = "RC/Filters"
    DESCRIPTION = "Add realistic film grain with luminance-based intensity and adjustable characteristics."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Overall grain intensity (0=no grain, 1.0=heavy grain)"
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.1,
                    "tooltip": "Grain particle size (smaller = finer grain, larger = coarser grain)"
                }),
                "luminance_response": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Luminance-based grain variation:\n- 0.0: Uniform grain across all tones\n- 1.0: Realistic film (DARKER areas get MORE grain, BRIGHTER areas get LESS grain)\n- 2.0: Extreme film grain (maximum grain in shadows, minimal in highlights)"
                }),
                "grain_midtone": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Grain distribution midtone (similar to Levels midtone slider):\n- 0.5: Standard distribution\n- <0.5: Pull midtones darker (MORE areas get grain)\n- >0.5: Pull midtones brighter (FEWER areas get grain, protect highlights)"
                }),
                "highlight_protection": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Additional highlight protection (applies after midtone adjustment):\n- 0.0: No additional protection\n- 0.5: Moderate black compression (stronger highlight protection)\n- 1.0: Maximum black compression (only deepest shadows get grain)"
                }),
                "grain_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Grain layer opacity (final blend strength):\n- 0.0: No grain visible\n- 0.5: 50% grain blend\n- 1.0: Full grain effect"
                }),
                "grain_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Blur applied to grain layer (soften grain):\n- 0.0: No blur (sharp grain)\n- 1.0: Slight softening\n- 3.0+: Very soft, subtle grain"
                }),
                "monochrome": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Monochrome grain (true for B&W film look, false for color film with chromatic grain)"
                }),
                "blend_mode": (["add", "screen", "overlay", "soft_light", "linear_dodge"], {
                    "default": "overlay",
                    "tooltip": "Grain blending mode:\n- add: Direct addition (strong, visible)\n- screen: Film-like brightening (soft, subtle)\n- overlay: Photoshop overlay (natural contrast)\n- soft_light: Photoshop soft light (gentle, smooth)\n- linear_dodge: Linear brightening (bright, glowy)"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible grain patterns"
                }),
            },
        }

    def generate_grain_layer(self, batch, height, width, channels, grain_size, monochrome, device, seed):
        """Generate realistic fine-grained noise layer"""
        generator = torch.Generator(device=device).manual_seed(seed)

        if monochrome:
            # Single channel grain for monochrome effect
            noise = torch.randn((batch, height, width, 1), generator=generator, device=device)
        else:
            # Per-channel grain for chromatic effect
            noise = torch.randn((batch, height, width, channels), generator=generator, device=device)

        # Apply slight smoothing based on grain size to simulate grain clumping
        if grain_size > 1.0:
            # Larger grain size: apply light blur
            kernel_size = int(grain_size * 2) * 2 + 1  # Ensure odd number
            noise = noise.permute(0, 3, 1, 2)  # BHWC -> BCHW

            # Create Gaussian kernel
            sigma = grain_size * 0.5
            kernel_range = torch.arange(kernel_size, device=device) - kernel_size // 2
            kernel = torch.exp(-kernel_range**2 / (2 * sigma**2))
            kernel = kernel / kernel.sum()

            # Apply separable 2D convolution
            # Horizontal
            kernel_h = kernel.view(1, 1, 1, -1).expand(noise.shape[1], 1, 1, -1)
            padding_h = kernel_size // 2
            noise = torch.nn.functional.conv2d(noise, kernel_h, padding=(0, padding_h), groups=noise.shape[1])

            # Vertical
            kernel_v = kernel.view(1, 1, -1, 1).expand(noise.shape[1], 1, -1, 1)
            padding_v = kernel_size // 2
            noise = torch.nn.functional.conv2d(noise, kernel_v, padding=(padding_v, 0), groups=noise.shape[1])

            noise = noise.permute(0, 2, 3, 1)  # BCHW -> BHWC

        # Normalize to zero mean, unit variance
        mean = noise.mean(dim=(1, 2, 3), keepdim=True)
        std = noise.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        grain = (noise - mean) / std

        if monochrome:
            # Expand to all channels
            grain = grain.expand(-1, -1, -1, channels)

        return grain

    def compute_luminance_mask(self, image):
        """Compute luminance for each pixel"""
        # Standard luminance calculation
        if image.shape[-1] >= 3:
            luminance = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]
        else:
            luminance = image[..., 0:1]
        return luminance

    def blend_grain(self, image, grain, blend_mode):
        """Blend grain with image using specified mode"""
        if blend_mode == "add":
            # Simple addition
            return image + grain
        elif blend_mode == "screen":
            # Photoshop screen blend mode: 1 - (1-base) * (1-blend)
            # Remap grain from [-1,1] to [0,1]
            blend = grain * 0.5 + 0.5
            return 1.0 - (1.0 - image) * (1.0 - blend)
        elif blend_mode == "linear_dodge":
            # Linear dodge (add): base + blend
            # Remap grain from [-1,1] to [0,1]
            blend = grain * 0.5 + 0.5
            return image + blend
        elif blend_mode == "overlay":
            # Photoshop overlay blend mode
            mask = (image < 0.5).float()
            # For dark areas: 2 * base * blend
            dark = 2.0 * image * (grain * 0.5 + 0.5)
            # For light areas: 1 - 2 * (1-base) * (1-blend)
            light = 1.0 - 2.0 * (1.0 - image) * (1.0 - (grain * 0.5 + 0.5))
            return mask * dark + (1.0 - mask) * light
        else:  # soft_light
            # Photoshop soft light blend mode
            blend = grain * 0.5 + 0.5  # Remap from [-1,1] to [0,1]
            mask = (blend < 0.5).float()
            # For dark blend: base - (1-2*blend) * base * (1-base)
            dark = image - (1.0 - 2.0 * blend) * image * (1.0 - image)
            # For light blend: base + (2*blend-1) * (sqrt(base) - base)
            light = image + (2.0 * blend - 1.0) * (torch.sqrt(torch.clamp(image, 0.0, 1.0)) - image)
            return mask * dark + (1.0 - mask) * light

    def apply_grain(self, image, intensity, grain_size, luminance_response,
                   grain_midtone, highlight_protection, grain_opacity, grain_blur,
                   monochrome, blend_mode, seed):

        device = image.device
        batch, height, width, channels = image.shape

        # Scale intensity: user range 0-1 maps to internal range 0-0.2
        intensity = intensity * 0.2

        # Generate fine grain layer
        grain = self.generate_grain_layer(batch, height, width, channels,
                                         grain_size, monochrome, device, seed)

        # Apply optional blur to grain layer for softer, more subtle grain
        if grain_blur > 0.0:
            kernel_size = int(grain_blur * 2) * 2 + 1  # Ensure odd number
            grain = grain.permute(0, 3, 1, 2)  # BHWC -> BCHW

            # Create Gaussian kernel
            sigma = grain_blur * 0.5
            kernel_range = torch.arange(kernel_size, device=device) - kernel_size // 2
            kernel = torch.exp(-kernel_range**2 / (2 * sigma**2))
            kernel = kernel / kernel.sum()

            # Apply separable 2D convolution
            # Horizontal
            kernel_h = kernel.view(1, 1, 1, -1).expand(grain.shape[1], 1, 1, -1)
            padding_h = kernel_size // 2
            grain = torch.nn.functional.conv2d(grain, kernel_h, padding=(0, padding_h), groups=grain.shape[1])

            # Vertical
            kernel_v = kernel.view(1, 1, -1, 1).expand(grain.shape[1], 1, -1, 1)
            padding_v = kernel_size // 2
            grain = torch.nn.functional.conv2d(grain, kernel_v, padding=(padding_v, 0), groups=grain.shape[1])

            grain = grain.permute(0, 2, 3, 1)  # BCHW -> BHWC

        # Compute luminance-based intensity modulation
        if luminance_response > 0.0:
            luminance = self.compute_luminance_mask(image)

            # Calculate shadow and highlight intensities based on response strength
            # Stronger curve for more dramatic effect
            # response=0: shadow=1.0, highlight=1.0 (uniform)
            # response=1.0: shadow=2.0, highlight=0.0 (dramatic film grain)
            # response=2.0: shadow=3.0, highlight=0.0 (extreme film grain)
            shadow_intensity = 1.0 + luminance_response * 1.0
            highlight_intensity = max(0.0, 1.0 - luminance_response * 1.0)

            # Create intensity map: more grain in shadows (dark areas), less in highlights (bright areas)
            # luminance is high in bright areas, low in dark areas
            # So (1.0 - luminance) is high in dark areas, low in bright areas
            intensity_map = (1.0 - luminance) * (shadow_intensity - highlight_intensity) + highlight_intensity

            # Apply midtone adjustment (similar to Photoshop Levels midtone slider)
            # midtone=0.5: no change (gamma=1.0)
            # midtone<0.5: brighten grain_map (more areas get grain)
            # midtone>0.5: darken grain_map (fewer areas get grain, protect highlights)
            if grain_midtone != 0.5:
                # Convert midtone to gamma: exponential mapping for wider range
                # gamma range: 0.125 (midtone=0.0) to 8.0 (midtone=1.0)
                gamma = 2.0 ** ((2.0 * grain_midtone - 1.0) * 3.0)
                intensity_map = torch.pow(torch.clamp(intensity_map, 1e-6, 1.0), gamma)

            # Apply additional highlight protection if enabled (power curve to compress blacks)
            if highlight_protection > 0.0:
                # Higher protection = higher power = darker grain map blacks
                # This makes the transition steeper, protecting more of the bright areas
                power = 1.0 + highlight_protection * 3.0  # Range: 1.0 to 4.0
                intensity_map = torch.pow(torch.clamp(intensity_map, 1e-6, 1.0), power)
        else:
            # Uniform grain (no luminance response)
            intensity_map = torch.ones_like(image[..., :1])

        # Apply intensity to grain
        modulated_grain = grain * intensity * intensity_map

        # Blend grain with image
        if blend_mode == "add":
            result = image + modulated_grain
        else:
            # For overlay and soft_light, we blend the grain layer
            result = self.blend_grain(image, modulated_grain / intensity, blend_mode)
            # Mix back based on intensity
            result = image + (result - image) * intensity

        # Apply grain opacity to control final blend strength
        if grain_opacity < 1.0:
            result = image + (result - image) * grain_opacity

        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)

        # Create grain map visualization (shows where grain is applied)
        # Brighter areas = more grain, darker areas = less grain
        grain_map = intensity_map.squeeze(-1)  # Remove channel dimension for MASK format

        return (result, grain_map)
