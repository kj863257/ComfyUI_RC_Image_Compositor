import os
import json
import random
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from nodes import SaveImage as BaseSaveImage


class RC_PreviewImageWithMetadata:
    """Preview image with transferred metadata from source image file."""

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_with_metadata"
    CATEGORY = "RC/Utilities"
    DESCRIPTION = "Transfer workflow metadata from source image file to target image and generate preview."

    @classmethod
    def IS_CHANGED(cls, source_image_file, **kwargs):
        # Force re-execution when source image changes
        import hashlib
        image_path = folder_paths.get_annotated_filepath(source_image_file)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return {
            "required": {
                "target_image": ("IMAGE", {
                    "tooltip": "Target image to receive metadata"
                }),
                "source_image_file": (sorted(files), {
                    "image_upload": True,
                    "tooltip": "Source image file containing workflow metadata to extract"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        self.compress_level = 4

    def preview_with_metadata(self, target_image, source_image_file, prompt=None, extra_pnginfo=None):
        # Extract metadata from source image file
        source_metadata = self._extract_metadata_from_file(source_image_file)

        # Get source image info
        source_info = self._get_source_image_info(source_image_file)

        # Save preview with transferred metadata
        preview_result = self._save_with_metadata(target_image, source_metadata, "ComfyUI")

        # Combine source and target images into one list
        # Source images first, then target images
        all_images = source_info + preview_result["ui"]["images"]

        return {
            "ui": {
                "images": all_images
            }
        }

    def _get_source_image_info(self, image_filename):
        """Get source image info for preview."""
        return [{
            "filename": image_filename,
            "subfolder": "",
            "type": "input"
        }]

    def _extract_metadata_from_file(self, image_filename):
        """Extract PNG metadata from source image file."""
        image_path = folder_paths.get_annotated_filepath(image_filename)

        metadata = {}
        try:
            with Image.open(image_path) as img:
                # Extract PNG text chunks
                if hasattr(img, 'text'):
                    for key, value in img.text.items():
                        metadata[key] = value
        except Exception as e:
            print(f"Warning: Could not extract metadata from {image_filename}: {e}")

        return metadata

    def _save_with_metadata(self, images, metadata, filename_prefix):
        """Save images with metadata."""
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0]
        )

        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Create PNG info with metadata
            pnginfo = None
            if metadata:
                pnginfo = PngInfo()
                for key, value in metadata.items():
                    if isinstance(value, str):
                        pnginfo.add_text(key, value)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=pnginfo,
                compress_level=self.compress_level
            )

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


class RC_SaveImageWithMetadata(BaseSaveImage):
    """Save image with transferred metadata from source image file."""

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "RC/Utilities"
    DESCRIPTION = "Transfer workflow metadata from source image file to target image and save to disk."

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return {
            "required": {
                "target_image": ("IMAGE", {
                    "tooltip": "Target image to save with transferred metadata"
                }),
                "source_image_file": (sorted(files), {
                    "image_upload": True,
                    "tooltip": "Source image file containing workflow metadata to extract"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Filename prefix for saved images"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    FUNCTION = "save_with_metadata"

    def save_with_metadata(self, target_image, source_image_file, filename_prefix="ComfyUI",
                          prompt=None, extra_pnginfo=None):
        """Save target image with metadata from source image file."""
        # Extract metadata from source file
        source_metadata = self._extract_metadata_from_file(source_image_file)

        # Get source image info
        source_info = self._get_source_image_info(source_image_file)

        # Prepare extra_pnginfo with source metadata
        if source_metadata:
            # Create new extra_pnginfo dict if needed
            if extra_pnginfo is None:
                extra_pnginfo = {}

            # Merge source metadata into extra_pnginfo
            for key, value in source_metadata.items():
                extra_pnginfo[key] = value

        # Save using base class method with merged metadata
        save_result = self.save_images(target_image, filename_prefix, prompt, extra_pnginfo)

        # Combine source and target images into one list
        if "ui" in save_result:
            all_images = source_info + save_result["ui"]["images"]
            save_result["ui"]["images"] = all_images

        return save_result

    def _get_source_image_info(self, image_filename):
        """Get source image info for preview."""
        return [{
            "filename": image_filename,
            "subfolder": "",
            "type": "input"
        }]

    def _extract_metadata_from_file(self, image_filename):
        """Extract PNG metadata from source image file."""
        image_path = folder_paths.get_annotated_filepath(image_filename)

        metadata = {}
        try:
            with Image.open(image_path) as img:
                # Extract PNG text chunks
                if hasattr(img, 'text'):
                    for key, value in img.text.items():
                        metadata[key] = value
        except Exception as e:
            print(f"Warning: Could not extract metadata from {image_filename}: {e}")

        return metadata
