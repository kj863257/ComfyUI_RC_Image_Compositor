import base64
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from aiohttp import web
from scipy.interpolate import interp1d
from server import PromptServer

SUPPORTED_PRESET_EXT = {".xmp", ".lrtemplate"}
SESSION_TTL_SECONDS = 900
MISSING_SELECTION_MESSAGE = "请先在浏览器中选择一个预设 / Select a preset from the browser before continuing."
PASS_FORWARD_MESSAGE = "预览满意后请开启 Pass Forward 继续执行 / Enable Pass Forward after previewing a preset to continue."


def _format_relative_path(root_dir: str, target_path: str) -> str:
    """Return POSIX-style relative path for UI rendering."""
    if not root_dir:
        return target_path.replace("\\", "/")
    rel_path = os.path.relpath(target_path, root_dir)
    if rel_path == ".":
        return ""
    return rel_path.replace("\\", "/")


class RC_LRPreset:
    """Lightroom/Camera Raw Preset Loader with interactive browser support."""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_preset"
    CATEGORY = "RC/Adjustments"
    DESCRIPTION = (
        "Browse Lightroom (.lrtemplate) or Camera Raw (.xmp) preset directories, "
        "preview them interactively, then apply the selection with adjustable strength."
    )
    SESSION_CACHE: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to grade with the selected preset"
                }),
                "preset_directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Root directory that will be scanned for .xmp or .lrtemplate presets (supports subfolders)"
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Preset influence (0=no effect, 100=full preset). Preview uses the same value."
                }),
                "pass_forward": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Enabled",
                    "label_off": "Blocked",
                    "tooltip": "Run once with Blocked to pick a preset, then enable to continue the workflow."
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "selected_preset": ("STRING", {"default": ""}),
                "browser_state": ("STRING", {"default": "{}"}),
            }
        }

    @staticmethod
    def _resolve_root_dir(path_value: str) -> str:
        if not path_value:
            return ""

        def _clean_path(raw: str) -> str:
            trimmed = raw.strip()
            if not trimmed:
                return ""
            if (trimmed.startswith('"') and trimmed.endswith('"')) or (trimmed.startswith("'") and trimmed.endswith("'")):
                trimmed = trimmed[1:-1]
            return trimmed.strip()

        cleaned = _clean_path(path_value)
        if not cleaned:
            return ""

        candidate = os.path.abspath(os.path.expanduser(cleaned))
        if os.path.isdir(candidate):
            return candidate

        win_match = re.match(r'^([a-zA-Z]):[\\/](.*)$', cleaned)
        if win_match:
            drive = win_match.group(1).lower()
            remainder = win_match.group(2).replace("\\", "/")
            converted = os.path.abspath(os.path.join(f"/mnt/{drive}", remainder))
            if os.path.isdir(converted):
                return converted

        return ""

    @staticmethod
    def _safe_state_dict(raw_state: str) -> Dict[str, Any]:
        if not raw_state:
            return {}
        try:
            parsed = json.loads(raw_state)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _numpy_to_tensor(image_np: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0)

    @staticmethod
    def _tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
        np_img = image_tensor[0].detach().cpu().numpy()
        return np.clip(np_img, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _numpy_to_base64(image_np: np.ndarray) -> str:
        img_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_uint8)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def _prune_sessions(cls):
        now = time.time()
        stale_ids = [
            key for key, value in cls.SESSION_CACHE.items()
            if now - value.get("timestamp", 0) > SESSION_TTL_SECONDS
        ]
        for key in stale_ids:
            cls.SESSION_CACHE.pop(key, None)

    @classmethod
    def _cache_session(cls, node_id: str, image_np: np.ndarray, preset_directory: str,
                       browser_state: str):
        cls._prune_sessions()
        existing = cls.SESSION_CACHE.get(node_id, {})
        resolved_dir = cls._resolve_root_dir(preset_directory)

        preserved = {}
        if existing and existing.get("root_dir") == resolved_dir:
            preserved = {
                "selected_preset": existing.get("selected_preset"),
                "browser_state": existing.get("browser_state", "{}"),
                "current_path": existing.get("current_path", ""),
                "selected_preview": existing.get("selected_preview"),
                "last_preview_b64": existing.get("last_preview_b64"),
            }

        cached = {
            "image_np": image_np.copy(),
            "base64": cls._numpy_to_base64(image_np),
            "root_dir": resolved_dir,
            "display_dir": preset_directory or "",
            "browser_state": browser_state or preserved.get("browser_state", "{}") or "{}",
            "current_path": preserved.get("current_path", ""),
            "selected_preset": preserved.get("selected_preset"),
            "selected_preview": preserved.get("selected_preview"),
            "last_preview_b64": preserved.get("last_preview_b64"),
            "timestamp": time.time(),
        }

        if browser_state:
            cached["browser_state"] = browser_state
            try:
                parsed = json.loads(browser_state)
                cached["current_path"] = parsed.get("currentPath", cached.get("current_path", ""))
            except Exception:
                pass
        if (not cached["browser_state"] or cached["browser_state"] == "{}") and cached.get("current_path"):
            cached["browser_state"] = json.dumps({"currentPath": cached["current_path"]})

        cls.SESSION_CACHE[node_id] = cached
        return cached

    @staticmethod
    def _resolve_preset_path(root_dir: str, selection: str) -> str:
        if not selection:
            return ""
        candidate = selection.strip()
        if os.path.isabs(candidate) and os.path.isfile(candidate):
            return candidate
        if not root_dir:
            return ""
        normalized = os.path.normpath(candidate.replace("/", os.sep).replace("\\", os.sep))
        resolved = os.path.abspath(os.path.join(root_dir, normalized))
        try:
            common = os.path.commonpath([resolved, root_dir])
        except ValueError:
            return ""
        if common != root_dir:
            return ""
        return resolved if os.path.isfile(resolved) else ""

    def _send_ui_event(self, node_id: str, session: Dict[str, Any],
                       selected_preset: str, strength: float):
        if not node_id or PromptServer.instance is None:
            return

        selected_for_ui = selected_preset or session.get("selected_preset") or ""
        browser_state = session.get("browser_state") or "{}"
        if (not browser_state or browser_state == "{}") and session.get("current_path"):
            browser_state = json.dumps({"currentPath": session.get("current_path")})

        preview_image = session.get("last_preview_b64", "")

        ui_payload = {
            "base_image": [session.get("base64")],
            "root_dir": [session.get("display_dir", "")],
            "has_root": [bool(session.get("root_dir"))],
            "selected_preset": [selected_for_ui],
            "browser_state": [browser_state],
            "strength": [float(strength)],
            "preview_image": [preview_image or ""],
        }

        detail = {"output": ui_payload, "node": node_id}
        try:
            PromptServer.instance.send_sync("rc_lr_preset_session", detail)
        except Exception:
            # UI refresh failures should not break execution
            pass

    def load_preset_params(self, preset_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(preset_path)[1].lower()
        if ext == ".xmp":
            return self.parse_xmp(preset_path)
        if ext == ".lrtemplate":
            return self.parse_lrtemplate(preset_path)
        raise ValueError(f"Unsupported preset format: {ext}")

    def process_pipeline(self, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        working = img.copy()
        working = self.apply_exposure_contrast(working, params)
        working = self.apply_highlights_shadows(working, params)
        working = self.apply_tone_curve(working, params)
        working = self.apply_hsl(working, params)
        working = self.apply_split_toning(working, params)
        working = self.apply_vibrance_saturation(working, params)
        return np.clip(working, 0.0, 1.0)

    def apply_preset(self, image, preset_directory, strength=100.0, pass_forward=False,
                     selected_preset="", browser_state="{}", node_id=None):
        node_key = str(node_id or "")
        # Use first frame for preview/cache
        base_np = self._tensor_to_numpy(image)
        cache = self._cache_session(node_key, base_np, preset_directory, browser_state)

        selected_value = (selected_preset or "").strip()
        if not selected_value:
            selected_value = cache.get("selected_preset") or cache.get("selected_preview") or ""

        if selected_value:
            cache["selected_preset"] = selected_value

        self._send_ui_event(node_key, cache, selected_value, strength)

        if not selected_value:
            raise RuntimeError(MISSING_SELECTION_MESSAGE)

        preset_path = self._resolve_preset_path(cache.get("root_dir", ""), selected_value)
        if not preset_path:
            raise RuntimeError("Selected preset cannot be resolved inside the configured directory.")

        if not pass_forward:
            raise RuntimeError(PASS_FORWARD_MESSAGE)

        # Load preset parameters once
        params = self.load_preset_params(preset_path)

        # Process all frames in the batch
        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            # Convert each frame to numpy
            frame_np = image[i].detach().cpu().numpy()
            frame_np = np.clip(frame_np, 0.0, 1.0).astype(np.float32)

            # Apply preset to this frame
            processed = self.process_pipeline(frame_np, params)

            # Apply strength blending
            if strength < 100.0:
                alpha = np.clip(strength / 100.0, 0.0, 1.0)
                processed = frame_np * (1 - alpha) + processed * alpha

            # Convert back to tensor for this frame
            result_tensor = torch.from_numpy(np.clip(processed, 0.0, 1.0).astype(np.float32))
            results.append(result_tensor)

        # Stack all results to create a batch tensor
        batch_result = torch.stack(results, dim=0)
        return (batch_result,)

    def parse_xmp(self, filepath):
        """Parse Adobe Camera Raw XMP preset file."""
        tree = ET.parse(filepath)
        root = tree.getroot()

        ns = {
            'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }

        params = {}
        desc = root.find('.//rdf:Description', ns)

        if desc is not None:
            for attr, value in desc.attrib.items():
                if '}' in attr:
                    attr = attr.split('}')[1]
                try:
                    params[attr] = float(value)
                except ValueError:
                    params[attr] = value

        tone_curve = desc.find('.//crs:ToneCurve/rdf:Seq', ns) if desc is not None else None
        if tone_curve is not None:
            curve_points = []
            for li in tone_curve.findall('rdf:li', ns):
                x, y = li.text.split(',')
                curve_points.append((int(x.strip()), int(y.strip())))
            params['ToneCurve'] = curve_points

        return params

    def parse_lrtemplate(self, filepath):
        """Parse Lightroom lrtemplate preset file (Lua table format)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        params: Dict[str, Any] = {}

        import re

        pattern = r'(\w+)\s*=\s*([^,\n]+)'
        matches = re.findall(pattern, content)

        for key, value in matches:
            value = value.strip()
            if key in ['id', 'internalName', 'title', 'type', 'uuid', 'version']:
                continue

            if value == 'true':
                params[key] = True
            elif value == 'false':
                params[key] = False
            elif value.startswith('"') or value.startswith("'"):
                params[key] = value.strip('"\'')
            else:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

        curve_pattern = r'(ToneCurvePV2012\w*)\s*=\s*\{([^}]+)\}'
        curve_matches = re.findall(curve_pattern, content)

        for curve_name, curve_data in curve_matches:
            values = [float(x.strip()) for x in curve_data.split(',') if x.strip()]
            points = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
            params[curve_name] = points

        return params

    def apply_exposure_contrast(self, img, params):
        exposure = params.get('Exposure2012', params.get('Exposure', 0.0))
        if exposure != 0:
            img = img * (2 ** exposure)

        contrast = params.get('Contrast2012', params.get('Contrast', 0.0))
        if contrast != 0:
            factor = (contrast / 100.0) + 1.0
            img = (img - 0.5) * factor + 0.5

        return img

    def apply_highlights_shadows(self, img, params):
        luminance = 0.299 * img[..., 0:1] + 0.587 * img[..., 1:2] + 0.114 * img[..., 2:3]

        highlights = params.get('Highlights2012', params.get('HighlightRecovery', 0.0))
        if highlights != 0:
            mask = np.maximum(0, (luminance - 0.5) * 2)
            adjustment = highlights / 100.0
            img = img + (img * adjustment * mask)

        shadows = params.get('Shadows2012', params.get('Shadows', 0.0))
        if shadows != 0:
            mask = np.maximum(0, (0.5 - luminance) * 2)
            adjustment = shadows / 100.0
            img = img + (img * adjustment * mask)

        whites = params.get('Whites2012', 0.0)
        if whites != 0:
            mask = np.maximum(0, (luminance - 0.7) / 0.3)
            adjustment = whites / 100.0
            img = img + (img * adjustment * mask)

        blacks = params.get('Blacks2012', 0.0)
        if blacks != 0:
            mask = np.maximum(0, (0.3 - luminance) / 0.3)
            adjustment = blacks / 100.0
            img = img + (img * adjustment * mask)

        return img

    def apply_tone_curve(self, img, params):
        def build_curve(points):
            if not points or len(points) < 2:
                return None
            try:
                coords = sorted((float(p[0]), float(p[1])) for p in points)
                filtered = []
                last_x = None
                for x_val, y_val in coords:
                    if last_x is None or abs(x_val - last_x) > 1e-6:
                        filtered.append((x_val, y_val))
                        last_x = x_val
                if len(filtered) < 2:
                    return None
                x_points = np.array([p[0] for p in filtered], dtype=np.float32) / 255.0
                y_points = np.array([p[1] for p in filtered], dtype=np.float32) / 255.0
                kind = 'linear' if len(filtered) < 4 else 'cubic'
                return interp1d(
                    x_points,
                    y_points,
                    kind=kind,
                    bounds_error=False,
                    fill_value='extrapolate'
                )
            except Exception:
                return None

        master_curve = build_curve(params.get('ToneCurvePV2012'))
        if master_curve is not None:
            img_flat = img.reshape(-1, img.shape[-1])
            for channel in range(img.shape[-1]):
                img_flat[:, channel] = np.clip(master_curve(img_flat[:, channel]), 0, 1)
            img = img_flat.reshape(img.shape)

        for channel, idx in [('Red', 0), ('Green', 1), ('Blue', 2)]:
            curve_func = build_curve(params.get(f'ToneCurvePV2012{channel}'))
            if curve_func is not None and idx < img.shape[-1]:
                img[..., idx] = np.clip(curve_func(img[..., idx]), 0, 1)

        return img

    def apply_hsl(self, img, params):
        img_np = img.copy()
        max_c = np.max(img_np, axis=-1)
        min_c = np.min(img_np, axis=-1)
        diff = max_c - min_c

        hue = np.zeros_like(max_c)
        mask_r = (max_c == img_np[..., 0]) & (diff > 0)
        mask_g = (max_c == img_np[..., 1]) & (diff > 0)
        mask_b = (max_c == img_np[..., 2]) & (diff > 0)

        hue[mask_r] = 60 * (((img_np[..., 1] - img_np[..., 2]) / diff)[mask_r] % 6)
        hue[mask_g] = 60 * (((img_np[..., 2] - img_np[..., 0]) / diff)[mask_g] + 2)
        hue[mask_b] = 60 * (((img_np[..., 0] - img_np[..., 1]) / diff)[mask_b] + 4)

        saturation = np.where(max_c > 0, diff / max_c, 0)
        value = max_c

        color_ranges = {
            'Red': [(345, 360), (0, 15)],
            'Orange': [(15, 45)],
            'Yellow': [(45, 75)],
            'Green': [(75, 165)],
            'Aqua': [(165, 195)],
            'Blue': [(195, 255)],
            'Purple': [(255, 285)],
            'Magenta': [(285, 345)]
        }

        for color, ranges in color_ranges.items():
            hue_adj = params.get(f'HueAdjustment{color}', 0.0)
            sat_adj = params.get(f'SaturationAdjustment{color}', 0.0)
            lum_adj = params.get(f'LuminanceAdjustment{color}', 0.0)

            if hue_adj == 0 and sat_adj == 0 and lum_adj == 0:
                continue

            mask = np.zeros_like(hue, dtype=bool)
            for start, end in ranges:
                if start > end:
                    mask |= (hue >= start) | (hue <= end)
                else:
                    mask |= (hue >= start) & (hue <= end)

            if hue_adj != 0:
                hue[mask] = (hue[mask] + hue_adj) % 360

            if sat_adj != 0:
                saturation[mask] = np.clip(saturation[mask] * (1 + sat_adj / 100.0), 0, 1)

            if lum_adj != 0:
                value[mask] = np.clip(value[mask] * (1 + lum_adj / 100.0), 0, 1)

        c = value * saturation
        x = c * (1 - np.abs(((hue / 60) % 2) - 1))
        m = value - c

        img_out = np.zeros_like(img_np)

        mask0 = (hue >= 0) & (hue < 60)
        mask1 = (hue >= 60) & (hue < 120)
        mask2 = (hue >= 120) & (hue < 180)
        mask3 = (hue >= 180) & (hue < 240)
        mask4 = (hue >= 240) & (hue < 300)
        mask5 = (hue >= 300) & (hue < 360)

        img_out[mask0] = np.stack([c[mask0], x[mask0], np.zeros_like(c[mask0])], axis=-1)
        img_out[mask1] = np.stack([x[mask1], c[mask1], np.zeros_like(c[mask1])], axis=-1)
        img_out[mask2] = np.stack([np.zeros_like(c[mask2]), c[mask2], x[mask2]], axis=-1)
        img_out[mask3] = np.stack([np.zeros_like(c[mask3]), x[mask3], c[mask3]], axis=-1)
        img_out[mask4] = np.stack([x[mask4], np.zeros_like(c[mask4]), c[mask4]], axis=-1)
        img_out[mask5] = np.stack([c[mask5], np.zeros_like(c[mask5]), x[mask5]], axis=-1)

        img_out += m[..., np.newaxis]

        return img_out

    def apply_vibrance_saturation(self, img, params):
        vibrance = params.get('Vibrance', 0.0)
        saturation = params.get('Saturation', 0.0)

        if vibrance == 0 and saturation == 0:
            return img

        max_c = np.max(img, axis=-1, keepdims=True)
        min_c = np.min(img, axis=-1, keepdims=True)
        diff = max_c - min_c

        sat = np.where(max_c > 0, diff / max_c, 0)

        if saturation != 0:
            sat_factor = 1.0 + (saturation / 100.0)
            sat = np.clip(sat * sat_factor, 0, 1)

        if vibrance != 0:
            vib_factor = vibrance / 100.0
            vib_mask = 1.0 - sat
            sat = sat + (vib_mask * vib_factor * (1 - sat))
            sat = np.clip(sat, 0, 1)

        gray = np.mean(img, axis=-1, keepdims=True)
        img = gray + (img - gray) * (sat / (diff / (max_c + 1e-10) + 1e-10))

        return img

    def apply_split_toning(self, img, params):
        shadow_hue = params.get('SplitToningShadowHue', 0.0)
        shadow_sat = params.get('SplitToningShadowSaturation', 0.0)
        highlight_hue = params.get('SplitToningHighlightHue', 0.0)
        highlight_sat = params.get('SplitToningHighlightSaturation', 0.0)
        balance = params.get('SplitToningBalance', 0.0)

        if shadow_sat == 0 and highlight_sat == 0:
            return img

        luminance = 0.299 * img[..., 0:1] + 0.587 * img[..., 1:2] + 0.114 * img[..., 2:3]

        def hue_to_rgb(hue, sat_value):
            hue = hue % 360
            c = sat_value / 100.0
            x = c * (1 - abs((hue / 60) % 2 - 1))

            if hue < 60:
                return np.array([c, x, 0])
            if hue < 120:
                return np.array([x, c, 0])
            if hue < 180:
                return np.array([0, c, x])
            if hue < 240:
                return np.array([0, x, c])
            if hue < 300:
                return np.array([x, 0, c])
            return np.array([c, 0, x])

        if shadow_sat > 0:
            shadow_color = hue_to_rgb(shadow_hue, shadow_sat)
            shadow_mask = (1.0 - luminance) ** 2
            img = img + shadow_color * shadow_mask

        if highlight_sat > 0:
            highlight_color = hue_to_rgb(highlight_hue, highlight_sat)
            highlight_mask = luminance ** 2
            img = img + highlight_color * highlight_mask

        if balance != 0:
            balance_factor = (balance + 100.0) / 200.0
            img = img * balance_factor + np.clip(img, 0, 1) * (1 - balance_factor)

        return img


routes = PromptServer.instance.routes


def _get_session_from_request(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    node_id = str(data.get("node_id", "")).strip()
    if not node_id:
        return None
    return RC_LRPreset.SESSION_CACHE.get(node_id)


@routes.post('/rc_lr_presets/list')
async def list_presets(request):
    try:
        data = await request.json()
        session = _get_session_from_request(data)
        if not session:
            return web.json_response({"status": "error", "message": "Preset browser is not initialized."}, status=400)

        root_dir = session.get("root_dir")
        if not root_dir or not os.path.isdir(root_dir):
            return web.json_response({"status": "error", "message": "Preset directory is empty or invalid. Re-run the node after setting it."}, status=400)

        target_path = data.get("path", "")
        normalized = os.path.normpath(target_path.replace("/", os.sep).replace("\\", os.sep)) if target_path else ""
        requested = os.path.abspath(os.path.join(root_dir, normalized)) if normalized else root_dir

        try:
            common = os.path.commonpath([requested, root_dir])
        except ValueError:
            common = ""
        if common != root_dir:
            requested = root_dir

        if not os.path.isdir(requested):
            requested = root_dir

        directories = []
        presets = []

        for entry in sorted(os.scandir(requested), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.is_dir():
                directories.append({
                    "name": entry.name,
                    "relative_path": _format_relative_path(root_dir, entry.path),
                })
            elif entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in SUPPORTED_PRESET_EXT:
                    presets.append({
                        "name": entry.name,
                        "relative_path": _format_relative_path(root_dir, entry.path),
                        "extension": ext,
                    })

        breadcrumbs = []
        rel_path = _format_relative_path(root_dir, requested)
        if rel_path:
            parts = rel_path.split("/")
            acc = []
            for part in parts:
                acc.append(part)
                breadcrumbs.append({
                    "label": part,
                    "path": "/".join(acc),
                })

        session["current_path"] = rel_path
        session["browser_state"] = json.dumps({"currentPath": rel_path})

        return web.json_response({
            "status": "success",
            "directories": directories,
            "presets": presets,
            "path": rel_path,
            "breadcrumbs": breadcrumbs,
        })
    except Exception as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=500)


@routes.post('/rc_lr_presets/preview')
async def preview_preset(request):
    try:
        data = await request.json()
        session = _get_session_from_request(data)
        if not session:
            return web.json_response({"status": "error", "message": "Preset browser is not initialized."}, status=400)

        preset_rel = data.get("preset", "")
        strength = float(data.get("strength", 100.0))

        preset_path = RC_LRPreset._resolve_preset_path(session.get("root_dir", ""), preset_rel)
        if not preset_path:
            return web.json_response({"status": "error", "message": "Preset path cannot be resolved. Make sure it stays inside the configured directory."}, status=400)

        image_np = session.get("image_np")
        if image_np is None:
            return web.json_response({"status": "error", "message": "Image cache is empty. Re-run the node."}, status=400)

        instance = RC_LRPreset()
        params = instance.load_preset_params(preset_path)
        preview_np = instance.process_pipeline(image_np.copy(), params)

        if strength < 100.0:
            alpha = np.clip(strength / 100.0, 0.0, 1.0)
            preview_np = image_np * (1 - alpha) + preview_np * alpha

        preview_b64 = instance._numpy_to_base64(preview_np)

        session["last_preview"] = preview_np
        session["last_preview_b64"] = preview_b64
        session["selected_preview"] = preset_rel
        session["selected_preset"] = preset_rel

        return web.json_response({
            "status": "success",
            "preview": preview_b64,
            "preset": {
                "relative_path": _format_relative_path(session.get("root_dir", ""), preset_path),
                "name": os.path.basename(preset_path),
            }
        })
    except Exception as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=500)
