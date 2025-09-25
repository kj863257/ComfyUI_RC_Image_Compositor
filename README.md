# RC Image Compositor  
## RC 图像合成器

A ComfyUI node that composites an overlay image onto a background with **Photoshop-compatible blend modes**, flexible positioning, scaling, rotation, and opacity control.  
一款 ComfyUI 节点，支持将贴图合成到背景上，并提供 **Photoshop 兼容的混合模式**、灵活的定位、缩放、旋转和透明度控制。

---

## ✨ Features / 功能特点

- **Photoshop Blend Modes**  
  Supports 14 standard blend modes in Photoshop order:  
  `正常`, `变暗`, `正片叠底`, `颜色加深`, `线性加深`, `变亮`, `滤色`, `颜色减淡`, `线性减淡（添加）`, `叠加`, `柔光`, `强光`, `差值`, `排除`  
  支持 14 种标准 Photoshop 混合模式（按官方顺序排列）

- **Precise Positioning**  
  Percentage-based (0–100%) + pixel offset (supports negative values)  
  基于百分比（0–100%）定位 + 像素偏移（支持负值）

- **Flexible Scaling**  
  - `relative_to_overlay`: Scale by original overlay size  
  - `relative_to_background_width`: Overlay width = scale × background width  
  - `relative_to_background_height`: Overlay height = scale × background height  
  三种缩放模式：按贴图自身尺寸、按背景宽度比例、按背景高度比例

- **High-Quality Transformations**  
  Rotation (±180°, BICUBIC), horizontal/vertical flip, and Lanczos resizing  
  高质量旋转（±180°，双三次插值）、翻转、Lanczos 缩放

- **Alpha Channel Support**  
  Automatically handles RGBA overlays; falls back to RGB if no alpha  
  自动识别透明通道（RGBA），无透明通道时按 RGB 处理

- **Opacity Control**  
  Global transparency from 0.0 (fully transparent) to 1.0 (fully opaque)  
  全局透明度控制（0.0 = 完全透明，1.0 = 完全不透明）

---

## 🎯 Default Behavior / 默认行为

By default, the overlay is positioned at the **top-right corner** with a slight inset:  
默认情况下，贴图位于**右上角**并内缩一定距离：

- `x_percent = 100` → right aligned | 右对齐  
- `y_percent = 0` → top aligned | 顶部对齐  
- `x_offset = -50` → move 50px left | 向左偏移 50 像素  
- `y_offset = 50` → move 50px down | 向下偏移 50 像素  
- `scale_mode = relative_to_background_width`  
- `scale = 0.3` → overlay width = 30% of background width | 贴图宽度为背景的 30%  
- `opacity = 0.7` → 70% opaque | 70% 不透明度  

👉 Perfect for **logos, watermarks, or UI badges**!  
👉 非常适合添加 **Logo、水印或角标**！

---

## 🛠️ Usage / 使用方法

1. Connect a background image to **`background`**  
   将背景图像连接到 **`background`** 输入端口

2. Connect an overlay (with or without alpha) to **`overlay`**  
   将贴图（带或不带透明通道）连接到 **`overlay`** 输入端口
   
> 💡 For transparent PNGs, use the RC_LoadImageWithAlpha node to ensure alpha channel is preserved 
> 💡 推荐使用 RC_LoadImageWithAlpha 节点加载透明 PNG，以确保 alpha 通道正确传递

3. Adjust parameters as needed  
   根据需求调整参数

4. The output is the composited image  
   输出即为合成结果

> 💡 **Tip / 提示**  
> To place an overlay at the bottom-left: set `x_percent=0`, `y_percent=100`, `x_offset=50`, `y_offset=-50`.  
> 若要将贴图放在左下角：设置 `x_percent=0`, `y_percent=100`, `x_offset=50`, `y_offset=-50`。

---

## 📦 Installation / 安装方法

### Method 1: Manual Install / 手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/comfyui-rc-image-compositor.git
```
Then restart ComfyUI.
然后重启 ComfyUI。

- ### Method 2: ComfyUI Manager (Recommended) / ComfyUI Manager（推荐）

1. Open **ComfyUI Manager**
   打开 **ComfyUI Manager**
2. Go to **Custom Nodes Manager** → **Install from URL**
   进入 **自定义节点管理器** → **从 URL 安装**
3. Paste the GitHub URL
   粘贴 GitHub 仓库地址
4. Click **Install**
   点击 **安装**

Search for **"RC 图像合成器"** or **"RC Image Compositor"** in the node menu.
在节点菜单中搜索 **"RC 图像合成器"** 或 **"RC Image Compositor"** 即可使用。
- ## 📁 Project Structure / 项目结构

```bash
comfyui-rc-image-compositor/
├── __init__.py                 # Plugin entry point / 插件入口
├── rc_image_compositor.py      # Node implementation / 节点实现
├── README.md                   # This file / 本文件
└── pyproject.toml              # Metadata for ComfyUI Manager / ComfyUI Manager 元数据
```

No external dependencies — uses only PyTorch, NumPy, and PIL (all included in ComfyUI).
无额外依赖 — 仅使用 PyTorch、NumPy 和 PIL（ComfyUI 已自带）。

> Designed for professional image composition workflows in ComfyUI.
> 专为 ComfyUI 中的专业图像合成工作流设计。
