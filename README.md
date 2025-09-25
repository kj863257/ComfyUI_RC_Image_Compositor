# RC Image Compositor 🎨
## RC 图像合成器

A comprehensive ComfyUI plugin suite that brings **professional Photoshop-style layer effects** and **advanced compositing capabilities** to your workflows. Now with modular architecture, complete 24 blend modes, and enhanced positioning system!

一套全面的 ComfyUI 插件套装，为您的工作流程带来 **专业的 Photoshop 风格图层效果** 和 **高级合成功能**。现在采用模块化架构，完整24种混合模式，增强的定位系统！

---

## ✨ Key Features | 核心特性

### 🎭 **Professional Layer Styles | 专业图层样式**
- **Drop Shadow | 投影** - Photoshop-compatible drop shadows with blur, offset, and color controls
- **Stroke | 描边** - Inside/outside/center stroke positioning with customizable colors and auto-canvas expansion
- **Outer Glow | 外发光** - Soft outer glow effects with spread, color options, and auto-canvas expansion

### 🎨 **Complete Blend Mode Suite | 完整混合模式套装**
- **24 Professional Blend Modes** including all HSL modes (Hue, Saturation, Color, Luminosity)
- **Enhanced positioning system** - No more negative offset issues! Use alignment options for precise control
- **Detailed tooltips** - Every blend mode explained with visual descriptions in both languages

### 🔧 **Professional Filters & Adjustments | 专业滤镜和调整**
- **Gaussian Blur | 高斯模糊** - Professional-grade blurring with PIL/OpenCV options
- **Unsharp Mask Sharpening | 反锐化蒙版** - Multiple sharpening algorithms
- **Hue/Saturation Adjustment | 色相/饱和度调整** - Targeted color editing like Photoshop
- **Opacity Control | 透明度控制** - Precise opacity adjustment with alpha channel support
- **Levels Adjustment | 色阶调整** - Input/output levels with gamma correction
- **Brightness/Contrast | 亮度/对比度** - Professional brightness and contrast controls
- **Color Balance | 色彩平衡** - CMY color balance with tone range selection
- **Channel Mixer | 通道混合器** - Advanced RGB channel mixing with monochrome option

### 🛠️ **Utility Tools | 实用工具**
- **Canvas Padding | 画布填充** - Multiple fill modes (solid, edge, mirror, transparent)
- **Image Scale | 图像缩放** - 6 scaling methods with high-quality resampling
- **Image Crop | 图像裁剪** - Manual, center, and aspect ratio cropping
- **Canvas Resize | 画布调整** - 9 anchor positions with background color control

---

## 🎯 Perfect For | 完美适用于

- **UI Design Workflows** - Logo placement, watermarks, interface elements
- **Text Effects** - Professional typography with shadows, strokes, and glows
- **Multi-Image Compositing** - Complex layer compositions with precise control
- **Photo Enhancement** - Color correction, sharpening, and artistic effects
- **Professional Design** - Complete Photoshop-style layer workflows

- **UI 设计工作流** - Logo 放置、水印、界面元素
- **文字特效** - 带阴影、描边和发光的专业排版
- **多图合成** - 精确控制的复杂图层合成
- **照片增强** - 色彩校正、锐化和艺术效果
- **专业设计** - 完整的 Photoshop 风格图层工作流

---

## 📦 Node Categories | 节点分类

### **RC/Image** - Core Compositing | 核心合成
- `RC 图像合成器 (完整版) | RC Image Compositor (Complete)` - Complete compositor with 24 blend modes and enhanced positioning
- `RC 加载透明图像 | RC Load Image (Alpha)` - RGBA image loading with alpha preservation

### **RC/Layer Effects** - Photoshop Layer Styles | Photoshop 图层样式
- `RC 投影效果 | RC Drop Shadow` - Professional drop shadows with auto-canvas expansion
- `RC 描边效果 | RC Stroke` - Inside/outside/center strokes with auto-canvas expansion
- `RC 外发光效果 | RC Outer Glow` - Soft outer glow effects with auto-canvas expansion

### **RC/Filters** - Image Processing | 图像处理
- `RC 高斯模糊 | RC Gaussian Blur` - Professional blurring with algorithm selection
- `RC 锐化滤镜 | RC Sharpen` - Multiple sharpening methods including unsharp mask

### **RC/Adjustments** - Color & Tone | 色彩和色调
- `RC 色相/饱和度 | RC Hue/Saturation` - Targeted color adjustment with colorize mode
- `RC 透明度调整 | RC Opacity Adjust` - Precise opacity control with alpha channel support
- `RC 色阶调整 | RC Levels` - Professional levels adjustment with gamma correction
- `RC 亮度/对比度 | RC Brightness/Contrast` - Dual-algorithm brightness and contrast
- `RC 色彩平衡 | RC Color Balance` - CMY color balance with tone range targeting
- `RC 通道混合器 | RC Channel Mixer` - Advanced RGB channel mixing with monochrome support

### **RC/Utilities** - Canvas & Transform | 画布和变换
- `RC 画布填充 | RC Canvas Padding` - Add padding with multiple fill modes
- `RC 图像缩放 | RC Image Scale` - Professional scaling with 6 different methods
- `RC 图像裁剪 | RC Image Crop` - Flexible cropping with manual, center, and ratio modes
- `RC 画布调整 | RC Canvas Resize` - Resize canvas with 9 anchor positions

---

## 🚀 Installation | 安装方法

### Method 1: ComfyUI Manager (Recommended) | 方法1：ComfyUI管理器（推荐）
1. Open **ComfyUI Manager** | 打开 **ComfyUI Manager**
2. Go to **Install Custom Nodes** | 进入 **安装自定义节点**
3. Search for **"RC Image Compositor"** | 搜索 **"RC Image Compositor"**
4. Click **Install** | 点击 **安装**

### Method 2: Manual Install | 方法2：手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kj863257/ComfyUI_RC_Image_Compositor
cd ComfyUI_RC_Image_Compositor
pip install -r requirements.txt
```
Then restart ComfyUI | 然后重启 ComfyUI

---

## 🎨 Enhanced Positioning System | 增强的定位系统

### **No More Negative Offset Issues! | 再也不用担心负数偏移！**

**Old Problem:** Want to place image at right edge? Can't use "-0" offset.

**New Solution:** Use alignment system!

- **Right edge, tight fit**: `x_percent=100`, `x_align=from_right`, `x_offset=0`
- **Left edge, tight fit**: `x_percent=0`, `x_align=from_left`, `x_offset=0`
- **Bottom edge, tight fit**: `y_percent=100`, `y_align=from_bottom`, `y_offset=0`

### **Usage Examples | 使用示例**
1. **Right-top corner with 10px margin**:
   - `x_percent=100`, `x_align=from_right`, `x_offset=10`
   - `y_percent=0`, `y_align=from_top`, `y_offset=10`

2. **Left-bottom corner with 20px margin**:
   - `x_percent=0`, `x_align=from_left`, `x_offset=20`
   - `y_percent=100`, `y_align=from_bottom`, `y_offset=20`

---

## 🔧 Advanced Features | 高级功能

### **Complete Blend Mode Compatibility | 完整混合模式兼容性**
All 24 blend modes are mathematically identical to Photoshop's implementation with detailed tooltips:

**🌟 Basic Modes | 基本模式:**
- Normal - Direct overlay

**🌑 Darken Modes | 变暗模式:**
- Darken, Multiply, Color Burn, Linear Burn

**🌕 Lighten Modes | 变亮模式:**
- Lighten, Screen, Color Dodge, Linear Dodge

**⚡ Contrast Modes | 对比模式:**
- Overlay, Soft Light, Hard Light, Vivid Light, Linear Light, Pin Light, Hard Mix

**🔄 Comparative Modes | 比较模式:**
- Difference, Exclusion, Subtract, Divide

**🎨 HSL Modes | HSL 模式:**
- Hue, Saturation, Color, Luminosity

### **Auto-Canvas Expansion | 自动画布扩展**
Layer effects automatically expand canvas when effects exceed original bounds:
- Drop Shadow: Expands to accommodate shadow distance and blur
- Stroke: Expands for outside and center strokes
- Outer Glow: Expands for glow size and spread

### **Professional Color Editing | 专业色彩编辑**
- Target specific color ranges (Reds, Blues, etc.) in Hue/Saturation
- Preserve luminosity in Color Balance adjustments
- Channel-specific Levels adjustments (RGB, Red, Green, Blue)
- Monochrome conversion with custom channel mixing

---

## 🌟 Why Choose RC Image Compositor? | 为什么选择 RC 图像合成器？

- **🎯 Photoshop Accuracy**: Mathematically identical blend modes and effects
- **🚫 No Negative Offset Issues**: Enhanced positioning system solves alignment problems
- **🌐 Bilingual Support**: Complete Chinese/English interface and documentation
- **🔧 Professional Grade**: Built for production workflows with auto-canvas expansion
- **📱 User Friendly**: Detailed tooltips explain every parameter and blend mode
- **🚀 High Performance**: Optimized algorithms with multiple implementation choices
- **🛠️ Extensible**: Clean modular architecture for easy customization

- **🎯 Photoshop 精度**：数学上完全相同的混合模式和效果
- **🚫 无负数偏移问题**：增强的定位系统解决对齐问题
- **🌐 双语支持**：完整的中英文界面和文档
- **🔧 专业级别**：专为生产工作流而构建，支持自动画布扩展
- **📱 用户友好**：详细的提示说明每个参数和混合模式
- **🚀 高性能**：优化的算法，提供多种实现选择
- **🛠️ 可扩展**：清晰的模块化架构便于自定义

---

## 📄 License | 许可证

This project is licensed under the MIT License. | 本项目采用 MIT 许可证。

---

*Crafted with ❤️ for the ComfyUI community | 为 ComfyUI 社区精心打造 ❤️*

> **"Complete Photoshop power in ComfyUI - 18 professional nodes, 24 blend modes, zero negative offset headaches."**
>
> **"ComfyUI 中的完整 Photoshop 功能 - 18个专业节点，24种混合模式。"**