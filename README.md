# RC Image Compositor 🎨
## RC 图像合成器

A comprehensive ComfyUI plugin suite that brings **professional Photoshop-style layer effects** and **advanced compositing capabilities** to your workflows. Now with modular architecture, complete 24 blend modes, and enhanced positioning system!

一套全面的 ComfyUI 插件套件，为您的工作流程带来 **专业的 Photoshop 风格图层效果** 和 **高级合成功能**。现在采用模块化架构，支持完整的 24 种混合模式，并配备增强的定位系统！

---

## ✨ Key Features | 核心特性

### 🎭 **Professional Layer Styles | 专业图层样式**
- **Drop Shadow | 投影** - 兼容 Photoshop 的投影效果，支持模糊、偏移和颜色控制
- **Stroke | 描边** - 内/外/居中描边定位，支持自定义颜色和自动画布扩展
- **Outer Glow | 外发光** - 柔和外发光效果，支持扩展、颜色选项和自动画布扩展

### 🎨 **Complete Blend Mode Suite | 完整混合模式套件**
- **24 专业混合模式** 包括全部 HSL 模式（色相、饱和度、颜色、明度）
- **增强定位系统** - 使用对齐选项实现精确定位控制
- **详细工具提示** - 每种混合模式都配有视觉描述，支持中英双语

### 🔧 **Professional Filters & Adjustments | 专业滤镜和调整**
- **Gaussian Blur | 高斯模糊** - 专业级模糊效果，支持 PIL/OpenCV 算法选择
- **Unsharp Mask Sharpening | 反锐化蒙版锐化** - 多种锐化算法
- **Hue/Saturation Adjustment | 色相/饱和度调整** - 像 Photoshop 一样的目标色彩编辑
- **Opacity Control | 透明度控制** - 支持 Alpha 通道的精确透明度调整
- **Levels Adjustment | 色阶调整** - 支持伽马校正的输入/输出色阶
- **Brightness/Contrast | 亮度/对比度** - 专业级亮度和对比度控制
- **Color Balance | 色彩平衡** - 支持色调范围选择的 CMY 色彩平衡
- **Channel Mixer | 通道混合器** - 高级 RGB 通道混合，支持单色选项

### 🛠️ **Utility Tools | 实用工具**
- **Canvas Padding | 画布填充** - 多种填充模式（纯色、边缘、镜像、透明）
- **Image Scale | 图像缩放** - 6 种缩放方法，支持高质量重采样
- **Image Crop | 图像裁剪** - 支持手动、中心和宽高比裁剪
- **Canvas Resize | 画布调整** - 9 个锚点位置，支持背景色控制

---

## 🎯 Perfect For | 适用场景

- **UI Design Workflows** - Logo placement, watermarks, interface elements
- **Text Effects** - Professional typography with shadows, strokes, and glows
- **Multi-Image Compositing** - Complex layer compositions with precise control
- **Photo Enhancement** - Color correction, sharpening, and artistic effects
- **Professional Design** - Complete Photoshop-style layer workflows

- **UI 设计工作流** - Logo 放置、水印、界面元素
- **文字特效** - 带阴影、描边和发光的专业排版效果
- **多图合成** - 带有精确控制的复杂图层合成
- **照片增强** - 色彩校正、锐化和艺术效果
- **专业设计** - 完整的 Photoshop 风格图层工作流

---

## 📦 Node Categories | 节点分类

### **RC/Image** - Core Compositing | 核心合成
- `RC 图像合成器 (完整版) | RC Image Compositor (Complete)` - 支持 24 种混合模式和增强定位的完整合成器
- `RC 加载透明图像 | RC Load Image (Alpha)` - 完整保留 Alpha 通道的 RGBA 图像加载

### **RC/Layer Effects** - Photoshop Layer Styles | Photoshop 图层样式
- `RC 投影效果 | RC Drop Shadow` - 支持自动画布扩展的专业投影效果
- `RC 描边效果 | RC Stroke` - 支持自动画布扩展的内/外/居中描边
- `RC 外发光效果 | RC Outer Glow` - 支持自动画布扩展的柔和外发光效果

### **RC/Filters** - Image Processing | 图像处理
- `RC 高斯模糊 | RC Gaussian Blur` - 支持算法选择的专业级模糊
- `RC 锐化滤镜 | RC Sharpen` - 包括反锐化蒙版在内的多种锐化方法

### **RC/Adjustments** - Color & Tone | 色彩和色调
- `RC 色相/饱和度 | RC Hue/Saturation` - 带着色模式的目标色彩调整
- `RC 透明度调整 | RC Opacity Adjust` - 支持 Alpha 通道的精确透明度控制
- `RC 色阶调整 | RC Levels` - 支持伽马校正的专业色阶调整
- `RC 亮度/对比度 | RC Brightness/Contrast` - 双算法亮度和对比度控制
- `RC 色彩平衡 | RC Color Balance` - 支持色调范围定位的 CMY 色彩平衡
- `RC 通道混合器 | RC Channel Mixer` - 支持单色模式的高级 RGB 通道混合

### **RC/Utilities** - Canvas & Transform | 画布和变换
- `RC 画布填充 | RC Canvas Padding` - 支持多种填充模式的画布扩展
- `RC 图像缩放 | RC Image Scale` - 支持 6 种不同方法的专业缩放
- `RC 图像裁剪 | RC Image Crop` - 支持手动、中心和比例模式的灵活裁剪
- `RC 画布调整 | RC Canvas Resize` - 支持 9 个锚点的画布调整

---

## 🚀 Installation | 安装方法

### Method 1: ComfyUI Manager (Recommended) | 方法 1：ComfyUI 管理器（推荐）
1. Open **ComfyUI Manager** | 打开 **ComfyUI Manager**
2. Go to **Install Custom Nodes** | 进入 **安装自定义节点**
3. Search for **"RC Image Compositor"** | 搜索 **"RC Image Compositor"**
4. Click **Install** | 点击 **安装**

### Method 2: Manual Install | 方法 2：手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kj863257/ComfyUI_RC_Image_Compositor
cd ComfyUI_RC_Image_Compositor
pip install -r requirements.txt
```
Then restart ComfyUI | 然后重启 ComfyUI

---

## 🎨 Enhanced Positioning System | 增强定位系统

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
- Normal（正常） - 直接覆盖

**🌑 Darken Modes | 变暗模式:**
- Darken（变暗）, Multiply（正片叠底）, Color Burn（颜色加深）, Linear Burn（线性加深）

**🌕 Lighten Modes | 变亮模式:**
- Lighten（变亮）, Screen（滤色）, Color Dodge（颜色减淡）, Linear Dodge（线性减淡）

**⚡ Contrast Modes | 对比模式:**
- Overlay（叠加）, Soft Light（柔光）, Hard Light（强光）, Vivid Light（亮光）, Linear Light（线性光）, Pin Light（点光）, Hard Mix（实色混合）

**🔄 Comparative Modes | 比较模式:**
- Difference（差值）, Exclusion（排除）, Subtract（减去）, Divide（划分）

**🎨 HSL Modes | HSL 模式:**
- Hue（色相）, Saturation（饱和度）, Color（颜色）, Luminosity（明度）

### **Auto-Canvas Expansion | 自动画布扩展**
图层效果会在效果超出原始边界时自动扩展画布：
- 投影：为投影距离和模糊扩展画布
- 描边：为外部和居中描边扩展画布
- 外发光：为发光大小和扩展扩展画布

### **Professional Color Editing | 专业色彩编辑**
- 在色相/饱和度中针对特定色彩范围（红、蓝等）进行调整
- 在色彩平衡调整中保留明度
- 通道特定的色阶调整（RGB、红、绿、蓝）
- 带自定义通道混合的单色转换

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
- **🚫 对齐问题**：增强的定位系统解决对齐问题
- **🌐 双语支持**：完整的中英双语界面和文档
- **🔧 专业级别**：专为生产工作流构建，支持自动画布扩展
- **📱 用户友好**：详细的工具提示解释每个参数和混合模式
- **🚀 高性能**：优化的算法，提供多种实现选择
- **🛠️ 可扩展**：简洁的模块化架构，便于自定义

---

## 📄 License | 许可证

This project is licensed under the MIT License. | 本项目采用 MIT 许可证。

---

*Crafted with ❤️ for the ComfyUI community | 为 ComfyUI 社区精心打造 ❤️*

> **"Complete Photoshop power in ComfyUI - 18 professional nodes, 24 blend modes, zero negative offset headaches."**
>
> **"在 ComfyUI 中实现完整的 Photoshop 功能 - 18 个专业节点，24 种混合模式"**