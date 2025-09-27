import { app } from "../../scripts/app.js";

// Translation support
const TRANSLATIONS = {
    Preset: { en: "Preset", zh: "预设" },
    PresetTooltip: {
        en: "Select a preset to apply predefined settings",
        zh: "选择预设以应用预定义设置"
    }
};

const translate = (key, fallback) => {
    let lang = "en";
    try {
        const stored = localStorage.getItem("Comfy.Settings.Locale");
        if (stored) {
            lang = stored.toLowerCase();
        } else if (navigator?.language) {
            lang = navigator.language.toLowerCase();
        }
    } catch (_) {
        lang = "en";
    }
    // Match language prefix (zh-CN, zh-TW -> zh)
    if (lang.startsWith("zh")) {
        lang = "zh";
    }
    return TRANSLATIONS[key]?.[lang] || fallback;
};

// Hue/Saturation presets for RC_HueSaturation
const HSL_PRESETS = {
    "Cyanotype": {
        "hue_shift": 210,
        "saturation": -50,
        "lightness": 0,
        "colorize": true,
        "colorize_hue": 210,
        "colorize_saturation": 25,
        "_description": { en: "Blue-tinted vintage photography effect", zh: "氰版法（蓝调古典摄影效果）" }
    },
    "Old Style": {
        "hue_shift": 30,
        "saturation": -25,
        "lightness": -10,
        "colorize": false,
        "colorize_hue": 0,
        "colorize_saturation": 50,
        "_description": { en: "Warm vintage color grading", zh: "老式风格（暖色调复古效果）" }
    },
    "Red Boost": {
        "hue_shift": 0,
        "saturation": 25,
        "lightness": 0,
        "colorize": false,
        "colorize_hue": 0,
        "colorize_saturation": 50,
        "_description": { en: "Enhanced red channel saturation", zh: "红色增强（提升红色通道饱和度）" }
    },
    "Increase Saturation": {
        "hue_shift": 0,
        "saturation": 20,
        "lightness": 0,
        "colorize": false,
        "colorize_hue": 0,
        "colorize_saturation": 50,
        "_description": { en: "Moderate saturation boost", zh: "饱和度增强（适度提升）" }
    },
    "Increase Saturation More": {
        "hue_shift": 0,
        "saturation": 40,
        "lightness": 0,
        "colorize": false,
        "colorize_hue": 0,
        "colorize_saturation": 50,
        "_description": { en: "Strong saturation enhancement", zh: "饱和度强化（大幅提升）" }
    },
    "Sepia": {
        "hue_shift": 35,
        "saturation": -20,
        "lightness": 5,
        "colorize": true,
        "colorize_hue": 35,
        "colorize_saturation": 35,
        "_description": { en: "Classic sepia tone effect", zh: "棕褐色调（经典复古色调）" }
    },
    "Desaturate": {
        "hue_shift": 0,
        "saturation": -50,
        "lightness": 0,
        "colorize": false,
        "colorize_hue": 0,
        "colorize_saturation": 50,
        "_description": { en: "Reduced color saturation", zh: "去饱和（降低色彩饱和度）" }
    },
    "Vibrance": {
        "hue_shift": 0,
        "saturation": 30,
        "lightness": 5,
        "colorize": false,
        "colorize_hue": 0,
        "colorize_saturation": 50,
        "_description": { en: "Enhanced color vibrancy", zh: "鲜艳度（增强色彩活力）" }
    }
};


function addPresetControls(nodeType, nodeName, presets, targetParam) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);

        // Get current language
        let lang = "en";
        try {
            const stored = localStorage.getItem("Comfy.Settings.Locale");
            if (stored) {
                lang = stored.toLowerCase();
            } else if (navigator?.language) {
                lang = navigator.language.toLowerCase();
            }
        } catch (_) {}
        if (lang.startsWith("zh")) lang = "zh";

        // Generate tooltip with all preset descriptions
        let tooltipLines = [];
        tooltipLines.push(lang === "zh" ? "预设选项：" : "Presets:");
        tooltipLines.push(lang === "zh" ? "• Default - 默认值（无调整）" : "• Default - No adjustment");
        for (const [name, preset] of Object.entries(presets)) {
            if (preset._description) {
                const desc = preset._description[lang] || preset._description.en || name;
                tooltipLines.push(`• ${name} - ${desc}`);
            } else {
                tooltipLines.push(`• ${name}`);
            }
        }
        const tooltipText = tooltipLines.join("\n");

        // Add preset dropdown
        const presetWidget = this.addWidget("combo", translate("Preset", "Preset"), "Default", (value) => {
            if (value !== "Default" && presets[value]) {
                const preset = presets[value];

                // Apply preset values to widgets (skip _description)
                for (const [paramName, paramValue] of Object.entries(preset)) {
                    if (paramName.startsWith("_")) continue; // Skip metadata
                    const widget = this.widgets.find(w => w.name === paramName);
                    if (widget) {
                        widget.value = paramValue;
                    }
                }

                // Trigger update
                if (this.onWidgetChanged) {
                    this.onWidgetChanged("preset", value, null);
                }
            }
        }, {
            values: ["Default", ...Object.keys(presets)]
        });

        // Add tooltip to preset widget
        if (presetWidget) {
            presetWidget.tooltip = tooltipText;
        }

        return r;
    };
}

// Register presets for different node types
app.registerExtension({
    name: "RC.Presets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RC_HueSaturation") {
            addPresetControls(nodeType, nodeData.name, HSL_PRESETS, null);
        } else if (nodeData.name === "RC_ChannelMixer") {
            const CHANNEL_MIXER_PRESETS = {
                "B&W Red Filter": {
                    "red_from_red": 85, "red_from_green": 15, "red_from_blue": 0,
                    "green_from_red": 85, "green_from_green": 15, "green_from_blue": 0,
                    "blue_from_red": 85, "blue_from_green": 15, "blue_from_blue": 0,
                    "monochrome": true,
                    "_description": { en: "B&W with red filter emphasis (dramatic skies)", zh: "黑白红色滤镜（戏剧性天空效果）" }
                },
                "B&W Orange Filter": {
                    "red_from_red": 70, "red_from_green": 30, "red_from_blue": 0,
                    "green_from_red": 70, "green_from_green": 30, "green_from_blue": 0,
                    "blue_from_red": 70, "blue_from_green": 30, "blue_from_blue": 0,
                    "monochrome": true,
                    "_description": { en: "B&W with orange filter (balanced portraits)", zh: "黑白橙色滤镜（平衡人像效果）" }
                },
                "B&W Yellow Filter": {
                    "red_from_red": 60, "red_from_green": 40, "red_from_blue": 0,
                    "green_from_red": 60, "green_from_green": 40, "green_from_blue": 0,
                    "blue_from_red": 60, "blue_from_green": 40, "blue_from_blue": 0,
                    "monochrome": true,
                    "_description": { en: "B&W with yellow filter (natural skin tones)", zh: "黑白黄色滤镜（自然肤色效果）" }
                },
                "B&W Green Filter": {
                    "red_from_red": 40, "red_from_green": 60, "red_from_blue": 0,
                    "green_from_red": 40, "green_from_green": 60, "green_from_blue": 0,
                    "blue_from_red": 40, "blue_from_green": 60, "blue_from_blue": 0,
                    "monochrome": true,
                    "_description": { en: "B&W with green filter (smooth skin, bright foliage)", zh: "黑白绿色滤镜（光滑肌肤，明亮植被）" }
                },
                "B&W Blue Filter": {
                    "red_from_red": 0, "red_from_green": 25, "red_from_blue": 75,
                    "green_from_red": 0, "green_from_green": 25, "green_from_blue": 75,
                    "blue_from_red": 0, "blue_from_green": 25, "blue_from_blue": 75,
                    "monochrome": true,
                    "_description": { en: "B&W with blue filter (enhanced haze and atmosphere)", zh: "黑白蓝色滤镜（增强雾霾和氛围）" }
                },
                "B&W Infrared": {
                    "red_from_red": 80, "red_from_green": 20, "red_from_blue": 0,
                    "green_from_red": 80, "green_from_green": 20, "green_from_blue": 0,
                    "blue_from_red": 80, "blue_from_green": 20, "blue_from_blue": 0,
                    "monochrome": true,
                    "_description": { en: "B&W infrared simulation (bright foliage, dark skies)", zh: "黑白红外模拟（明亮植被，黑暗天空）" }
                }
            };
            addPresetControls(nodeType, nodeData.name, CHANNEL_MIXER_PRESETS, null);
        } else if (nodeData.name === "RC_LevelsAdjust") {
            const LEVELS_PRESETS = {
                "Increase Contrast": {
                    "input_black": 0.078, "input_white": 0.922, "gamma": 1.0,
                    "output_black": 0.0, "output_white": 1.0,
                    "_description": { en: "Moderate contrast enhancement (clip shadows/highlights)", zh: "适度对比度增强（剪裁阴影/高光）" }
                },
                "Lighten": {
                    "input_black": 0.0, "input_white": 0.784, "gamma": 0.8,
                    "output_black": 0.0, "output_white": 1.0,
                    "_description": { en: "Brighten image (compress highlights, gamma boost)", zh: "提亮图像（压缩高光，伽马提升）" }
                },
                "Darken": {
                    "input_black": 0.216, "input_white": 1.0, "gamma": 1.2,
                    "output_black": 0.0, "output_white": 1.0,
                    "_description": { en: "Darken image (expand shadows, gamma reduction)", zh: "压暗图像（扩展阴影，伽马降低）" }
                },
                "High Contrast": {
                    "input_black": 0.157, "input_white": 0.843, "gamma": 1.0,
                    "output_black": 0.0, "output_white": 1.0,
                    "_description": { en: "Strong contrast enhancement (aggressive clipping)", zh: "强烈对比度增强（激进剪裁）" }
                }
            };
            addPresetControls(nodeType, nodeData.name, LEVELS_PRESETS, null);
        }
    }
});