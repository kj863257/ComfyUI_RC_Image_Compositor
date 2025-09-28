// RC Gradient Common Components
// 公共渐变组件，用于渐变生成器和渐变映射

// CSS样式
export const gradientStyles = `
.rc-gradient-editor {
    background: #1e1e1e;
    padding: 8px;
    border-radius: 4px;
    margin: 2px 0;
    box-sizing: border-box;
}

.rc-gradient-bar-container {
    position: relative;
    width: 100%;
    height: 28px;
    background: #333;
    border-radius: 3px;
    margin: 8px 0;
    cursor: crosshair;
    background-image:
        linear-gradient(45deg, #ccc 25%, transparent 25%, transparent 75%, #ccc 75%),
        linear-gradient(45deg, #ccc 25%, #eee 25%, #eee 75%, #ccc 75%);
    background-size: 12px 12px;
    background-position: 0 0, 6px 6px;
}

.rc-gradient-bar {
    width: 100%;
    height: 100%;
    border-radius: 3px;
}

.rc-gradient-stop {
    position: absolute;
    width: 14px;
    height: 14px;
    border: 2px solid white;
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    cursor: grab;
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
    z-index: 10;
}

.rc-gradient-stop:hover {
    border-color: #4a9eff;
    transform: translate(-50%, -50%) scale(1.1);
}

.rc-gradient-stop.selected {
    border-color: #4a9eff;
    box-shadow: 0 0 8px rgba(74, 158, 255, 0.6);
}

.rc-control-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
    font-size: 11px;
}

.rc-control-label {
    min-width: 50px;
    color: #ddd;
    font-size: 11px;
}

.rc-control-input {
    background: #333;
    border: 1px solid #555;
    border-radius: 2px;
    color: #ddd;
    font-size: 11px;
}

.rc-control-slider {
    flex: 2;
    height: 18px;
}

.rc-control-number {
    width: 60px;
    padding: 2px 4px;
    text-align: right;
}

.rc-color-picker {
    width: 35px;
    height: 22px;
    border: 1px solid #444;
    border-radius: 3px;
    cursor: pointer;
    padding: 0;
    flex-shrink: 0;
}

.rc-btn {
    background: #4a9eff;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
    cursor: pointer;
    font-size: 11px;
    margin: 0;
    flex: 1;
}

.rc-btn:hover {
    background: #3a8eef;
}

.rc-btn.danger {
    background: #ff4a4a;
}

.rc-btn.danger:hover {
    background: #ef3a3a;
}

.rc-presets {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    margin-top: 6px;
}

.rc-preset {
    width: 45px;
    height: 18px;
    border: 1px solid #444;
    border-radius: 3px;
    cursor: pointer;
    flex-shrink: 0;
}

.rc-preset:hover {
    border-color: #4a9eff;
}
`;

// 预设渐变 - 针对渐变映射优化的常用预设
export const gradientPresets = [
    {
        name: "Black to White",
        stops: [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 1.0, color: [255, 255, 255, 255] }
        ]
    },
    {
        name: "Sepia Tone",
        stops: [
            { position: 0.0, color: [45, 31, 18, 255] },
            { position: 0.5, color: [150, 120, 82, 255] },
            { position: 1.0, color: [240, 220, 180, 255] }
        ]
    },
    {
        name: "Cool Blue",
        stops: [
            { position: 0.0, color: [20, 30, 50, 255] },
            { position: 0.5, color: [80, 120, 160, 255] },
            { position: 1.0, color: [180, 210, 240, 255] }
        ]
    },
    {
        name: "Infrared",
        stops: [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 0.3, color: [80, 0, 80, 255] },
            { position: 0.7, color: [255, 50, 50, 255] },
            { position: 1.0, color: [255, 255, 255, 255] }
        ]
    },
    {
        name: "Cross Process",
        stops: [
            { position: 0.0, color: [30, 50, 70, 255] },
            { position: 0.3, color: [120, 100, 60, 255] },
            { position: 0.7, color: [200, 160, 80, 255] },
            { position: 1.0, color: [255, 230, 150, 255] }
        ]
    },
    {
        name: "Thermal",
        stops: [
            { position: 0.0, color: [0, 0, 100, 255] },
            { position: 0.25, color: [0, 100, 255, 255] },
            { position: 0.5, color: [0, 255, 0, 255] },
            { position: 0.75, color: [255, 255, 0, 255] },
            { position: 1.0, color: [255, 0, 0, 255] }
        ]
    },
    {
        name: "Film Noir",
        stops: [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 0.6, color: [40, 35, 30, 255] },
            { position: 0.9, color: [120, 115, 110, 255] },
            { position: 1.0, color: [255, 255, 255, 255] }
        ]
    },
    {
        name: "Forest Green",
        stops: [
            { position: 0.0, color: [20, 40, 10, 255] },
            { position: 0.5, color: [60, 120, 30, 255] },
            { position: 1.0, color: [150, 255, 100, 255] }
        ]
    },
    {
        name: "Fire Heat",
        stops: [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 0.3, color: [80, 0, 0, 255] },
            { position: 0.6, color: [255, 80, 0, 255] },
            { position: 0.85, color: [255, 200, 0, 255] },
            { position: 1.0, color: [255, 255, 200, 255] }
        ]
    },
    {
        name: "Purple Neon",
        stops: [
            { position: 0.0, color: [10, 5, 20, 255] },
            { position: 0.3, color: [60, 20, 80, 255] },
            { position: 0.7, color: [150, 50, 200, 255] },
            { position: 1.0, color: [255, 150, 255, 255] }
        ]
    }
];

// 国际化函数
export const getI18nText = (key, defaultText) => {
    let lang = 'en';

    try {
        const stored = localStorage.getItem("Comfy.Settings.Locale");
        if (stored) {
            lang = stored.toLowerCase();
        } else if (navigator?.language) {
            lang = navigator.language.toLowerCase();
        }
    } catch (_) {
        lang = 'en';
    }

    const translations = {
        'Pos': { en: 'Pos', zh: '位置' },
        'Color': { en: 'Color', zh: '颜色' },
        'Alpha': { en: 'Alpha', zh: '透明度' },
        'Delete': { en: 'Delete', zh: '删除' },
        'Reverse': { en: 'Reverse', zh: '反转' },
        'Presets': { en: 'Presets', zh: '预设' }
    };

    const translation = translations[key];
    if (!translation) return defaultText;

    return (
        translation[lang] ||
        (lang.length > 2 ? translation[lang.slice(0, 2)] : undefined) ||
        defaultText
    );
};

// 实用工具函数
export const gradientUtils = {
    // 生成CSS渐变字符串
    genGradientCSS(stops) {
        const colorStops = stops.map(stop => {
            const [r, g, b, a] = stop.color;
            return `rgba(${r}, ${g}, ${b}, ${a / 255}) ${(stop.position * 100).toFixed(1)}%`;
        }).join(', ');
        return `linear-gradient(to right, ${colorStops})`;
    },

    // 颜色数组转十六进制
    colorToHex(color) {
        const [r, g, b] = color;
        return "#" + [r, g, b].map(x => {
            const hex = x.toString(16);
            return hex.length === 1 ? "0" + hex : hex;
        }).join("");
    },

    // 十六进制转颜色数组
    hexToColor(hex) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return [r, g, b];
    },

    // 约束值到范围
    clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    },

    // 深拷贝渐变数据
    cloneGradient(stops) {
        return stops.map(stop => ({
            position: stop.position,
            color: [...stop.color]
        }));
    }
};

// 渐变编辑器类
export class GradientEditor {
    constructor(container, initialStops = null) {
        this.container = container;
        this.stops = initialStops || [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 1.0, color: [255, 255, 255, 255] }
        ];
        this.selectedStopIndex = 0;
        this.onUpdate = null;

        this.init();
    }

    init() {
        this.container.innerHTML = '';
        this.container.className = 'rc-gradient-editor';

        // 渐变条
        this.barContainer = document.createElement("div");
        this.barContainer.className = "rc-gradient-bar-container";
        this.gradientBar = document.createElement("div");
        this.gradientBar.className = "rc-gradient-bar";
        this.barContainer.appendChild(this.gradientBar);
        this.container.appendChild(this.barContainer);

        // 控制面板
        this.createControls();
        this.createPresets();

        // 事件监听
        this.setupEvents();

        // 初始更新
        this.updateGradient();
        this.updateControls();
    }

    createControls() {
        const posRow = document.createElement("div");
        posRow.className = "rc-control-row";
        posRow.innerHTML = `
            <span class="rc-control-label">${getI18nText('Pos', 'Pos')}:</span>
            <input type="range" class="rc-control-input rc-control-slider pos-slider" min="0" max="1" step="0.001" value="0">
            <input type="number" class="rc-control-input rc-control-number pos-number" min="0" max="1" step="0.01" value="0">
        `;

        const colorRow = document.createElement("div");
        colorRow.className = "rc-control-row";
        colorRow.innerHTML = `
            <span class="rc-control-label">${getI18nText('Color', 'Color')}:</span>
            <input type="color" class="rc-color-picker" value="#000000">
        `;

        const alphaRow = document.createElement("div");
        alphaRow.className = "rc-control-row";
        alphaRow.innerHTML = `
            <span class="rc-control-label">${getI18nText('Alpha', 'Alpha')}:</span>
            <input type="range" class="rc-control-input rc-control-slider alpha-slider" min="0" max="255" step="1" value="255">
            <input type="number" class="rc-control-input rc-control-number alpha-number" min="0" max="255" step="1" value="255">
        `;

        const buttonRow = document.createElement("div");
        buttonRow.className = "rc-control-row";
        buttonRow.innerHTML = `
            <button class="rc-btn danger delete-btn">${getI18nText('Delete', 'Delete')}</button>
            <button class="rc-btn reverse-btn">${getI18nText('Reverse', 'Reverse')}</button>
        `;

        this.container.appendChild(posRow);
        this.container.appendChild(colorRow);
        this.container.appendChild(alphaRow);
        this.container.appendChild(buttonRow);

        // 获取控制元素引用
        this.posSlider = this.container.querySelector(".pos-slider");
        this.posNumber = this.container.querySelector(".pos-number");
        this.colorPicker = this.container.querySelector(".rc-color-picker");
        this.alphaSlider = this.container.querySelector(".alpha-slider");
        this.alphaNumber = this.container.querySelector(".alpha-number");
        this.deleteBtn = this.container.querySelector(".delete-btn");
        this.reverseBtn = this.container.querySelector(".reverse-btn");
    }

    createPresets() {
        const presetsRow = document.createElement("div");
        presetsRow.className = "rc-control-row";
        presetsRow.innerHTML = `
            <span class="rc-control-label">${getI18nText('Presets', 'Presets')}:</span>
        `;

        const presetsContainer = document.createElement("div");
        presetsContainer.className = "rc-presets";

        gradientPresets.forEach(preset => {
            const presetEl = document.createElement("div");
            presetEl.className = "rc-preset";
            presetEl.style.background = gradientUtils.genGradientCSS(preset.stops);
            presetEl.title = preset.name;
            presetEl.addEventListener('click', () => {
                this.loadPreset(preset.stops);
            });
            presetsContainer.appendChild(presetEl);
        });

        this.container.appendChild(presetsRow);
        this.container.appendChild(presetsContainer);
    }

    setupEvents() {
        // 渐变条点击添加色标
        this.barContainer.addEventListener('click', (e) => {
            if (e.target === this.barContainer || e.target === this.gradientBar) {
                this.addStopAtPosition(e);
            }
        });

        // 位置控制
        this.posSlider.addEventListener('input', (e) => {
            this.stops[this.selectedStopIndex].position = parseFloat(e.target.value);
            this.posNumber.value = e.target.value;
            this.updateGradient();
            this.triggerUpdate();
        });

        this.posNumber.addEventListener('input', (e) => {
            const value = gradientUtils.clamp(parseFloat(e.target.value), 0, 1);
            this.stops[this.selectedStopIndex].position = value;
            this.posSlider.value = value;
            this.updateGradient();
            this.triggerUpdate();
        });

        // 颜色控制
        this.colorPicker.addEventListener('input', (e) => {
            const color = gradientUtils.hexToColor(e.target.value);
            this.stops[this.selectedStopIndex].color[0] = color[0];
            this.stops[this.selectedStopIndex].color[1] = color[1];
            this.stops[this.selectedStopIndex].color[2] = color[2];
            this.updateGradient();
            this.triggerUpdate();
        });

        // 透明度控制
        this.alphaSlider.addEventListener('input', (e) => {
            this.stops[this.selectedStopIndex].color[3] = parseInt(e.target.value);
            this.alphaNumber.value = e.target.value;
            this.updateGradient();
            this.triggerUpdate();
        });

        this.alphaNumber.addEventListener('input', (e) => {
            const value = gradientUtils.clamp(parseInt(e.target.value), 0, 255);
            this.stops[this.selectedStopIndex].color[3] = value;
            this.alphaSlider.value = value;
            this.updateGradient();
            this.triggerUpdate();
        });

        // 删除按钮
        this.deleteBtn.addEventListener('click', () => {
            if (this.stops.length > 2) {
                this.stops.splice(this.selectedStopIndex, 1);
                this.selectedStopIndex = Math.min(this.selectedStopIndex, this.stops.length - 1);
                this.updateGradient();
                this.updateControls();
                this.triggerUpdate();
            }
        });

        // 反转按钮
        this.reverseBtn.addEventListener('click', () => {
            this.stops.forEach(stop => {
                stop.position = 1.0 - stop.position;
            });
            this.stops.reverse();
            this.updateGradient();
            this.updateControls();
            this.triggerUpdate();
        });
    }

    addStopAtPosition(e) {
        const rect = this.barContainer.getBoundingClientRect();
        const position = (e.clientX - rect.left) / rect.width;
        const clampedPos = gradientUtils.clamp(position, 0, 1);

        // 在合适的位置插入新色标
        let insertIndex = this.stops.length;
        for (let i = 0; i < this.stops.length; i++) {
            if (this.stops[i].position > clampedPos) {
                insertIndex = i;
                break;
            }
        }

        const newStop = {
            position: clampedPos,
            color: [0, 0, 0, 255] // 默认黑色
        };

        this.stops.splice(insertIndex, 0, newStop);
        this.selectedStopIndex = insertIndex;
        this.updateGradient();
        this.updateControls();
        this.triggerUpdate();
    }

    updateGradient() {
        // 排序色标
        this.stops.sort((a, b) => a.position - b.position);

        // 更新渐变显示
        this.gradientBar.style.background = gradientUtils.genGradientCSS(this.stops);

        // 清除现有色标标记
        const existingStops = this.barContainer.querySelectorAll('.rc-gradient-stop');
        existingStops.forEach(stop => stop.remove());

        // 添加色标标记
        this.stops.forEach((stop, index) => {
            const stopEl = document.createElement('div');
            stopEl.className = 'rc-gradient-stop';
            if (index === this.selectedStopIndex) {
                stopEl.classList.add('selected');
            }
            stopEl.style.left = (stop.position * 100) + '%';
            stopEl.style.backgroundColor = gradientUtils.colorToHex(stop.color);

            stopEl.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectedStopIndex = index;
                this.updateGradient();
                this.updateControls();
            });

            this.barContainer.appendChild(stopEl);
        });
    }

    updateControls() {
        const stop = this.stops[this.selectedStopIndex];
        this.posSlider.value = stop.position;
        this.posNumber.value = stop.position.toFixed(3);
        this.colorPicker.value = gradientUtils.colorToHex(stop.color);
        this.alphaSlider.value = stop.color[3];
        this.alphaNumber.value = stop.color[3];

        this.deleteBtn.disabled = this.stops.length <= 2;
    }

    loadPreset(presetStops) {
        this.stops = gradientUtils.cloneGradient(presetStops);
        this.selectedStopIndex = 0;
        this.updateGradient();
        this.updateControls();
        this.triggerUpdate();
    }

    getStops() {
        return gradientUtils.cloneGradient(this.stops);
    }

    setStops(stops) {
        this.stops = gradientUtils.cloneGradient(stops);
        this.selectedStopIndex = 0;
        this.updateGradient();
        this.updateControls();
    }

    triggerUpdate() {
        if (this.onUpdate) {
            this.onUpdate(this.getStops());
        }
    }
}