import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// CSS styles
const styles = `
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
}

.rc-gradient-bar-container {
    position: relative;
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

.rc-gradient-stop:active {
    cursor: grabbing;
}

.rc-gradient-stop.selected {
    border-color: #4a9eff;
    box-shadow: 0 0 8px rgba(74, 158, 255, 1);
}

.rc-control-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    gap: 6px;
}

.rc-control-label {
    color: #ccc;
    font-size: 11px;
    min-width: 38px;
    flex-shrink: 0;
}

.rc-control-input {
    background: #2a2a2a;
    border: 1px solid #444;
    color: #ddd;
    padding: 3px 6px;
    border-radius: 3px;
    font-size: 11px;
    box-sizing: border-box;
}

.rc-control-slider {
    flex: 1;
    min-width: 0;
    height: 18px;
    margin: 0;
    padding: 0;
}

.alpha-slider {
    flex: 1.2;  /* 让透明度滑块稍微宽一些 */
}

.rc-control-number {
    width: 50px;
    flex-shrink: 0;
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

if (!document.getElementById("rc-gradient-styles")) {
    const styleEl = document.createElement("style");
    styleEl.id = "rc-gradient-styles";
    styleEl.textContent = styles;
    document.head.appendChild(styleEl);
}

const PRESETS = [
    {
        name: "BW",
        stops: [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 1.0, color: [255, 255, 255, 255] }
        ]
    },
    {
        name: "Trans",
        stops: [
            { position: 0.0, color: [0, 0, 0, 0] },
            { position: 1.0, color: [0, 0, 0, 255] }
        ]
    },
    {
        name: "Rainbow",
        stops: [
            { position: 0.0, color: [255, 0, 0, 255] },
            { position: 0.25, color: [255, 255, 0, 255] },
            { position: 0.5, color: [0, 255, 0, 255] },
            { position: 0.75, color: [0, 0, 255, 255] },
            { position: 1.0, color: [255, 0, 255, 255] }
        ]
    },
    {
        name: "Sunset",
        stops: [
            { position: 0.0, color: [255, 94, 77, 255] },
            { position: 1.0, color: [252, 206, 77, 255] }
        ]
    },
    {
        name: "Ocean",
        stops: [
            { position: 0.0, color: [0, 32, 96, 255] },
            { position: 1.0, color: [0, 150, 200, 255] }
        ]
    },
    {
        name: "Gold",
        stops: [
            { position: 0.0, color: [212, 175, 55, 255] },
            { position: 0.3, color: [255, 223, 128, 255] },
            { position: 0.6, color: [218, 165, 32, 255] },
            { position: 0.85, color: [184, 134, 11, 255] },
            { position: 1.0, color: [139, 101, 8, 255] }
        ]
    },
    {
        name: "Purple",
        stops: [
            { position: 0.0, color: [131, 58, 180, 255] },
            { position: 0.5, color: [189, 93, 214, 255] },
            { position: 1.0, color: [253, 145, 213, 255] }
        ]
    },
    {
        name: "Fire",
        stops: [
            { position: 0.0, color: [255, 0, 0, 255] },
            { position: 0.5, color: [255, 140, 0, 255] },
            { position: 1.0, color: [255, 255, 0, 255] }
        ]
    }
];

app.registerExtension({
    name: "RC.GradientGenerator",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RC_GradientGenerator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated?.apply(this, arguments);
                
                const widget = this.widgets.find(w => w.name === "gradient_data");
                if (!widget) return ret;
                
                // 基于浏览器语言的国际化函数
                const getI18nText = (key, defaultText) => {
                    // 获取浏览器语言或ComfyUI设置
                    let lang = 'en'; // 默认语言
                    
                    // 尝试从ComfyUI设置获取语言
                    if (typeof localStorage !== 'undefined') {
                        const comfyLang = localStorage.getItem("Comfy.Settings.Locale");
                        if (comfyLang) {
                            lang = comfyLang.toLowerCase().substring(0, 2); // 获取语言代码前两位
                        } else {
                            // 如果没有ComfyUI设置，则使用浏览器语言
                            lang = (navigator.language || navigator.userLanguage || 'en').toLowerCase().substring(0, 2);
                        }
                    }
                    
                    // 翻译映射
                    const translations = {
                        'Pos': { en: 'Pos', zh: '位置', 'zh-cn': '位置' },
                        'Color': { en: 'Color', zh: '颜色', 'zh-cn': '颜色' },
                        'Alpha': { en: 'Alpha', zh: '透明', 'zh-cn': '透明' },
                        'Delete': { en: 'Delete', zh: '删除', 'zh-cn': '删除' },
                        'Reverse': { en: 'Reverse', zh: '反转', 'zh-cn': '反转' },
                        'Presets': { en: 'Presets', zh: '预设', 'zh-cn': '预设' }
                    };
                    
                    const langTranslations = translations[key];
                    if (langTranslations) {
                        // 优先使用完整语言代码，否则使用语言前缀
                        return langTranslations[lang] || langTranslations[lang.substring(0, 2)] || defaultText;
                    }
                    return defaultText;
                };
                
                // Hide original widget
                widget.type = "converted-widget";
                widget.serializeValue = () => widget.value;
                
                // Parse initial data
                let stops;
                try {
                    const data = JSON.parse(widget.value);
                    stops = data.stops;
                } catch {
                    stops = [
                        { position: 0.0, color: [0, 0, 0, 255] },
                        { position: 1.0, color: [255, 255, 255, 255] }
                    ];
                }
                
                let selectedStop = 0;
                
                // Create UI
                const container = document.createElement("div");
                container.className = "rc-gradient-editor";
                
                // Gradient bar
                const barContainer = document.createElement("div");
                barContainer.className = "rc-gradient-bar-container";
                const gradientBar = document.createElement("div");
                gradientBar.className = "rc-gradient-bar";
                barContainer.appendChild(gradientBar);
                container.appendChild(barContainer);
                
                // 获取翻译文本
                const posLabel = getI18nText('Pos', 'Pos');
                const colorLabel = getI18nText('Color', 'Color');
                const alphaLabel = getI18nText('Alpha', 'Alpha');
                const deleteLabel = getI18nText('Delete', 'Delete');
                const reverseLabel = getI18nText('Reverse', 'Reverse');
                const presetsLabel = getI18nText('Presets', 'Presets');
                
                // Controls - create elements with placeholders first
                const posRow = document.createElement("div");
                posRow.className = "rc-control-row";
                posRow.innerHTML = `
                    <span class="rc-control-label" data-i18n-key="Pos">Pos:</span>
                    <input type="range" class="rc-control-input rc-control-slider pos-slider" min="0" max="1" step="0.001" value="0">
                    <input type="number" class="rc-control-input rc-control-number pos-number" min="0" max="1" step="0.01" value="0">
                `;
                
                const colorRow = document.createElement("div");
                colorRow.className = "rc-control-row";
                colorRow.innerHTML = `
                    <span class="rc-control-label" data-i18n-key="Color">Color:</span>
                    <input type="color" class="rc-color-picker" value="#000000">
                `;
                
                const alphaRow = document.createElement("div");
                alphaRow.className = "rc-control-row";
                alphaRow.innerHTML = `
                    <span class="rc-control-label" data-i18n-key="Alpha">Alpha:</span>
                    <input type="range" class="rc-control-input rc-control-slider alpha-slider" min="0" max="255" step="1" value="255">
                    <input type="number" class="rc-control-input rc-control-number alpha-number" min="0" max="255" step="1" value="255">
                `;
                
                const buttonRow = document.createElement("div");
                buttonRow.className = "rc-control-row";
                buttonRow.innerHTML = `
                    <button class="rc-btn danger delete-btn" data-i18n-key="Delete">Delete</button>
                    <button class="rc-btn reverse-btn" data-i18n-key="Reverse">Reverse</button>
                `;
                
                const presetsRow = document.createElement("div");
                presetsRow.className = "rc-control-row";
                presetsRow.innerHTML = `
                    <span class="rc-control-label" data-i18n-key="Presets">Presets:</span>
                `;
                
                const presetsContainer = document.createElement("div");
                presetsContainer.className = "rc-presets";
                
                container.appendChild(posRow);
                container.appendChild(colorRow);
                container.appendChild(alphaRow);
                container.appendChild(buttonRow);
                container.appendChild(presetsRow);
                container.appendChild(presetsContainer);
                
                const posSlider = container.querySelector(".pos-slider");
                const posNumber = container.querySelector(".pos-number");
                const colorPicker = container.querySelector(".rc-color-picker");
                const alphaSlider = container.querySelector(".alpha-slider");
                const alphaNumber = container.querySelector(".alpha-number");
                const deleteBtn = container.querySelector(".delete-btn");
                const reverseBtn = container.querySelector(".reverse-btn");
                
                posRow.querySelector('.rc-control-label').textContent = posLabel + ':';
                colorRow.querySelector('.rc-control-label').textContent = colorLabel + ':';
                alphaRow.querySelector('.rc-control-label').textContent = alphaLabel + ':';
                deleteBtn.textContent = deleteLabel;
                reverseBtn.textContent = reverseLabel;
                presetsRow.querySelector('.rc-control-label').textContent = presetsLabel + ':';
                
                // Helper functions
                const genGradientCSS = (stops) => {
                    if (!stops || stops.length === 0) {
                        return 'linear-gradient(to right, rgba(0,0,0,1) 0%, rgba(255,255,255,1) 100%)';
                    }
                    
                    // 确保有起始和结束点
                    let sortedStops = [...stops].sort((a, b) => a.position - b.position);
                    
                    // 如果第一个停止点不是0，则从相同颜色添加一个0%的点
                    if (sortedStops[0].position > 0) {
                        const firstColor = sortedStops[0].color;
                        sortedStops = [{ position: 0, color: firstColor }, ...sortedStops];
                    }
                    
                    // 如果最后一个停止点不是1，则从相同颜色添加一个100%的点
                    if (sortedStops[sortedStops.length - 1].position < 1) {
                        const lastColor = sortedStops[sortedStops.length - 1].color;
                        sortedStops = [...sortedStops, { position: 1, color: lastColor }];
                    }
                    
                    const strs = sortedStops.map(s => {
                        const [r, g, b, a] = s.color;
                        return `rgba(${r},${g},${b},${a/255}) ${s.position*100}%`;
                    });
                    return `linear-gradient(to right, ${strs.join(', ')})`;
                };
                
                const interpolateColor = (pos) => {
                    const sorted = [...stops].sort((a,b) => a.position - b.position);
                    let left = sorted[0], right = sorted[sorted.length-1];
                    for (let i = 0; i < sorted.length - 1; i++) {
                        if (sorted[i].position <= pos && pos <= sorted[i+1].position) {
                            left = sorted[i];
                            right = sorted[i+1];
                            break;
                        }
                    }
                    const t = (pos - left.position) / (right.position - left.position + 0.0001);
                    return left.color.map((c, i) => Math.round(c * (1-t) + right.color[i] * t));
                };
                
                const updateData = () => {
                    widget.value = JSON.stringify({ stops });
                };
                
                const updateUI = () => {
                    gradientBar.style.background = genGradientCSS(stops);
                    
                    // Remove old stops
                    barContainer.querySelectorAll(".rc-gradient-stop").forEach(el => el.remove());
                    
                    // Create stop elements
                    stops.forEach((stop, idx) => {
                        const stopEl = document.createElement("div");
                        stopEl.className = "rc-gradient-stop" + (idx === selectedStop ? " selected" : "");
                        stopEl.style.left = `${stop.position * 100}%`;
                        const [r, g, b, a] = stop.color;
                        stopEl.style.backgroundColor = `rgba(${r},${g},${b},${a/255})`;
                        
                        stopEl.onclick = (e) => {
                            e.stopPropagation();
                            selectedStop = idx;
                            updateUI();
                        };
                        
                        let dragging = false;
                        stopEl.onmousedown = (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            dragging = true;
                            selectedStop = idx;
                            updateUI();
                        };
                        
                        const onMouseMove = (e) => {
                            if (!dragging) return;
                            const rect = barContainer.getBoundingClientRect();
                            let pos = (e.clientX - rect.left) / rect.width;
                            pos = Math.max(0, Math.min(1, pos));
                            stops[idx].position = pos;
                            updateUI();
                            updateData();
                        };
                        
                        const onMouseUp = () => {
                            if (dragging) {
                                dragging = false;
                                stops.sort((a, b) => a.position - b.position);
                                selectedStop = stops.indexOf(stop);
                                updateUI();
                            }
                        };
                        
                        document.addEventListener("mousemove", onMouseMove);
                        document.addEventListener("mouseup", onMouseUp);
                        
                        barContainer.appendChild(stopEl);
                    });
                    
                    // Update controls
                    const curr = stops[selectedStop];
                    posSlider.value = curr.position;
                    posNumber.value = curr.position.toFixed(3);
                    const [r, g, b, a] = curr.color;
                    colorPicker.value = `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
                    alphaSlider.value = a;
                    alphaNumber.value = a;
                };
                
                // Bar click to add stop
                barContainer.onclick = (e) => {
                    if (e.target === barContainer || e.target === gradientBar) {
                        const rect = barContainer.getBoundingClientRect();
                        const pos = (e.clientX - rect.left) / rect.width;
                        stops.push({ position: pos, color: interpolateColor(pos) });
                        stops.sort((a, b) => a.position - b.position);
                        selectedStop = stops.findIndex(s => s.position === pos);
                        updateUI();
                        updateData();
                    }
                };
                
                // Event listeners
                posSlider.oninput = () => {
                    let pos = parseFloat(posSlider.value);
                    // 确保值在有效范围内
                    pos = Math.max(0, Math.min(1, pos));
                    posNumber.value = pos.toFixed(3);
                    posSlider.value = pos;
                    stops[selectedStop].position = pos;
                    stops.sort((a, b) => a.position - b.position);
                    selectedStop = stops.indexOf(stops.find(s => Math.abs(s.position - pos) < 0.001));
                    updateUI();
                    updateData();
                };
                
                posNumber.oninput = () => {
                    let pos = parseFloat(posNumber.value);
                    if (!isNaN(pos)) {
                        // 确保值在有效范围内
                        pos = Math.max(0, Math.min(1, pos));
                        posSlider.value = pos;
                        posNumber.value = pos.toFixed(3);
                        stops[selectedStop].position = pos;
                        stops.sort((a, b) => a.position - b.position);
                        selectedStop = stops.indexOf(stops.find(s => Math.abs(s.position - pos) < 0.001));
                        updateUI();
                        updateData();
                    }
                };
                
                // 同时处理输入框的change事件，确保值被正确应用
                posNumber.onchange = () => {
                    let pos = parseFloat(posNumber.value);
                    if (!isNaN(pos)) {
                        // 确保值在有效范围内
                        pos = Math.max(0, Math.min(1, pos));
                        posSlider.value = pos;
                        posNumber.value = pos.toFixed(3);
                        stops[selectedStop].position = pos;
                        stops.sort((a, b) => a.position - b.position);
                        selectedStop = stops.indexOf(stops.find(s => Math.abs(s.position - pos) < 0.001));
                        updateUI();
                        updateData();
                    }
                };
                
                colorPicker.oninput = () => {
                    const hex = colorPicker.value;
                    const r = parseInt(hex.substr(1, 2), 16);
                    const g = parseInt(hex.substr(3, 2), 16);
                    const b = parseInt(hex.substr(5, 2), 16);
                    stops[selectedStop].color = [r, g, b, stops[selectedStop].color[3]];
                    updateUI();
                    updateData();
                };
                
                alphaSlider.oninput = () => {
                    const a = parseInt(alphaSlider.value);
                    alphaNumber.value = a;
                    stops[selectedStop].color[3] = a;
                    updateUI();
                    updateData();
                };
                
                alphaNumber.oninput = () => {
                    const a = parseInt(alphaNumber.value);
                    if (!isNaN(a)) {
                        alphaSlider.value = a;
                        stops[selectedStop].color[3] = a;
                        updateUI();
                        updateData();
                    }
                };
                
                // 同时处理输入框的change事件，确保值被正确应用
                alphaNumber.onchange = () => {
                    let a = parseInt(alphaNumber.value);
                    // 确保值在有效范围内
                    a = Math.max(0, Math.min(255, a));
                    alphaNumber.value = a;
                    alphaSlider.value = a;
                    stops[selectedStop].color[3] = a;
                    updateUI();
                    updateData();
                };
                
                deleteBtn.onclick = () => {
                    if (stops.length <= 2) return;
                    stops.splice(selectedStop, 1);
                    selectedStop = Math.max(0, selectedStop - 1);
                    updateUI();
                    updateData();
                };
                
                reverseBtn.onclick = () => {
                    stops.forEach(s => s.position = 1 - s.position);
                    stops.sort((a, b) => a.position - b.position);
                    updateUI();
                    updateData();
                };
                
                // Create presets
                PRESETS.forEach(preset => {
                    const presetEl = document.createElement("div");
                    presetEl.className = "rc-preset";
                    presetEl.title = preset.name;
                    presetEl.style.background = genGradientCSS(preset.stops);
                    presetEl.onclick = () => {
                        stops = JSON.parse(JSON.stringify(preset.stops));
                        selectedStop = 0;
                        updateUI();
                        updateData();
                    };
                    presetsContainer.appendChild(presetEl);
                });
                
                updateUI();
                
                // Add custom HTML widget
                const htmlWidget = this.addDOMWidget("gradient_editor", "div", container);
                htmlWidget.computeSize = function(width) {
                    return [width, 280];  // 恢复合适的高度
                };
                
                // 确保节点足够宽
                this.setSize([Math.max(this.size[0], 340), this.size[1]]);
                
                return ret;
            };
        }
    }
});