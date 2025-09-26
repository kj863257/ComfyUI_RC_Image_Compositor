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
                
                // 获取国际化文本，使用ComfyUI的国际化系统
                const getI18nText = (key, defaultText) => {
                    if (app?.getTranslation) {
                        // 使用ComfyUI的国际化系统
                        // 传递完整的路径以获取翻译
                        const translation = app.getTranslation(key);
                        return translation !== key ? translation : defaultText; // 如果没有翻译，返回默认文本
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
                    <span class="rc-control-label">Pos:</span>
                    <input type="range" class="rc-control-input rc-control-slider pos-slider" min="0" max="1" step="0.001" value="0">
                    <input type="number" class="rc-control-input rc-control-number pos-number" min="0" max="1" step="0.01" value="0">
                `;
                
                const colorRow = document.createElement("div");
                colorRow.className = "rc-control-row";
                colorRow.innerHTML = `
                    <span class="rc-control-label">Color:</span>
                    <input type="color" class="rc-color-picker" value="#000000">
                    <span class="rc-control-label" style="min-width:38px;margin-left:6px">Alpha:</span>
                    <input type="range" class="rc-control-input rc-control-slider alpha-slider" min="0" max="255" step="1" value="255">
                    <input type="number" class="rc-control-input rc-control-number alpha-number" min="0" max="255" step="1" value="255">
                `;
                
                const buttonRow = document.createElement("div");
                buttonRow.className = "rc-control-row";
                buttonRow.innerHTML = `
                    <button class="rc-btn danger delete-btn">Delete</button>
                    <button class="rc-btn reverse-btn">Reverse</button>
                `;
                
                const presetsRow = document.createElement("div");
                presetsRow.className = "rc-control-row";
                presetsRow.innerHTML = `
                    <span class="rc-control-label">Presets:</span>
                `;
                
                const presetsContainer = document.createElement("div");
                presetsContainer.className = "rc-presets";
                
                container.appendChild(posRow);
                container.appendChild(colorRow);
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
                
                // Apply translations after elements are added to DOM
                const posLabel = getI18nText('Pos', 'Pos');
                const colorLabel = getI18nText('Color', 'Color');
                const alphaLabel = getI18nText('Alpha', 'Alpha');
                const deleteLabel = getI18nText('Delete', 'Delete');
                const reverseLabel = getI18nText('Reverse', 'Reverse');
                const presetsLabel = getI18nText('Presets', 'Presets');
                
                posRow.querySelector('.rc-control-label').textContent = posLabel + ':';
                colorRow.querySelector('.rc-control-label').textContent = colorLabel + ':';
                colorRow.querySelectorAll('.rc-control-label')[1].textContent = alphaLabel + ':';
                deleteBtn.textContent = deleteLabel;
                reverseBtn.textContent = reverseLabel;
                presetsRow.querySelector('.rc-control-label').textContent = presetsLabel + ':';
                
                // Helper functions
                const genGradientCSS = (stops) => {
                    const strs = stops.map(s => {
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
                posSlider.oninput = posNumber.oninput = () => {
                    const pos = parseFloat(posSlider.value);
                    posNumber.value = pos.toFixed(3);
                    posSlider.value = pos;
                    stops[selectedStop].position = pos;
                    stops.sort((a, b) => a.position - b.position);
                    selectedStop = stops.findIndex(s => s.position === pos);
                    updateUI();
                    updateData();
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
                
                alphaSlider.oninput = alphaNumber.oninput = () => {
                    const a = parseInt(alphaSlider.value);
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