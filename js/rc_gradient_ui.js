import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// CSS styles
const styles = `
.rc-gradient-editor {
    background: #1e1e1e;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}

.rc-gradient-bar-container {
    position: relative;
    width: 100%;
    height: 30px;
    background: #333;
    border-radius: 4px;
    margin: 10px 0;
    cursor: crosshair;
}

.rc-gradient-bar {
    width: 100%;
    height: 100%;
    border-radius: 4px;
}

.rc-gradient-stop {
    position: absolute;
    width: 16px;
    height: 16px;
    border: 3px solid white;
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    cursor: grab;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    z-index: 10;
}

.rc-gradient-stop:active {
    cursor: grabbing;
}

.rc-gradient-stop.selected {
    border-color: #4a9eff;
    box-shadow: 0 0 10px rgba(74, 158, 255, 1);
}

.rc-control-row {
    display: flex;
    align-items: center;
    margin: 8px 0;
    gap: 8px;
}

.rc-control-label {
    color: #ccc;
    font-size: 11px;
    min-width: 50px;
}

.rc-control-input {
    background: #2a2a2a;
    border: 1px solid #444;
    color: #ddd;
    padding: 4px 8px;
    border-radius: 3px;
    font-size: 11px;
}

.rc-control-slider {
    flex: 1;
}

.rc-control-number {
    width: 60px;
}

.rc-color-picker {
    width: 40px;
    height: 24px;
    border: 1px solid #444;
    border-radius: 3px;
    cursor: pointer;
}

.rc-btn {
    background: #4a9eff;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
    cursor: pointer;
    font-size: 11px;
    margin: 2px;
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
    margin-top: 8px;
}

.rc-preset {
    width: 50px;
    height: 18px;
    border: 2px solid #444;
    border-radius: 3px;
    cursor: pointer;
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
                container.style.width = "calc(100% - 20px)";
                
                // Gradient bar
                const barContainer = document.createElement("div");
                barContainer.className = "rc-gradient-bar-container";
                const gradientBar = document.createElement("div");
                gradientBar.className = "rc-gradient-bar";
                barContainer.appendChild(gradientBar);
                container.appendChild(barContainer);
                
                // Controls
                const controlsHtml = `
                    <div class="rc-control-row">
                        <span class="rc-control-label" data-i18n="Position:">Position:</span>
                        <input type="range" class="rc-control-input rc-control-slider pos-slider" min="0" max="1" step="0.001" value="0">
                        <input type="number" class="rc-control-input rc-control-number pos-number" min="0" max="1" step="0.01" value="0">
                    </div>
                    <div class="rc-control-row">
                        <span class="rc-control-label" data-i18n="Color:">Color:</span>
                        <input type="color" class="rc-color-picker" value="#000000">
                        <span class="rc-control-label" style="margin-left:10px" data-i18n="Alpha:">Alpha:</span>
                        <input type="range" class="rc-control-input rc-control-slider alpha-slider" min="0" max="255" step="1" value="255">
                        <input type="number" class="rc-control-input rc-control-number alpha-number" min="0" max="255" step="1" value="255">
                    </div>
                    <div class="rc-control-row">
                        <button class="rc-btn danger delete-btn" data-i18n="Delete">Delete</button>
                        <button class="rc-btn reverse-btn" data-i18n="Reverse">Reverse</button>
                    </div>
                    <div class="rc-control-row">
                        <span class="rc-control-label" data-i18n="Presets:">Presets:</span>
                    </div>
                    <div class="rc-presets"></div>
                `;
                container.insertAdjacentHTML('beforeend', controlsHtml);
                
                // Get elements
                const posSlider = container.querySelector(".pos-slider");
                const posNumber = container.querySelector(".pos-number");
                const colorPicker = container.querySelector(".rc-color-picker");
                const alphaSlider = container.querySelector(".alpha-slider");
                const alphaNumber = container.querySelector(".alpha-number");
                const deleteBtn = container.querySelector(".delete-btn");
                const reverseBtn = container.querySelector(".reverse-btn");
                const presetsContainer = container.querySelector(".rc-presets");
                
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
                    presetEl.title = app.getTranslation ? app.getTranslation(preset.name) : preset.name;
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
                    return [width, 230];
                };
                
                return ret;
            };
        }
    }
});