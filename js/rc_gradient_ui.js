import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// CSS styles for gradient editor
const styles = `
.rc-gradient-editor {
    width: 100%;
    margin: 10px 0;
    font-family: Arial, sans-serif;
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
    position: relative;
}

.rc-gradient-stop {
    position: absolute;
    width: 14px;
    height: 14px;
    border: 2px solid white;
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    cursor: pointer;
    box-shadow: 0 0 4px rgba(0,0,0,0.5);
    transition: transform 0.1s;
}

.rc-gradient-stop:hover {
    transform: translate(-50%, -50%) scale(1.2);
}

.rc-gradient-stop.selected {
    border-color: #4a9eff;
    box-shadow: 0 0 8px rgba(74, 158, 255, 0.8);
}

.rc-gradient-controls {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 10px;
    padding: 10px;
    background: #2a2a2a;
    border-radius: 4px;
}

.rc-gradient-control-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.rc-gradient-control-label {
    min-width: 60px;
    color: #ddd;
    font-size: 12px;
}

.rc-gradient-color-input {
    width: 50px;
    height: 25px;
    border: 1px solid #555;
    border-radius: 3px;
    cursor: pointer;
}

.rc-gradient-slider {
    flex: 1;
    height: 4px;
}

.rc-gradient-number {
    width: 50px;
    background: #1a1a1a;
    border: 1px solid #555;
    color: #ddd;
    padding: 4px;
    border-radius: 3px;
    font-size: 12px;
}

.rc-gradient-button {
    padding: 5px 12px;
    background: #4a9eff;
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.2s;
}

.rc-gradient-button:hover {
    background: #3a8eef;
}

.rc-gradient-button.delete {
    background: #ff4a4a;
}

.rc-gradient-button.delete:hover {
    background: #ef3a3a;
}

.rc-gradient-presets {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    margin-top: 5px;
}

.rc-gradient-preset {
    width: 60px;
    height: 20px;
    border: 2px solid #555;
    border-radius: 3px;
    cursor: pointer;
    transition: border-color 0.2s;
}

.rc-gradient-preset:hover {
    border-color: #4a9eff;
}
`;

// Add styles to document
const styleSheet = document.createElement("style");
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);

// Gradient presets
const GRADIENT_PRESETS = [
    {
        name: "Black to White",
        stops: [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 1.0, color: [255, 255, 255, 255] }
        ]
    },
    {
        name: "Transparent to Black",
        stops: [
            { position: 0.0, color: [0, 0, 0, 0] },
            { position: 1.0, color: [0, 0, 0, 255] }
        ]
    },
    {
        name: "Rainbow",
        stops: [
            { position: 0.0, color: [255, 0, 0, 255] },
            { position: 0.2, color: [255, 255, 0, 255] },
            { position: 0.4, color: [0, 255, 0, 255] },
            { position: 0.6, color: [0, 255, 255, 255] },
            { position: 0.8, color: [0, 0, 255, 255] },
            { position: 1.0, color: [255, 0, 255, 255] }
        ]
    },
    {
        name: "Sunset",
        stops: [
            { position: 0.0, color: [255, 94, 77, 255] },
            { position: 0.5, color: [245, 140, 60, 255] },
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

class GradientEditor {
    constructor(node, inputName, inputData, app) {
        this.node = node;
        this.inputName = inputName;
        this.selectedStop = 0;
        
        // Initialize gradient stops
        this.stops = [
            { position: 0.0, color: [0, 0, 0, 255] },
            { position: 1.0, color: [255, 255, 255, 255] }
        ];
        
        this.createWidget();
    }
    
    createWidget() {
        const container = document.createElement("div");
        container.className = "rc-gradient-editor";
        
        // Gradient bar
        const barContainer = document.createElement("div");
        barContainer.className = "rc-gradient-bar-container";
        
        const gradientBar = document.createElement("div");
        gradientBar.className = "rc-gradient-bar";
        barContainer.appendChild(gradientBar);
        
        // Add click handler to create new stops
        barContainer.addEventListener("click", (e) => {
            if (e.target === barContainer || e.target === gradientBar) {
                const rect = barContainer.getBoundingClientRect();
                const position = (e.clientX - rect.left) / rect.width;
                this.addStop(position);
            }
        });
        
        container.appendChild(barContainer);
        
        // Controls
        const controls = document.createElement("div");
        controls.className = "rc-gradient-controls";
        
        // Position control
        const posRow = this.createControlRow("Position:", "range", 0, 1, 0.01);
        controls.appendChild(posRow);
        
        // Color control
        const colorRow = document.createElement("div");
        colorRow.className = "rc-gradient-control-row";
        const colorLabel = document.createElement("span");
        colorLabel.className = "rc-gradient-control-label";
        colorLabel.textContent = "Color:";
        const colorInput = document.createElement("input");
        colorInput.type = "color";
        colorInput.className = "rc-gradient-color-input";
        colorRow.appendChild(colorLabel);
        colorRow.appendChild(colorInput);
        controls.appendChild(colorRow);
        
        // Alpha control
        const alphaRow = this.createControlRow("Alpha:", "range", 0, 255, 1);
        controls.appendChild(alphaRow);
        
        // Buttons
        const buttonRow = document.createElement("div");
        buttonRow.className = "rc-gradient-control-row";
        
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "rc-gradient-button delete";
        deleteBtn.textContent = "Delete Stop";
        deleteBtn.addEventListener("click", () => this.deleteSelectedStop());
        
        const reverseBtn = document.createElement("button");
        reverseBtn.className = "rc-gradient-button";
        reverseBtn.textContent = "Reverse";
        reverseBtn.addEventListener("click", () => this.reverseGradient());
        
        buttonRow.appendChild(deleteBtn);
        buttonRow.appendChild(reverseBtn);
        controls.appendChild(buttonRow);
        
        // Presets
        const presetsLabel = document.createElement("div");
        presetsLabel.style.color = "#ddd";
        presetsLabel.style.fontSize = "12px";
        presetsLabel.style.marginTop = "10px";
        presetsLabel.textContent = "Presets:";
        controls.appendChild(presetsLabel);
        
        const presets = document.createElement("div");
        presets.className = "rc-gradient-presets";
        GRADIENT_PRESETS.forEach(preset => {
            const presetDiv = document.createElement("div");
            presetDiv.className = "rc-gradient-preset";
            presetDiv.title = preset.name;
            presetDiv.style.background = this.generateGradientCSS(preset.stops);
            presetDiv.addEventListener("click", () => this.loadPreset(preset));
            presets.appendChild(presetDiv);
        });
        controls.appendChild(presets);
        
        container.appendChild(controls);
        
        // Store references
        this.container = container;
        this.gradientBar = gradientBar;
        this.barContainer = barContainer;
        this.positionSlider = posRow.querySelector("input[type='range']");
        this.positionNumber = posRow.querySelector("input[type='number']");
        this.colorInput = colorInput;
        this.alphaSlider = alphaRow.querySelector("input[type='range']");
        this.alphaNumber = alphaRow.querySelector("input[type='number']");
        
        // Add event listeners
        this.positionSlider.addEventListener("input", () => this.updateStopPosition());
        this.positionNumber.addEventListener("input", () => this.updateStopPosition());
        this.colorInput.addEventListener("input", () => this.updateStopColor());
        this.alphaSlider.addEventListener("input", () => this.updateStopAlpha());
        this.alphaNumber.addEventListener("input", () => this.updateStopAlpha());
        
        this.updateDisplay();
        this.updateGradientData();
    }
    
    createControlRow(label, type, min, max, step) {
        const row = document.createElement("div");
        row.className = "rc-gradient-control-row";
        
        const labelEl = document.createElement("span");
        labelEl.className = "rc-gradient-control-label";
        labelEl.textContent = label;
        
        const slider = document.createElement("input");
        slider.type = type;
        slider.className = "rc-gradient-slider";
        slider.min = min;
        slider.max = max;
        slider.step = step;
        
        const number = document.createElement("input");
        number.type = "number";
        number.className = "rc-gradient-number";
        number.min = min;
        number.max = max;
        number.step = step;
        
        row.appendChild(labelEl);
        row.appendChild(slider);
        row.appendChild(number);
        
        return row;
    }
    
    addStop(position) {
        const color = this.interpolateColor(position);
        this.stops.push({ position, color });
        this.stops.sort((a, b) => a.position - b.position);
        this.selectedStop = this.stops.findIndex(s => s.position === position);
        this.updateDisplay();
        this.updateGradientData();
    }
    
    deleteSelectedStop() {
        if (this.stops.length <= 2) return;
        this.stops.splice(this.selectedStop, 1);
        this.selectedStop = Math.max(0, this.selectedStop - 1);
        this.updateDisplay();
        this.updateGradientData();
    }
    
    reverseGradient() {
        this.stops.forEach(stop => stop.position = 1 - stop.position);
        this.stops.sort((a, b) => a.position - b.position);
        this.updateDisplay();
        this.updateGradientData();
    }
    
    loadPreset(preset) {
        this.stops = JSON.parse(JSON.stringify(preset.stops));
        this.selectedStop = 0;
        this.updateDisplay();
        this.updateGradientData();
    }
    
    interpolateColor(position) {
        const sortedStops = [...this.stops].sort((a, b) => a.position - b.position);
        
        let leftStop = sortedStops[0];
        let rightStop = sortedStops[sortedStops.length - 1];
        
        for (let i = 0; i < sortedStops.length - 1; i++) {
            if (sortedStops[i].position <= position && position <= sortedStops[i + 1].position) {
                leftStop = sortedStops[i];
                rightStop = sortedStops[i + 1];
                break;
            }
        }
        
        const t = (position - leftStop.position) / (rightStop.position - leftStop.position + 0.0001);
        const color = leftStop.color.map((c, i) => 
            Math.round(c * (1 - t) + rightStop.color[i] * t)
        );
        
        return color;
    }
    
    updateStopPosition() {
        const pos = parseFloat(this.positionSlider.value);
        this.stops[this.selectedStop].position = pos;
        this.stops.sort((a, b) => a.position - b.position);
        this.selectedStop = this.stops.findIndex(s => s.position === pos);
        this.updateDisplay();
        this.updateGradientData();
    }
    
    updateStopColor() {
        const hex = this.colorInput.value;
        const r = parseInt(hex.substr(1, 2), 16);
        const g = parseInt(hex.substr(3, 2), 16);
        const b = parseInt(hex.substr(5, 2), 16);
        const a = this.stops[this.selectedStop].color[3];
        this.stops[this.selectedStop].color = [r, g, b, a];
        this.updateDisplay();
        this.updateGradientData();
    }
    
    updateStopAlpha() {
        const alpha = parseInt(this.alphaSlider.value);
        this.stops[this.selectedStop].color[3] = alpha;
        this.updateDisplay();
        this.updateGradientData();
    }
    
    generateGradientCSS(stops) {
        const stopStrings = stops.map(stop => {
            const [r, g, b, a] = stop.color;
            return `rgba(${r},${g},${b},${a/255}) ${stop.position * 100}%`;
        });
        return `linear-gradient(to right, ${stopStrings.join(', ')})`;
    }
    
    updateDisplay() {
        // Update gradient bar
        this.gradientBar.style.background = this.generateGradientCSS(this.stops);
        
        // Remove old stop elements
        const oldStops = this.barContainer.querySelectorAll(".rc-gradient-stop");
        oldStops.forEach(el => el.remove());
        
        // Create stop elements
        this.stops.forEach((stop, index) => {
            const stopEl = document.createElement("div");
            stopEl.className = "rc-gradient-stop";
            if (index === this.selectedStop) {
                stopEl.classList.add("selected");
            }
            stopEl.style.left = `${stop.position * 100}%`;
            const [r, g, b, a] = stop.color;
            stopEl.style.backgroundColor = `rgba(${r},${g},${b},${a/255})`;
            
            stopEl.addEventListener("click", (e) => {
                e.stopPropagation();
                this.selectedStop = index;
                this.updateDisplay();
            });
            
            // Drag functionality
            let isDragging = false;
            stopEl.addEventListener("mousedown", (e) => {
                e.preventDefault();
                isDragging = true;
            });
            
            document.addEventListener("mousemove", (e) => {
                if (!isDragging) return;
                const rect = this.barContainer.getBoundingClientRect();
                let pos = (e.clientX - rect.left) / rect.width;
                pos = Math.max(0, Math.min(1, pos));
                this.stops[index].position = pos;
                this.updateDisplay();
                this.updateGradientData();
            });
            
            document.addEventListener("mouseup", () => {
                if (isDragging) {
                    isDragging = false;
                    this.stops.sort((a, b) => a.position - b.position);
                    this.selectedStop = this.stops.findIndex(s => s === stop);
                    this.updateDisplay();
                }
            });
            
            this.barContainer.appendChild(stopEl);
        });
        
        // Update controls
        const currentStop = this.stops[this.selectedStop];
        this.positionSlider.value = currentStop.position;
        this.positionNumber.value = currentStop.position.toFixed(2);
        
        const [r, g, b, a] = currentStop.color;
        const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
        this.colorInput.value = hex;
        
        this.alphaSlider.value = a;
        this.alphaNumber.value = a;
    }
    
    updateGradientData() {
        const data = { stops: this.stops };
        const widget = this.node.widgets.find(w => w.name === "gradient_data");
        if (widget) {
            widget.value = JSON.stringify(data);
        }
    }
    
    getWidget() {
        return { widget: this.container };
    }
}

// Register the extension
app.registerExtension({
    name: "RC.GradientGenerator",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RC_GradientGenerator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Find the gradient_data widget
                const gradientDataWidget = this.widgets.find(w => w.name === "gradient_data");
                if (gradientDataWidget) {
                    // Hide the default string widget
                    gradientDataWidget.type = "converted-widget";
                    gradientDataWidget.computeSize = () => [0, -4];
                    
                    // Create custom gradient editor
                    const editor = new GradientEditor(this, "gradient_data", {}, app);
                    
                    // Add editor to node
                    this.addCustomWidget(editor.getWidget());
                }
                
                return result;
            };
        }
    }
});