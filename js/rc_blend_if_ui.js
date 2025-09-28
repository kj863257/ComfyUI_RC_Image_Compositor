import { app } from "../../scripts/app.js";

const styles = `
.rc-blend-if-editor {
    background: #1e1e1e;
    padding: 12px;
    border-radius: 4px;
    margin: 2px 0;
    box-sizing: border-box;
    min-height: 180px;
}

.rc-blend-if-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}

.rc-blend-if-label {
    color: #cfcfcf;
    font-size: 12px;
    font-weight: 500;
    min-width: 60px;
}

.rc-blend-if-channel-select {
    background: #242424;
    border: 1px solid #3a3a3a;
    color: #e0e0e0;
    border-radius: 3px;
    padding: 4px 8px;
    font-size: 11px;
    min-width: 80px;
}

.rc-blend-if-section {
    margin-bottom: 16px;
}

.rc-blend-if-section-title {
    color: #cfcfcf;
    font-size: 11px;
    margin-bottom: 8px;
    font-weight: 500;
}

.rc-blend-if-slider-container {
    position: relative;
    height: 24px;
    background: linear-gradient(to right, #000 0%, #fff 100%);
    border: 1px solid #2b2b2b;
    border-radius: 3px;
    margin-bottom: 8px;
    cursor: crosshair;
    overflow: visible;
}

.rc-blend-if-slider-track {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: transparent;
}

.rc-blend-if-triangle {
    position: absolute;
    top: 50%;
    width: 0;
    height: 0;
    cursor: pointer;
    user-select: none;
    z-index: 2;
}

.rc-blend-if-triangle.black {
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 10px solid #606060;
    transform: translate(-50%, 0%);
    top: 100%;
    filter: drop-shadow(0 0 0 1px rgba(255, 255, 255, 0.3)) drop-shadow(0 1px 2px rgba(0, 0, 0, 0.3));
    transition: border-bottom-color 0.15s ease, filter 0.15s ease;
}

.rc-blend-if-triangle.white {
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 10px solid #c0c0c0;
    transform: translate(-50%, 0%);
    top: 100%;
    filter: drop-shadow(0 0 0 1px rgba(0, 0, 0, 0.4)) drop-shadow(0 1px 2px rgba(0, 0, 0, 0.2));
    transition: border-bottom-color 0.15s ease, filter 0.15s ease;
}


.rc-blend-if-triangle.black:hover {
    border-bottom-color: #707070;
    filter: drop-shadow(0 0 0 1px rgba(255, 255, 255, 0.5)) drop-shadow(0 1px 3px rgba(0, 0, 0, 0.4));
}

.rc-blend-if-triangle.white:hover {
    border-bottom-color: #d0d0d0;
    filter: drop-shadow(0 0 0 1px rgba(0, 0, 0, 0.6)) drop-shadow(0 1px 3px rgba(0, 0, 0, 0.3));
}

.rc-blend-if-triangle.active {
    border-bottom-color: #4a9eff !important;
    filter: drop-shadow(0 0 4px rgba(74, 158, 255, 0.6)) !important;
}

/* Split triangles - left half (right-angled triangle, tip points up-left) */
.rc-blend-if-triangle.split-left {
    border-left: 6px solid transparent;
    border-right: 0px solid transparent;
    border-bottom: 10px solid inherit;
    border-top: 0px solid transparent;
    transform: translate(-50%, 0%);
    top: 100%;
    margin-left: -3px;
}

/* Split triangles - right half (left-angled triangle, tip points up-right) */
.rc-blend-if-triangle.split-right {
    border-left: 0px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 10px solid inherit;
    border-top: 0px solid transparent;
    transform: translate(-50%, 0%);
    top: 100%;
    margin-left: 3px;
}

/* Split triangles hover effects */
.rc-blend-if-triangle.split-left:hover,
.rc-blend-if-triangle.split-right:hover {
    filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.4));
}

.rc-blend-if-values {
    display: flex;
    gap: 6px;
    align-items: center;
    font-size: 10px;
    color: #888;
    margin-top: 18px;
    width: 100%;
    justify-content: space-between;
}

.rc-blend-if-value {
    background: #242424;
    border: 1px solid #3a3a3a;
    border-radius: 2px;
    color: #e0e0e0;
    padding: 2px 4px;
    flex: 1;
    font-size: 10px;
    text-align: center;
    max-width: calc(25% - 4.5px);
}

.rc-blend-if-reset-btn {
    background: #3a3a3a;
    border: 1px solid #4a4a4a;
    color: #dcdcdc;
    border-radius: 3px;
    padding: 3px 8px;
    font-size: 10px;
    cursor: pointer;
    transition: background 0.15s ease;
    margin-left: auto;
}

.rc-blend-if-reset-btn:hover {
    background: #545454;
}

.rc-blend-if-hint {
    color: #666;
    font-size: 9px;
    margin-top: 8px;
    text-align: center;
}
`;

if (!document.getElementById("rc-blend-if-styles")) {
    const styleEl = document.createElement("style");
    styleEl.id = "rc-blend-if-styles";
    styleEl.textContent = styles;
    document.head.appendChild(styleEl);
}

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

    const table = {
        BlendIf: { en: "Blend If", zh: "混合颜色带" },
        Channel: { en: "Channel", zh: "通道" },
        Gray: { en: "Gray", zh: "灰色" },
        Red: { en: "Red", zh: "红" },
        Green: { en: "Green", zh: "绿" },
        Blue: { en: "Blue", zh: "蓝" },
        ThisLayer: { en: "This Layer", zh: "本图层" },
        UnderlyingLayer: { en: "Underlying Layer", zh: "下一图层" },
        Black: { en: "Black", zh: "暗调" },
        White: { en: "White", zh: "亮调" },
        Start: { en: "Start", zh: "开始" },
        End: { en: "End", zh: "结束" },
        Reset: { en: "Reset", zh: "重置" },
        BlendIfHint: {
            en: "Drag triangles to adjust blend ranges. Split ranges create smooth transitions.",
            zh: "拖动三角形调整混合范围。分离范围可创建平滑过渡效果。"
        },
        BlackStart: { en: "Black Start", zh: "黑色起始" },
        BlackEnd: { en: "Black End", zh: "黑色结束" },
        WhiteStart: { en: "White Start", zh: "白色起始" },
        WhiteEnd: { en: "White End", zh: "白色结束" }
    };

    const translations = table[key];
    if (!translations) return fallback;

    return (
        translations[lang] ||
        (lang.length > 2 ? translations[lang.slice(0, 2)] : undefined) ||
        fallback
    );
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

app.registerExtension({
    name: "RC.BlendIf",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "RC_ImageCompositor") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Find the blend_if_data widget
            const widget = this.widgets?.find(w => w.name === "blend_if_data");
            if (!widget) return result;

            // Hide the widget and make it serializable
            widget.type = "converted-widget";
            widget.serializeValue = () => widget.value;

            // Create UI container
            const container = document.createElement("div");
            container.className = "rc-blend-if-editor";

            // Header with channel selection
            const header = document.createElement("div");
            header.className = "rc-blend-if-header";

            const channelLabel = document.createElement("span");
            channelLabel.className = "rc-blend-if-label";
            channelLabel.textContent = translate("BlendIf", "Blend If") + ":";

            const channelSelect = document.createElement("select");
            channelSelect.className = "rc-blend-if-channel-select";

            const channels = [
                { value: "gray", label: translate("Gray", "Gray") },
                { value: "red", label: translate("Red", "Red") },
                { value: "green", label: translate("Green", "Green") },
                { value: "blue", label: translate("Blue", "Blue") }
            ];

            channels.forEach(ch => {
                const option = document.createElement("option");
                option.value = ch.value;
                option.textContent = ch.label;
                channelSelect.appendChild(option);
            });

            header.appendChild(channelLabel);
            header.appendChild(channelSelect);
            container.appendChild(header);

            // This Layer section
            const thisLayerSection = document.createElement("div");
            thisLayerSection.className = "rc-blend-if-section";

            const thisLayerTitle = document.createElement("div");
            thisLayerTitle.className = "rc-blend-if-section-title";
            thisLayerTitle.textContent = translate("ThisLayer", "This Layer");

            const thisLayerSlider = this.createBlendIfSlider("this_layer");
            const thisLayerValues = this.createValueInputs("this_layer");

            thisLayerSection.appendChild(thisLayerTitle);
            thisLayerSection.appendChild(thisLayerSlider.container);
            thisLayerSection.appendChild(thisLayerValues.container);

            // Underlying Layer section
            const underLayerSection = document.createElement("div");
            underLayerSection.className = "rc-blend-if-section";

            const underLayerTitle = document.createElement("div");
            underLayerTitle.className = "rc-blend-if-section-title";
            underLayerTitle.textContent = translate("UnderlyingLayer", "Underlying Layer");

            const underLayerSlider = this.createBlendIfSlider("underlying_layer");
            const underLayerValues = this.createValueInputs("underlying_layer");

            underLayerSection.appendChild(underLayerTitle);
            underLayerSection.appendChild(underLayerSlider.container);
            underLayerSection.appendChild(underLayerValues.container);

            container.appendChild(thisLayerSection);
            container.appendChild(underLayerSection);

            // Reset button and hint
            const footer = document.createElement("div");
            footer.style.display = "flex";
            footer.style.alignItems = "center";
            footer.style.marginTop = "8px";

            const resetBtn = document.createElement("button");
            resetBtn.className = "rc-blend-if-reset-btn";
            resetBtn.textContent = translate("Reset", "Reset");

            const hint = document.createElement("div");
            hint.className = "rc-blend-if-hint";
            hint.textContent = translate("BlendIfHint", "Drag triangles to adjust blend ranges. Alt+drag to split triangles for smooth transitions.");

            footer.appendChild(resetBtn);
            container.appendChild(footer);
            container.appendChild(hint);

            const node = this;

            // Parse initial data from widget
            let settings;
            try {
                settings = JSON.parse(widget.value);
            } catch {
                settings = {
                    channel: "gray",
                    this_layer: { black: [0.0, 0.0], white: [1.0, 1.0] },
                    underlying_layer: { black: [0.0, 0.0], white: [1.0, 1.0] }
                };
            }

            // Update widget when UI changes
            const updateData = () => {
                const data = {
                    channel: channelSelect.value,
                    this_layer: {
                        black: [thisLayerSlider.blackStart, thisLayerSlider.blackEnd],
                        white: [thisLayerSlider.whiteStart, thisLayerSlider.whiteEnd]
                    },
                    underlying_layer: {
                        black: [underLayerSlider.blackStart, underLayerSlider.blackEnd],
                        white: [underLayerSlider.whiteStart, underLayerSlider.whiteEnd]
                    }
                };
                widget.value = JSON.stringify(data);
                if (widget.callback) {
                    widget.callback(widget.value, node, app);
                }
                node?.graph?.setDirtyCanvas(true, true);
            };

            // Initialize values from settings
            channelSelect.value = settings.channel;

            thisLayerSlider.blackStart = settings.this_layer.black[0];
            thisLayerSlider.blackEnd = settings.this_layer.black[1];
            thisLayerSlider.whiteStart = settings.this_layer.white[0];
            thisLayerSlider.whiteEnd = settings.this_layer.white[1];

            underLayerSlider.blackStart = settings.underlying_layer.black[0];
            underLayerSlider.blackEnd = settings.underlying_layer.black[1];
            underLayerSlider.whiteStart = settings.underlying_layer.white[0];
            underLayerSlider.whiteEnd = settings.underlying_layer.white[1];

            // Update slider background based on channel
            const updateSliderBackground = () => {
                const channel = channelSelect.value;
                let gradient;
                switch (channel) {
                    case "red":
                        gradient = "linear-gradient(to right, #000 0%, #ff0000 100%)";
                        break;
                    case "green":
                        gradient = "linear-gradient(to right, #000 0%, #00ff00 100%)";
                        break;
                    case "blue":
                        gradient = "linear-gradient(to right, #000 0%, #0000ff 100%)";
                        break;
                    case "gray":
                    default:
                        gradient = "linear-gradient(to right, #000 0%, #fff 100%)";
                        break;
                }
                thisLayerSlider.container.style.background = gradient;
                underLayerSlider.container.style.background = gradient;
            };

            // Event listeners
            channelSelect.addEventListener("change", () => {
                updateSliderBackground();
                updateData();
            });

            thisLayerSlider.onUpdate = updateData;
            underLayerSlider.onUpdate = updateData;

            // Store references to update functions for each slider
            thisLayerSlider.updateValues = () => thisLayerValues.update();
            underLayerSlider.updateValues = () => underLayerValues.update();

            resetBtn.addEventListener("click", () => {
                channelSelect.value = "gray";
                thisLayerSlider.reset();
                underLayerSlider.reset();
                thisLayerValues.update();
                underLayerValues.update();
                updateData();
            });

            // Update value displays
            thisLayerValues.update = () => {
                thisLayerValues.blackStartInput.value = Math.round(thisLayerSlider.blackStart * 255);
                thisLayerValues.blackEndInput.value = Math.round(thisLayerSlider.blackEnd * 255);
                thisLayerValues.whiteStartInput.value = Math.round(thisLayerSlider.whiteStart * 255);
                thisLayerValues.whiteEndInput.value = Math.round(thisLayerSlider.whiteEnd * 255);
            };

            underLayerValues.update = () => {
                underLayerValues.blackStartInput.value = Math.round(underLayerSlider.blackStart * 255);
                underLayerValues.blackEndInput.value = Math.round(underLayerSlider.blackEnd * 255);
                underLayerValues.whiteStartInput.value = Math.round(underLayerSlider.whiteStart * 255);
                underLayerValues.whiteEndInput.value = Math.round(underLayerSlider.whiteEnd * 255);
            };

            // Add input event listeners for two-way sync
            thisLayerValues.blackStartInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                thisLayerSlider.blackStart = clamp(value, 0, thisLayerSlider.blackEnd);
                thisLayerSlider.updateTriangles();
                thisLayerValues.update();
                updateData();
            });

            thisLayerValues.blackEndInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                thisLayerSlider.blackEnd = clamp(value, thisLayerSlider.blackStart, 1);
                thisLayerSlider.updateTriangles();
                thisLayerValues.update();
                updateData();
            });

            thisLayerValues.whiteStartInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                thisLayerSlider.whiteStart = clamp(value, 0, thisLayerSlider.whiteEnd);
                thisLayerSlider.updateTriangles();
                thisLayerValues.update();
                updateData();
            });

            thisLayerValues.whiteEndInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                thisLayerSlider.whiteEnd = clamp(value, thisLayerSlider.whiteStart, 1);
                thisLayerSlider.updateTriangles();
                thisLayerValues.update();
                updateData();
            });

            underLayerValues.blackStartInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                underLayerSlider.blackStart = clamp(value, 0, underLayerSlider.blackEnd);
                underLayerSlider.updateTriangles();
                underLayerValues.update();
                updateData();
            });

            underLayerValues.blackEndInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                underLayerSlider.blackEnd = clamp(value, underLayerSlider.blackStart, 1);
                underLayerSlider.updateTriangles();
                underLayerValues.update();
                updateData();
            });

            underLayerValues.whiteStartInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                underLayerSlider.whiteStart = clamp(value, 0, underLayerSlider.whiteEnd);
                underLayerSlider.updateTriangles();
                underLayerValues.update();
                updateData();
            });

            underLayerValues.whiteEndInput.addEventListener("input", (e) => {
                const value = clamp(parseFloat(e.target.value) / 255, 0, 1);
                underLayerSlider.whiteEnd = clamp(value, underLayerSlider.whiteStart, 1);
                underLayerSlider.updateTriangles();
                underLayerValues.update();
                updateData();
            });

            // Apply settings from widget data to UI
            const syncDataToUI = (data) => {
                // Update channel selection
                channelSelect.value = data.channel || "gray";

                // Update this layer sliders
                const thisLayer = data.this_layer || { black: [0.0, 0.0], white: [1.0, 1.0] };
                thisLayerSlider.blackStart = thisLayer.black[0] || 0.0;
                thisLayerSlider.blackEnd = thisLayer.black[1] || 0.0;
                thisLayerSlider.whiteStart = thisLayer.white[0] || 1.0;
                thisLayerSlider.whiteEnd = thisLayer.white[1] || 1.0;

                // Update underlying layer sliders
                const underLayer = data.underlying_layer || { black: [0.0, 0.0], white: [1.0, 1.0] };
                underLayerSlider.blackStart = underLayer.black[0] || 0.0;
                underLayerSlider.blackEnd = underLayer.black[1] || 0.0;
                underLayerSlider.whiteStart = underLayer.white[0] || 1.0;
                underLayerSlider.whiteEnd = underLayer.white[1] || 1.0;
            };

            // Sync initial data from widget to UI
            syncDataToUI(settings);

            // Listen for external widget value changes (e.g., loading saved workflows)
            let lastWidgetValue = widget.value;
            const checkWidgetValueChange = () => {
                if (widget.value !== lastWidgetValue) {
                    lastWidgetValue = widget.value;
                    try {
                        const newSettings = JSON.parse(widget.value);
                        syncDataToUI(newSettings);
                        updateSliderBackground();
                        thisLayerValues.update();
                        underLayerValues.update();
                        thisLayerSlider.updateTriangles();
                        underLayerSlider.updateTriangles();
                    } catch {
                        // If parsing fails, ignore
                    }
                }
            };

            // Check for widget value changes periodically
            setInterval(checkWidgetValueChange, 100);

            // Initial update
            updateSliderBackground();
            thisLayerValues.update();
            underLayerValues.update();
            thisLayerSlider.updateTriangles();
            underLayerSlider.updateTriangles();

            // Add to node
            const htmlWidget = node.addDOMWidget("blend_if_editor", "div", container);
            htmlWidget.computeSize = function (width) {
                return [width, 310];
            };

            node.setSize([
                Math.max(node.size[0], 350),
                Math.max(node.size[1], node.size[1] + 340)
            ]);

            return result;
        };

        // Add helper methods to prototype
        nodeType.prototype.createBlendIfSlider = function(prefix) {
            const container = document.createElement("div");
            container.className = "rc-blend-if-slider-container";

            const track = document.createElement("div");
            track.className = "rc-blend-if-slider-track";
            container.appendChild(track);

            const slider = {
                container,
                track,
                blackStart: 0.0,
                blackEnd: 0.0,
                whiteStart: 1.0,
                whiteEnd: 1.0,
                triangles: {},
                dragging: null,
                onUpdate: null
            };

            // Create triangles
            const createTriangle = (type, position) => {
                const triangle = document.createElement("div");
                triangle.className = `rc-blend-if-triangle ${type}`;
                triangle.style.left = (position * 100) + "%";
                container.appendChild(triangle);
                return triangle;
            };

            slider.triangles.blackStart = createTriangle("black", 0);
            slider.triangles.blackEnd = createTriangle("black", 0);
            slider.triangles.whiteStart = createTriangle("white", 1);
            slider.triangles.whiteEnd = createTriangle("white", 1);

            slider.updateTriangles = () => {
                // Update visual styles for split triangles
                const blackSplit = Math.abs(slider.blackStart - slider.blackEnd) > 0.001;
                const whiteSplit = Math.abs(slider.whiteStart - slider.whiteEnd) > 0.001;

                // Update positions and styles for black triangles
                if (blackSplit) {
                    // When split, position each triangle at its own location
                    slider.triangles.blackStart.style.left = (slider.blackStart * 100) + "%";
                    slider.triangles.blackEnd.style.left = (slider.blackEnd * 100) + "%";
                    slider.triangles.blackStart.className = `rc-blend-if-triangle black split-left`;
                    slider.triangles.blackEnd.className = `rc-blend-if-triangle black split-right`;
                    slider.triangles.blackStart.style.display = 'block';
                    slider.triangles.blackEnd.style.display = 'block';
                } else {
                    // When not split, position both at the same location but only show one
                    const pos = (slider.blackStart * 100) + "%";
                    slider.triangles.blackStart.style.left = pos;
                    slider.triangles.blackEnd.style.left = pos;
                    slider.triangles.blackStart.className = `rc-blend-if-triangle black`;
                    slider.triangles.blackEnd.className = `rc-blend-if-triangle black`;
                    slider.triangles.blackStart.style.display = 'block';
                    slider.triangles.blackEnd.style.display = 'none';
                }

                // Update positions and styles for white triangles
                if (whiteSplit) {
                    // When split, position each triangle at its own location
                    slider.triangles.whiteStart.style.left = (slider.whiteStart * 100) + "%";
                    slider.triangles.whiteEnd.style.left = (slider.whiteEnd * 100) + "%";
                    slider.triangles.whiteStart.className = `rc-blend-if-triangle white split-left`;
                    slider.triangles.whiteEnd.className = `rc-blend-if-triangle white split-right`;
                    slider.triangles.whiteStart.style.display = 'block';
                    slider.triangles.whiteEnd.style.display = 'block';
                } else {
                    // When not split, position both at the same location but only show one
                    const pos = (slider.whiteStart * 100) + "%";
                    slider.triangles.whiteStart.style.left = pos;
                    slider.triangles.whiteEnd.style.left = pos;
                    slider.triangles.whiteStart.className = `rc-blend-if-triangle white`;
                    slider.triangles.whiteEnd.className = `rc-blend-if-triangle white`;
                    slider.triangles.whiteStart.style.display = 'block';
                    slider.triangles.whiteEnd.style.display = 'none';
                }
            };

            slider.reset = () => {
                slider.blackStart = 0.0;
                slider.blackEnd = 0.0;
                slider.whiteStart = 1.0;
                slider.whiteEnd = 1.0;
                slider.updateTriangles();
            };

            // Mouse handling
            let dragStart = null;
            let dragTriangle = null;
            let altPressed = false;

            // Store update functions for later reference
            let updateThisLayerValues = null;
            let updateUnderLayerValues = null;

            const getTriangleFromElement = (element) => {
                if (element === slider.triangles.blackStart) return 'blackStart';
                if (element === slider.triangles.blackEnd) return 'blackEnd';
                if (element === slider.triangles.whiteStart) return 'whiteStart';
                if (element === slider.triangles.whiteEnd) return 'whiteEnd';
                return null;
            };

            const handleMouseDown = (e) => {
                e.preventDefault();
                const rect = container.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width;
                const triangleType = getTriangleFromElement(e.target);

                if (triangleType) {
                    dragTriangle = triangleType;
                    dragStart = { x, originalValue: slider[triangleType] };
                    altPressed = e.altKey;

                    // Alt+drag handling will be done in mousemove based on direction

                    // Set active state
                    Object.values(slider.triangles).forEach(t => t.classList.remove('active'));
                    e.target.classList.add('active');
                }
            };

            const handleMouseMove = (e) => {
                if (!dragTriangle || !dragStart) return;

                const rect = container.getBoundingClientRect();
                const x = clamp((e.clientX - rect.left) / rect.width, 0, 1);

                if (altPressed) {
                    // Alt+drag: auto-split and choose correct half based on drag direction
                    if (dragTriangle === 'blackStart' || dragTriangle === 'blackEnd') {
                        const currentCenter = (slider.blackStart + slider.blackEnd) / 2;
                        const wasSplit = Math.abs(slider.blackStart - slider.blackEnd) > 0.001;

                        if (!wasSplit) {
                            // Not split yet - determine which half to drag based on direction
                            if (x > currentCenter) {
                                // Moving right - drag the end (right half)
                                slider.blackEnd = clamp(x, currentCenter, 1);
                                dragTriangle = 'blackEnd';
                            } else {
                                // Moving left - drag the start (left half)
                                slider.blackStart = clamp(x, 0, currentCenter);
                                dragTriangle = 'blackStart';
                            }
                        } else {
                            // Already split - move the specific triangle
                            if (dragTriangle === 'blackStart') {
                                slider.blackStart = clamp(x, 0, slider.blackEnd);
                            } else {
                                slider.blackEnd = clamp(x, slider.blackStart, 1);
                            }
                        }
                    } else {
                        const currentCenter = (slider.whiteStart + slider.whiteEnd) / 2;
                        const wasSplit = Math.abs(slider.whiteStart - slider.whiteEnd) > 0.001;

                        if (!wasSplit) {
                            // Not split yet - determine which half to drag based on direction
                            if (x > currentCenter) {
                                // Moving right - drag the end (right half)
                                slider.whiteEnd = clamp(x, currentCenter, 1);
                                dragTriangle = 'whiteEnd';
                            } else {
                                // Moving left - drag the start (left half)
                                slider.whiteStart = clamp(x, 0, currentCenter);
                                dragTriangle = 'whiteStart';
                            }
                        } else {
                            // Already split - move the specific triangle
                            if (dragTriangle === 'whiteStart') {
                                slider.whiteStart = clamp(x, 0, slider.whiteEnd);
                            } else {
                                slider.whiteEnd = clamp(x, slider.whiteStart, 1);
                            }
                        }
                    }
                } else {
                    // Check if triangles are currently split
                    const blackSplit = Math.abs(slider.blackStart - slider.blackEnd) > 0.001;
                    const whiteSplit = Math.abs(slider.whiteStart - slider.whiteEnd) > 0.001;

                    if (dragTriangle === 'blackStart' || dragTriangle === 'blackEnd') {
                        if (blackSplit) {
                            // If already split, move individual triangle
                            if (dragTriangle === 'blackStart') {
                                slider.blackStart = clamp(x, 0, slider.blackEnd);
                            } else {
                                slider.blackEnd = clamp(x, slider.blackStart, 1);
                            }
                        } else {
                            // If not split, move both together
                            slider.blackStart = slider.blackEnd = clamp(x, 0, Math.min(slider.whiteStart, slider.whiteEnd));
                        }
                    } else {
                        if (whiteSplit) {
                            // If already split, move individual triangle
                            if (dragTriangle === 'whiteStart') {
                                slider.whiteStart = clamp(x, 0, slider.whiteEnd);
                            } else {
                                slider.whiteEnd = clamp(x, slider.whiteStart, 1);
                            }
                        } else {
                            // If not split, move both together
                            slider.whiteStart = slider.whiteEnd = clamp(x, Math.max(slider.blackStart, slider.blackEnd), 1);
                        }
                    }
                }

                slider.updateTriangles();
                // Update value displays during dragging
                if (slider.updateValues) {
                    slider.updateValues();
                }
                if (slider.onUpdate) slider.onUpdate();
            };

            const handleMouseUp = () => {
                if (dragTriangle) {
                    Object.values(slider.triangles).forEach(t => t.classList.remove('active'));
                }
                dragTriangle = null;
                dragStart = null;
            };

            // Event listeners
            container.addEventListener("mousedown", handleMouseDown);
            document.addEventListener("mousemove", handleMouseMove);
            document.addEventListener("mouseup", handleMouseUp);

            return slider;
        };

        nodeType.prototype.createValueInputs = function(prefix) {
            const container = document.createElement("div");
            container.className = "rc-blend-if-values";

            const createInput = (label) => {
                const input = document.createElement("input");
                input.type = "number";
                input.min = "0";
                input.max = "255";
                input.step = "1";
                input.className = "rc-blend-if-value";
                input.title = label;
                return input;
            };

            const blackStartInput = createInput(translate("BlackStart", "Black Start"));
            const blackEndInput = createInput(translate("BlackEnd", "Black End"));
            const whiteStartInput = createInput(translate("WhiteStart", "White Start"));
            const whiteEndInput = createInput(translate("WhiteEnd", "White End"));

            container.appendChild(blackStartInput);
            container.appendChild(blackEndInput);
            container.appendChild(whiteStartInput);
            container.appendChild(whiteEndInput);

            return {
                container,
                blackStartInput,
                blackEndInput,
                whiteStartInput,
                whiteEndInput
            };
        };
    }
});