import { app } from "../../scripts/app.js";

const TEXT = {
    InputLevels: { en: "Input Levels", zh: "输入色阶" },
    OutputLevels: { en: "Output Levels", zh: "输出色阶" },
    Shadow: { en: "Shadow", zh: "阴影" },
    Midtone: { en: "Midtone", zh: "中间调" },
    Highlight: { en: "Highlight", zh: "高光" },
    Black: { en: "Black", zh: "黑场" },
    White: { en: "White", zh: "白场" }
};

const getLocale = () => {
    try {
        const stored = localStorage.getItem("Comfy.Settings.Locale");
        if (stored) return stored.toLowerCase().startsWith("zh") ? "zh" : "en";
    } catch (_) {}
    return navigator?.language?.toLowerCase().startsWith("zh") ? "zh" : "en";
};

const t = (key) => TEXT[key]?.[getLocale()] || TEXT[key]?.en || key;

const safeParseJSON = (value) => {
    if (!value) return null;
    try {
        const parsed = JSON.parse(value);
        return typeof parsed === "object" && parsed !== null ? parsed : null;
    } catch (_) {
        return null;
    }
};

const hideWidget = (widget) => {
    if (!widget || widget._rcHidden) return;
    widget._rcHidden = true;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const markDirty = (node) => {
    node.widgets_changed = true;
    node?.graph?.setDirtyCanvas(true, true);
    app?.graph?.setDirtyCanvas(true, true);
};

const formatValue = (value, digits = 2) => (Math.round(value * 100) / 100).toFixed(digits);

const updateGradientColors = (state, channel) => {
    if (!state?.inputTrack || !state?.outputTrack) return;

    const inputGradient = state.inputTrack.querySelector('.rc-levels-gradient');
    const outputGradient = state.outputTrack.querySelector('.rc-output-gradient');

    let gradientCSS;
    switch (channel) {
        case "Red":
            gradientCSS = "linear-gradient(90deg, #000, #f00)";
            break;
        case "Green":
            gradientCSS = "linear-gradient(90deg, #000, #0f0)";
            break;
        case "Blue":
            gradientCSS = "linear-gradient(90deg, #000, #00f)";
            break;
        default: // RGB
            gradientCSS = "linear-gradient(90deg, #000, #fff)";
            break;
    }

    if (inputGradient) {
        inputGradient.style.background = gradientCSS;
    }
    if (outputGradient) {
        outputGradient.style.background = gradientCSS;
    }
};

const GAMMA_MIN = 0.1;
const GAMMA_MAX = 3.0;
const TRACK_PAD_LEFT = 10;
const TRACK_PAD_RIGHT = 10;

const sliderFromGamma = (gamma) => {
    gamma = clamp(gamma, GAMMA_MIN, GAMMA_MAX);
    if (gamma >= 1.0) {
        const span = GAMMA_MAX - 1.0;
        return 0.5 + (gamma - 1.0) / (span || 1e-6) * 0.5;
    }
    const span = 1.0 - GAMMA_MIN;
    return 0.5 - (1.0 - gamma) / (span || 1e-6) * 0.5;
};

const gammaFromSlider = (value) => {
    value = clamp(value, 0, 1);
    if (value >= 0.5) {
        const span = GAMMA_MAX - 1.0;
        return 1.0 + span * ((value - 0.5) / 0.5);
    }
    const span = 1.0 - GAMMA_MIN;
    return 1.0 - span * ((0.5 - value) / 0.5);
};

const computeTrackMetrics = (track) => {
    const rect = track.getBoundingClientRect();
    const trackWidth = rect.width;
    const gradientWidth = trackWidth - TRACK_PAD_LEFT - TRACK_PAD_RIGHT;
    return { rect, trackWidth, gradientWidth };
};

const fractionToLeft = (metrics, fraction) => {
    // Map fraction (0-1 on gradient) to percentage (on track)
    // Gradient starts at TRACK_PAD_LEFT pixels and has gradientWidth
    const percentage = (TRACK_PAD_LEFT + fraction * metrics.gradientWidth) / metrics.trackWidth * 100;
    return `${percentage}%`;
};

const writeStateJSON = (node) => {
    const state = node.__rcLevelsState;
    if (!state?.stateWidget || state.suspendStateWrite) return;
    const payload = {
        channel: state.currentChannel || "RGB",
        values: {
            input_black: state.inputBlack || 0,
            input_white: state.inputWhite || 1,
            gamma: state.gamma || 1,
            output_black: state.outputBlack || 0,
            output_white: state.outputWhite || 1
        }
    };
    const serialized = JSON.stringify(payload);
    if (state.stateWidget.value !== serialized) {
        state.stateWidget.value = serialized;
        markDirty(node);
    }
};

const updateNumericInputs = (state) => {
    if (!state?.inputs) return;
    const applyValue = (input, value, digits = 2) => {
        if (!input) return;
        if (document.activeElement === input) return;
        input.value = formatValue(value, digits);
    };
    applyValue(state.inputs.inputBlack, state.inputBlack);
    applyValue(state.inputs.inputWhite, state.inputWhite);
    applyValue(state.inputs.gamma, state.gamma, 2);
    applyValue(state.inputs.outputBlack, state.outputBlack);
    applyValue(state.inputs.outputWhite, state.outputWhite);
};

const updateHandles = (node) => {
    const state = node.__rcLevelsState;
    if (!state) return;

    // Read from levels_state JSON
    const stateWidget = node.widgets?.find(w => w.name === "levels_state");
    const parsed = safeParseJSON(stateWidget?.value);
    const values = parsed?.values || {};
    const channel = parsed?.channel || "RGB";

    const inputBlack = state.inputBlack = clamp(values.input_black || 0, 0, 0.95);
    const inputWhite = state.inputWhite = clamp(values.input_white || 1, inputBlack + 0.01, 1);
    const gamma = state.gamma = clamp(values.gamma || 1, GAMMA_MIN, GAMMA_MAX);
    const outputBlack = state.outputBlack = clamp(values.output_black || 0, 0, 0.95);
    const outputWhite = state.outputWhite = clamp(values.output_white || 1, outputBlack + 0.01, 1);
    state.currentChannel = channel;

    // Sync channel selector
    if (state.channelSelect && state.channelSelect.value !== channel) {
        state.channelSelect.value = channel;
    }

    // Update gradient colors based on channel
    updateGradientColors(state, channel);

    const midNormalized = sliderFromGamma(gamma);
    const midValue = clamp(inputBlack + (inputWhite - inputBlack) * midNormalized, inputBlack + 0.01, inputWhite - 0.01);

    const inputMetrics = computeTrackMetrics(state.inputTrack);
    const outputMetrics = computeTrackMetrics(state.outputTrack);

    state.handles.input.black.style.left = fractionToLeft(inputMetrics, inputBlack);
    state.handles.input.white.style.left = fractionToLeft(inputMetrics, inputWhite);
    state.handles.input.mid.style.left = fractionToLeft(inputMetrics, midValue);

    state.handles.output.black.style.left = fractionToLeft(outputMetrics, outputBlack);
    state.handles.output.white.style.left = fractionToLeft(outputMetrics, outputWhite);

    updateNumericInputs(state);
};

const setupDrag = (node, handle, type) => {
    const state = node.__rcLevelsState;
    const getTrack = () => type.startsWith("output") ? state.outputTrack : state.inputTrack;

    const onPointerDown = (event) => {
        event.preventDefault();
        handle.setPointerCapture(event.pointerId);

        const move = (ev) => {
            const track = getTrack();
            const metrics = computeTrackMetrics(track);

            // Calculate ratio relative to gradient area
            const x = ev.clientX - metrics.rect.left;
            const ratio = clamp((x - TRACK_PAD_LEFT) / metrics.gradientWidth, 0, 1);
            if (type === "input-black") {
                const maxVal = state.inputWhite - 0.01;
                state.inputBlack = clamp(ratio, 0, maxVal);
            } else if (type === "input-white") {
                const minVal = state.inputBlack + 0.01;
                state.inputWhite = clamp(ratio, minVal, 1);
            } else if (type === "input-mid") {
                const minVal = state.inputBlack + 0.01;
                const maxVal = state.inputWhite - 0.01;
                const clamped = clamp(ratio, minVal, maxVal);
                const normalized = clamp(
                    (clamped - state.inputBlack) / Math.max(state.inputWhite - state.inputBlack, 0.0001),
                    0, 1
                );
                const gamma = gammaFromSlider(normalized);
                state.gamma = parseFloat(gamma.toFixed(3));
            } else if (type === "output-black") {
                const maxVal = state.outputWhite - 0.01;
                state.outputBlack = clamp(ratio, 0, maxVal);
            } else if (type === "output-white") {
                const minVal = state.outputBlack + 0.01;
                state.outputWhite = clamp(ratio, minVal, 1);
            }
            writeStateJSON(node);
            updateHandles(node);
        };

        const up = (ev) => {
            handle.releasePointerCapture(ev.pointerId);
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
        };

        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up, { once: true });
    };

    handle.addEventListener("pointerdown", onPointerDown);
};

const buildHandle = (className) => {
    const div = document.createElement("div");
    div.className = `rc-levels-handle ${className}`;
    return div;
};

const createValueBlock = (label, digits = 2) => {
    const block = document.createElement("div");
    block.className = "rc-levels-block";
    const labelSpan = document.createElement("span");
    labelSpan.textContent = label;
    const input = document.createElement("input");
    input.type = "number";
    input.step = digits === 2 ? "0.01" : "0.1";
    input.className = "rc-levels-input";
    block.appendChild(labelSpan);
    block.appendChild(input);
    return { block, input };
};

const bindNumericInput = (node, input, getter, setter) => {
    if (!input) return;
    const commit = () => {
        let value = parseFloat(input.value);
        if (!Number.isFinite(value)) {
            input.value = formatValue(getter());
            return;
        }
        setter(value);
        requestAnimationFrame(() => updateHandles(node));
    };
    input.addEventListener("change", commit);
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") {
            ev.preventDefault();
            commit();
        }
    });
};

const setupLevelsUI = (node) => {
    // Hide the levels_state widget (JSON data managed by UI)
    const stateWidget = node.widgets?.find(w => w.name === "levels_state");
    if (stateWidget) {
        hideWidget(stateWidget);
    }

    const container = document.createElement("div");
    container.className = "rc-levels-panel";
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.boxSizing = "border-box";

    // Channel selector
    const channelRow = document.createElement("div");
    channelRow.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:8px;";

    const channelLabel = document.createElement("span");
    channelLabel.textContent = "Channel:";
    channelLabel.style.cssText = "font-size:12px;color:#cbd6ff;text-transform:uppercase;letter-spacing:0.8px;min-width:60px;";

    const channelSelect = document.createElement("select");
    channelSelect.className = "rc-levels-input";
    channelSelect.style.flex = "1";
    ["RGB", "Red", "Green", "Blue"].forEach(ch => {
        const opt = document.createElement("option");
        opt.value = ch;
        opt.textContent = ch;
        channelSelect.appendChild(opt);
    });

    channelSelect.addEventListener("change", () => {
        const state = node.__rcLevelsState;
        if (state) {
            state.currentChannel = channelSelect.value;
            writeStateJSON(node);
            updateHandles(node);
        }
    });

    channelRow.appendChild(channelLabel);
    channelRow.appendChild(channelSelect);
    container.appendChild(channelRow);

    const inputSection = document.createElement("div");
    inputSection.className = "rc-levels-section";

    const inputHeader = document.createElement("div");
    inputHeader.className = "rc-levels-title";
    inputHeader.textContent = t("InputLevels");

    const inputTrack = document.createElement("div");
    inputTrack.className = "rc-levels-track";
    inputTrack.innerHTML = `<div class="rc-levels-gradient"></div>`;

    const handleBlack = buildHandle("handle-black");
    const handleMid = buildHandle("handle-mid");
    const handleWhite = buildHandle("handle-white");

    inputTrack.appendChild(handleBlack);
    inputTrack.appendChild(handleMid);
    inputTrack.appendChild(handleWhite);

    const inputValues = document.createElement("div");
    inputValues.className = "rc-levels-grid rc-levels-grid-three";
    const shadowRow = createValueBlock(t("Shadow"));
    const midRow = createValueBlock(t("Midtone"), 2);
    const highlightRow = createValueBlock(t("Highlight"));
    inputValues.appendChild(shadowRow.block);
    inputValues.appendChild(midRow.block);
    inputValues.appendChild(highlightRow.block);

    inputSection.appendChild(inputHeader);
    inputSection.appendChild(inputTrack);
    inputSection.appendChild(inputValues);

    const outputSection = document.createElement("div");
    outputSection.className = "rc-levels-section";

    const outputHeader = document.createElement("div");
    outputHeader.className = "rc-levels-title";
    outputHeader.textContent = t("OutputLevels");

    const outputTrack = document.createElement("div");
    outputTrack.className = "rc-levels-track rc-output-track";
    outputTrack.innerHTML = `<div class="rc-output-gradient"></div>`;

    const outputBlackHandle = buildHandle("handle-output-black");
    const outputWhiteHandle = buildHandle("handle-output-white");
    outputTrack.appendChild(outputBlackHandle);
    outputTrack.appendChild(outputWhiteHandle);

    const outputValues = document.createElement("div");
    outputValues.className = "rc-levels-grid rc-levels-grid-two";
    const outBlackRow = createValueBlock(t("Black"));
    const outWhiteRow = createValueBlock(t("White"));
    outputValues.appendChild(outBlackRow.block);
    outputValues.appendChild(outWhiteRow.block);

    outputSection.appendChild(outputHeader);
    outputSection.appendChild(outputTrack);
    outputSection.appendChild(outputValues);

    container.appendChild(inputSection);
    container.appendChild(outputSection);

    node.addDOMWidget("rc_levels_ui", "div", container, {
        serialize: false,
        hideOnZoom: false
    });
    const domWrapper = container.parentElement;
    if (domWrapper && domWrapper.classList?.contains("dom-widget")) {
        domWrapper.style.width = "100%";
        domWrapper.style.height = "100%";
    }

    const state = {
        inputTrack,
        outputTrack,
        stateWidget,
        channelSelect,
        currentChannel: "RGB",
        handles: {
            input: {
                black: handleBlack,
                mid: handleMid,
                white: handleWhite
            },
            output: {
                black: outputBlackHandle,
                white: outputWhiteHandle
            }
        },
        inputs: {
            inputBlack: shadowRow.input,
            gamma: midRow.input,
            inputWhite: highlightRow.input,
            outputBlack: outBlackRow.input,
            outputWhite: outWhiteRow.input
        },
        suspendStateWrite: false
    };

    node.__rcLevelsState = state;

    setupDrag(node, handleBlack, "input-black");
    setupDrag(node, handleMid, "input-mid");
    setupDrag(node, handleWhite, "input-white");
    setupDrag(node, outputBlackHandle, "output-black");
    setupDrag(node, outputWhiteHandle, "output-white");

    bindNumericInput(node, shadowRow.input, () => state.inputBlack, (value) => {
        const maxVal = state.inputWhite - 0.01;
        state.inputBlack = clamp(value, 0, maxVal);
        writeStateJSON(node);
    });
    bindNumericInput(node, midRow.input, () => state.gamma, (value) => {
        state.gamma = clamp(value, GAMMA_MIN, GAMMA_MAX);
        writeStateJSON(node);
    });
    bindNumericInput(node, highlightRow.input, () => state.inputWhite, (value) => {
        const minVal = state.inputBlack + 0.01;
        state.inputWhite = clamp(value, minVal, 1);
        writeStateJSON(node);
    });
    bindNumericInput(node, outBlackRow.input, () => state.outputBlack, (value) => {
        const maxVal = state.outputWhite - 0.01;
        state.outputBlack = clamp(value, 0, maxVal);
        writeStateJSON(node);
    });
    bindNumericInput(node, outWhiteRow.input, () => state.outputWhite, (value) => {
        const minVal = state.outputBlack + 0.01;
        state.outputWhite = clamp(value, minVal, 1);
        writeStateJSON(node);
    });

    const originalOnWidgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function(name, value, widget) {
        const res = originalOnWidgetChanged?.apply(this, arguments);
        if (name === "levels_state") {
            requestAnimationFrame(() => updateHandles(node));
        }
        return res;
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function() {
        node.__rcLevelsState = null;
        return onRemoved?.apply(this, arguments);
    };

    if (!node.__rcLevelsSizeInit) {
        const MIN_WIDTH = 380;
        const MIN_HEIGHT = 500;
        const originalSetSize = node.setSize ? node.setSize.bind(node) : null;
        node.setSize = function(size = []) {
            const target = [
                Math.max(size?.[0] ?? MIN_WIDTH, MIN_WIDTH),
                Math.max(size?.[1] ?? MIN_HEIGHT, MIN_HEIGHT)
            ];
            if (originalSetSize) {
                originalSetSize(target);
            } else {
                this.size = target;
            }
            // Update handle positions when node is resized
            requestAnimationFrame(() => updateHandles(node));
        };
        node.setSize([MIN_WIDTH, MIN_HEIGHT]);
        node.__rcLevelsSizeInit = true;
    }

    // Also listen to onResize
    const originalOnResize = node.onResize;
    node.onResize = function(size) {
        const res = originalOnResize?.apply(this, arguments);
        requestAnimationFrame(() => updateHandles(node));
        return res;
    };

    // Initialize UI from state
    requestAnimationFrame(() => updateHandles(node));
};

app.registerExtension({
    name: "RC.LevelsUI",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RC_LevelsAdjust") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const res = onNodeCreated?.apply(this, arguments);
            setupLevelsUI(this);
            return res;
        };
    }
});

const style = document.createElement("style");
style.textContent = `
.rc-levels-panel {
    background: #1b1b1b;
    border: 1px solid #2f2f2f;
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.rc-levels-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.rc-levels-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #cbd6ff;
}
.rc-levels-track {
    position: relative;
    height: 72px;
    background: #141414;
    border-radius: 6px;
    border: 1px solid #333;
    overflow: visible;
}
.rc-levels-gradient,
.rc-output-gradient {
    position: absolute;
    left: 10px;
    right: 10px;
    top: 10px;
    bottom: 22px;
    border-radius: 4px;
    background: linear-gradient(90deg, #000, #fff);
    overflow: hidden;
}
.rc-levels-handle {
    position: absolute;
    width: 16px;
    height: 12px;
    background: #c2c5cd;
    border: 1px solid #6a6a6a;
    border-radius: 2px;
    clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
    cursor: pointer;
    bottom: 6px;
    transform: translateX(-50%);
    box-shadow: 0 1px 2px rgba(0,0,0,0.45);
}
.rc-levels-handle:hover {
    background: #e0e4eb;
    border-color: #9aa0a8;
}
.rc-levels-grid {
    display: flex;
    gap: 10px;
    width: 100%;
}
.rc-levels-grid-three .rc-levels-block {
    flex: 1 1 calc(33.33% - 7px);
}
.rc-levels-grid-two .rc-levels-block {
    flex: 1 1 50%;
}
.rc-levels-block {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: #dcdcdc;
}
.rc-levels-block span {
    color: #aeb7c8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.rc-levels-input {
    width: 100%;
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid #3c3c3c;
    background: #111;
    color: #f2f2f2;
    font-size: 12px;
}
.rc-levels-input:focus {
    outline: none;
    border-color: #5f8cff;
    box-shadow: 0 0 0 1px rgba(95,140,255,0.4);
}
`;
document.head.appendChild(style);
