import { app } from "../../scripts/app.js";

const styles = `
.rc-curves-editor {
    background: #1e1e1e;
    padding: 8px;
    border-radius: 4px;
    margin: 2px 0;
    box-sizing: border-box;
}

.rc-curves-channel-row {
    display: flex;
    gap: 4px;
    margin-bottom: 8px;
}

.rc-curves-channel {
    flex: 1;
    background: #2c2c2c;
    color: #cfcfcf;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 4px 0;
    text-align: center;
    font-size: 11px;
    cursor: pointer;
    user-select: none;
    transition: background 0.15s ease;
}

.rc-curves-channel.active {
    background: #4a9eff;
    border-color: #4a9eff;
    color: #ffffff;
}

.rc-curves-channel.modified {
    position: relative;
}

.rc-curves-channel.modified::after {
    content: "●";
    position: absolute;
    top: 2px;
    right: 4px;
    font-size: 8px;
    color: #ffa500;
}

.rc-curves-canvas-wrapper {
    position: relative;
    width: 100%;
    height: 320px;
    background: #141414;
    border: 1px solid #2b2b2b;
    border-radius: 4px;
    overflow: hidden;
}

.rc-curves-canvas {
    width: 100%;
    height: 100%;
    display: block;
}

.rc-curves-info-row,
.rc-curves-actions-row,
.rc-curves-presets-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
}

.rc-curves-label {
    color: #cfcfcf;
    font-size: 11px;
    min-width: 46px;
    flex-shrink: 0;
}

.rc-curves-number {
    background: #242424;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    color: #e0e0e0;
    padding: 3px 6px;
    width: 60px;
    box-sizing: border-box;
    font-size: 11px;
}

.rc-curves-btn {
    background: #3a3a3a;
    border: 1px solid #4a4a4a;
    color: #dcdcdc;
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s ease;
    flex-shrink: 0;
}

.rc-curves-btn.primary {
    background: #4a9eff;
    border-color: #4a9eff;
    color: #ffffff;
}

.rc-curves-btn.danger {
    background: #d24c4c;
    border-color: #d24c4c;
    color: #ffffff;
}

.rc-curves-btn:disabled {
    opacity: 0.5;
    cursor: default;
}

.rc-curves-btn:not(:disabled):hover {
    background: #545454;
}

.rc-curves-btn.primary:not(:disabled):hover {
    background: #3a8eef;
}

.rc-curves-btn.danger:not(:disabled):hover {
    background: #c53b3b;
}

.rc-curves-select {
    background: #242424;
    border: 1px solid #3a3a3a;
    color: #e0e0e0;
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 11px;
}

.rc-curves-hint {
    color: #888;
    font-size: 10px;
    margin-left: auto;
}
`;

if (!document.getElementById("rc-curves-styles")) {
    const styleEl = document.createElement("style");
    styleEl.id = "rc-curves-styles";
    styleEl.textContent = styles;
    document.head.appendChild(styleEl);
}

const CHANNELS = [
    { id: "RGB", key: "RGB", color: "#f0f0f0" },
    { id: "R", key: "Red", color: "#ff6b6b" },
    { id: "G", key: "Green", color: "#7bd88f" },
    { id: "B", key: "Blue", color: "#6fa8ff" },
    { id: "A", key: "Alpha", color: "#dcdcdc" }
];

const PRESETS = [
    {
        id: "linear",
        key: "Linear",
        points: [
            { x: 0.0, y: 0.0 },
            { x: 1.0, y: 1.0 }
        ]
    },
    {
        id: "color_negative",
        key: "ColorNegative",
        channels: {
            R: [{x: 0.0, y: 1.0}, {x: 0.1304, y: 1.0}, {x: 0.4667, y: 0.498}, {x: 0.7255, y: 0.0}, {x: 1.0, y: 0.0}],
            G: [{x: 0.0, y: 1.0}, {x: 0.1098, y: 1.0}, {x: 0.302, y: 0.498}, {x: 0.5176, y: 0.0}, {x: 1.0, y: 0.0}],
            B: [{x: 0.0, y: 1.0}, {x: 0.098, y: 1.0}, {x: 0.2353, y: 0.498}, {x: 0.4235, y: 0.0}, {x: 1.0, y: 0.0}]
        }
    },
    {
        id: "cross_process",
        key: "CrossProcess",
        channels: {
            R: [{x: 0.0, y: 0.0}, {x: 0.251, y: 0.1569}, {x: 0.502, y: 0.4902}, {x: 0.6863, y: 0.7451}, {x: 1.0, y: 1.0}],
            G: [{x: 0.0, y: 0.0}, {x: 0.251, y: 0.1882}, {x: 0.3804, y: 0.502}, {x: 0.7451, y: 0.8157}, {x: 1.0, y: 0.8157}],
            B: [{x: 0.0, y: 0.0}, {x: 0.2314, y: 0.0941}, {x: 0.7098, y: 0.8745}, {x: 1.0, y: 1.0}]
        }
    },
    {
        id: "darker",
        key: "Darker",
        points: [
            {x: 0.0, y: 0.0},
            {x: 0.5098, y: 0.3961},
            {x: 1.0, y: 1.0}
        ]
    },
    {
        id: "increase_contrast",
        key: "IncreaseContrast",
        points: [
            {x: 0.0, y: 0.0},
            {x: 0.149, y: 0.0667},
            {x: 0.8314, y: 0.9059},
            {x: 0.9059, y: 0.9804},
            {x: 1.0, y: 1.0}
        ]
    },
    {
        id: "lighter",
        key: "Lighter",
        points: [
            {x: 0.0, y: 0.0},
            {x: 0.4039, y: 0.4902},
            {x: 1.0, y: 1.0}
        ]
    },
    {
        id: "linear_contrast",
        key: "LinearContrast",
        points: [
            {x: 0.0, y: 0.0},
            {x: 0.3059, y: 0.2863},
            {x: 0.6941, y: 0.7137},
            {x: 1.0, y: 1.0}
        ]
    },
    {
        id: "medium_contrast",
        key: "MediumContrast",
        points: [
            {x: 0.0, y: 0.0},
            {x: 0.2863, y: 0.2196},
            {x: 0.6392, y: 0.6431},
            {x: 1.0, y: 1.0}
        ]
    },
    {
        id: "strong_contrast",
        key: "StrongContrast",
        points: [
            {x: 0.0, y: 0.0},
            {x: 0.302, y: 0.1961},
            {x: 0.5922, y: 0.6},
            {x: 0.6863, y: 0.7373},
            {x: 1.0, y: 1.0}
        ]
    },
    {
        id: "negative",
        key: "Negative",
        points: [
            { x: 0.0, y: 1.0 },
            { x: 1.0, y: 0.0 }
        ]
    }
];

const MAX_POINTS = 16;
const POINT_HIT_RADIUS = 12;
const CURVE_SAMPLES = 256;
const TANGENT_TENSION = 0.0;
const TANGENT_LIMIT = 120;
const EPSILON = 1e-6;

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

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
        RGB: { en: "RGB", zh: "RGB" },
        Red: { en: "Red", zh: "红" },
        Green: { en: "Green", zh: "绿" },
        Blue: { en: "Blue", zh: "蓝" },
        Alpha: { en: "Alpha", zh: "Alpha" },
        Input: { en: "Input", zh: "输入" },
        Output: { en: "Output", zh: "输出" },
        DeletePoint: { en: "Delete", zh: "删除" },
        ResetChannel: { en: "Reset Channel", zh: "重置通道" },
        ResetAll: { en: "Reset All", zh: "全部重置" },
        Presets: { en: "Presets", zh: "预设" },
        Hint: { en: "Click to add, drag to adjust, double-click to delete", zh: "单击添加,拖动调整,双击删除" },
        ApplyPreset: { en: "Apply", zh: "应用" },
        AutoCurve: { en: "Auto", zh: "自动" },
        Linear: { en: "Linear", zh: "默认值" },
        ColorNegative: { en: "Color Negative (RGB)", zh: "彩色负片(RGB)" },
        CrossProcess: { en: "Cross Process (RGB)", zh: "反冲 (RGB)" },
        Darker: { en: "Darker (RGB)", zh: "较暗(RGB)" },
        IncreaseContrast: { en: "Increase Contrast (RGB)", zh: "增加对比度(RGB)" },
        Lighter: { en: "Lighter (RGB)", zh: "较亮(RGB)" },
        LinearContrast: { en: "Linear Contrast (RGB)", zh: "线性对比度(RGB)" },
        MediumContrast: { en: "Medium Contrast (RGB)", zh: "中对比度(RGB)" },
        Negative: { en: "Negative (RGB)", zh: "负片(RGB)" },
        StrongContrast: { en: "Strong Contrast (RGB)", zh: "强对比度(RGB)" }
    };

    const translations = table[key];
    if (!translations) return fallback;

    return (
        translations[lang] ||
        (lang.length > 2 ? translations[lang.slice(0, 2)] : undefined) ||
        fallback
    );
};

const clonePoints = (points) => points.map(pt => ({ x: pt.x, y: pt.y }));

const sortPoints = (points) => {
    points.sort((a, b) => a.x - b.x);
    return points;
};




const normalizePoints = (points) => {
    if (!Array.isArray(points) || points.length < 2) {
        return clonePoints(PRESETS[0].points);
    }

    const normalized = sortPoints(clonePoints(points)).map(pt => ({
        x: clamp(Number(pt.x) || 0, 0, 1),
        y: clamp(Number(pt.y) || 0, 0, 1)
    }));

    const first = normalized[0];
    if (first.x !== 0) {
        normalized[0] = { x: 0, y: first.y };
    }

    const lastIdx = normalized.length - 1;
    const last = normalized[lastIdx];
    if (last.x !== 1) {
        normalized[lastIdx] = { x: 1, y: last.y };
    }

    const deduped = [normalized[0]];
    for (let i = 1; i < normalized.length; i++) {
        const prev = deduped[deduped.length - 1];
        const current = normalized[i];
        if (Math.abs(current.x - prev.x) < 1e-6) {
            deduped[deduped.length - 1] = current;
        } else {
            deduped.push(current);
        }
    }

    if (deduped.length > MAX_POINTS) {
        const keep = [deduped[0]];
        const middle = deduped.slice(1, deduped.length - 1);
        const step = middle.length / (MAX_POINTS - 2);
        for (let i = 0; i < MAX_POINTS - 2; i++) {
            const idx = Math.min(middle.length - 1, Math.round(i * step));
            if (middle[idx]) keep.push(middle[idx]);
        }
        keep.push(deduped[deduped.length - 1]);
        return keep;
    }

    return deduped;
};

// Check if curve is modified (not linear)
const isCurveModified = (points) => {
    if (points.length !== 2) return true;
    const p0 = points[0];
    const p1 = points[1];
    return !(Math.abs(p0.x - 0) < EPSILON && 
             Math.abs(p0.y - 0) < EPSILON && 
             Math.abs(p1.x - 1) < EPSILON && 
             Math.abs(p1.y - 1) < EPSILON);
};

// Natural Cubic Spline calculation (like Photoshop)
const calculateNaturalSpline = (points) => {
    const n = points.length;
    if (n < 3) return null;

    const h = [];
    const alpha = [];
    const l = [];
    const mu = [];
    const z = [];
    const c = new Array(n).fill(0);
    const b = [];
    const d = [];

    // Calculate intervals
    for (let i = 0; i < n - 1; i++) {
        h[i] = points[i + 1].x - points[i].x;
    }

    // Calculate alpha values
    for (let i = 1; i < n - 1; i++) {
        alpha[i] = (3 / h[i]) * (points[i + 1].y - points[i].y) -
                  (3 / h[i - 1]) * (points[i].y - points[i - 1].y);
    }

    // Natural boundary conditions: second derivative = 0
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    // Solve tridiagonal matrix
    for (let i = 1; i < n - 1; i++) {
        l[i] = 2 * (points[i + 1].x - points[i - 1].x) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1;
    z[n - 1] = 0;
    c[n - 1] = 0;

    // Back substitution
    for (let j = n - 2; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (points[j + 1].y - points[j].y) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    return { b, c, d };
};

// Evaluate Natural Cubic Spline at point x
const evaluateNaturalSpline = (x, points, coeffs) => {
    const n = points.length;

    // Boundary handling
    if (x <= points[0].x) return points[0].y;
    if (x >= points[n - 1].x) return points[n - 1].y;

    // Find interval containing x
    let i = 0;
    for (i = 0; i < n - 1; i++) {
        if (x >= points[i].x && x <= points[i + 1].x) {
            break;
        }
    }

    // If no coefficients (only two points), use linear interpolation
    if (!coeffs) {
        const t = (x - points[i].x) / (points[i + 1].x - points[i].x);
        return points[i].y + t * (points[i + 1].y - points[i].y);
    }

    // Calculate relative position
    const dx = x - points[i].x;

    // Cubic spline formula: y = a + b*dx + c*dx² + d*dx³
    const a = points[i].y;
    const b = coeffs.b[i];
    const c = coeffs.c[i];
    const d = coeffs.d[i];

    return a + b * dx + c * dx * dx + d * dx * dx * dx;
};

// Natural Cubic Spline interpolation (like Photoshop)
const interpolateWithNaturalSpline = (points, x) => {
    if (points.length < 2) return 0;

    // Handle boundary cases
    if (x <= points[0].x) return points[0].y;
    if (x >= points[points.length - 1].x) return points[points.length - 1].y;

    // If only two points, use linear interpolation
    if (points.length === 2) {
        const t = (x - points[0].x) / (points[1].x - points[0].x);
        return points[0].y + t * (points[1].y - points[0].y);
    }

    // Calculate Natural Cubic Spline coefficients
    const coeffs = calculateNaturalSpline(points);

    // Evaluate spline at x
    return evaluateNaturalSpline(x, points, coeffs);
};

// Sampling function consistent with Python code
const sampleCurve = (points, sampleCount = CURVE_SAMPLES) => {
    if (points.length < 2) return [];
    const samples = [];

    // Sample entire curve using Natural Cubic Spline
    for (let i = 0; i < sampleCount; i++) {
        const x = i / (sampleCount - 1);
        const y = interpolateWithNaturalSpline(points, x);
        samples.push({
            x: x,
            y: clamp(y, 0, 1)
        });
    }

    return samples;
};

const findInsertionIndex = (points, x) => {
    let idx = points.findIndex(pt => pt.x > x);
    if (idx === -1) idx = points.length;
    return idx;
};

const clampPointX = (points, idx, candidate) => {
    if (idx <= 0) return 0;
    if (idx >= points.length - 1) return 1;
    const prev = points[idx - 1];
    const next = points[idx + 1];
    const minX = prev.x + EPSILON;
    const maxX = next.x - EPSILON;
    return clamp(candidate, minX, maxX);
};

const locatePointByPixel = (points, padding, innerWidth, innerHeight, mouseX, mouseY) => {
    const radius = POINT_HIT_RADIUS;
    const radiusSq = radius * radius;
    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        const px = padding + pt.x * innerWidth;
        const py = padding + (1 - pt.y) * innerHeight;
        const dx = mouseX - px;
        const dy = mouseY - py;
        if (dx * dx + dy * dy <= radiusSq) {
            return i;
        }
    }
    return -1;
};

app.registerExtension({
    name: "RC.CurvesAdjust",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "RC_CurvesAdjust") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const widget = this.widgets?.find(w => w.name === "curve_data");
            if (!widget) return result;

            widget.type = "converted-widget";
            widget.serializeValue = () => widget.value;

            const curves = {};
            const selectedPerChannel = {};
            let activeChannel = "RGB";

            const applyPayload = (value) => {
                let payload;
                try {
                    payload = JSON.parse(value);
                } catch (_) {
                    payload = {};
                }

                CHANNELS.forEach(ch => {
                    const channelPoints = payload?.channels?.[ch.id] || payload?.[ch.id];
                    curves[ch.id] = normalizePoints(channelPoints);
                    const points = curves[ch.id];
                    selectedPerChannel[ch.id] = clamp(payload?.selection?.[ch.id] ?? 0, 0, points.length - 1);
                });

                activeChannel = CHANNELS.find(ch => ch.id === payload?.active_channel)?.id || "RGB";
            };

            applyPayload(widget.value);

            const container = document.createElement("div");
            container.className = "rc-curves-editor";

            const channelRow = document.createElement("div");
            channelRow.className = "rc-curves-channel-row";
            container.appendChild(channelRow);

            const channelButtons = {};
            CHANNELS.forEach(ch => {
                const btn = document.createElement("button");
                btn.className = "rc-curves-channel";
                btn.textContent = translate(ch.key, ch.id);
                btn.style.borderColor = ch.color;
                btn.onclick = () => {
                    activeChannel = ch.id;
                    updateChannelButtons();
                    updateInfo();
                    draw();
                    save();
                };
                channelButtons[ch.id] = btn;
                channelRow.appendChild(btn);
            });

            const canvasWrapper = document.createElement("div");
            canvasWrapper.className = "rc-curves-canvas-wrapper";
            const canvas = document.createElement("canvas");
            canvas.className = "rc-curves-canvas";
            canvasWrapper.appendChild(canvas);
            container.appendChild(canvasWrapper);

            const ctx = canvas.getContext("2d");
            const padding = 16;

            let lastLogicalWidth = 0;
            let lastLogicalHeight = 0;

            // 同步 canvas 尺寸，确保缩放后仍然清晰
            const syncCanvasSize = () => {
                const rect = canvasWrapper.getBoundingClientRect();
                const measuredWidth = rect?.width ?? 0;
                const measuredHeight = rect?.height ?? 0;
                const logicalWidth = measuredWidth > 0 ? measuredWidth : (lastLogicalWidth || canvas.width || 1);
                const logicalHeight = measuredHeight > 0 ? measuredHeight : (lastLogicalHeight || canvas.height || 1);
                const width = Math.max(1, Math.round(logicalWidth));
                const height = Math.max(1, Math.round(logicalHeight));
                let resized = false;
                if (canvas.width !== width) {
                    canvas.width = width;
                    resized = true;
                }
                if (canvas.height !== height) {
                    canvas.height = height;
                    resized = true;
                }
                lastLogicalWidth = logicalWidth;
                lastLogicalHeight = logicalHeight;
                return { logicalWidth, logicalHeight, width, height, resized };
            };

            let scheduledFrame = null;
            const requestRender = () => {
                if (scheduledFrame !== null) return;
                scheduledFrame = requestAnimationFrame(() => {
                    scheduledFrame = null;
                    draw();
                });
            };

            // 使用ResizeObserver监听容器大小变化
            const resizeObserver = new ResizeObserver(() => {
                requestRender();
            });
            resizeObserver.observe(canvasWrapper);

            const pollIntervalMs = 150;
            const checkSizeWithTimer = () => {
                const rect = canvasWrapper.getBoundingClientRect();
                const width = rect?.width ?? 0;
                const height = rect?.height ?? 0;
                if (width <= 0 && height <= 0) return;
                if (Math.abs(width - lastLogicalWidth) > 0.5 || Math.abs(height - lastLogicalHeight) > 0.5) {
                    lastLogicalWidth = width;
                    lastLogicalHeight = height;
                    requestRender();
                }
            };
            const sizePoller = setInterval(checkSizeWithTimer, pollIntervalMs);
            checkSizeWithTimer();

            window.addEventListener("resize", requestRender);

            const infoRow = document.createElement("div");
            infoRow.className = "rc-curves-info-row";
            container.appendChild(infoRow);

            const inputLabel = document.createElement("span");
            inputLabel.className = "rc-curves-label";
            inputLabel.textContent = translate("Input", "Input") + ":";
            const inputNumber = document.createElement("input");
            inputNumber.type = "number";
            inputNumber.min = "0";
            inputNumber.max = "255";
            inputNumber.step = "1";
            inputNumber.className = "rc-curves-number";

            const outputLabel = document.createElement("span");
            outputLabel.className = "rc-curves-label";
            outputLabel.textContent = translate("Output", "Output") + ":";
            const outputNumber = document.createElement("input");
            outputNumber.type = "number";
            outputNumber.min = "0";
            outputNumber.max = "255";
            outputNumber.step = "1";
            outputNumber.className = "rc-curves-number";

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "rc-curves-btn danger";
            deleteBtn.textContent = translate("DeletePoint", "Delete");

            infoRow.appendChild(inputLabel);
            infoRow.appendChild(inputNumber);
            infoRow.appendChild(outputLabel);
            infoRow.appendChild(outputNumber);
            infoRow.appendChild(deleteBtn);

            const actionsRow = document.createElement("div");
            actionsRow.className = "rc-curves-actions-row";
            container.appendChild(actionsRow);

            const resetChannelBtn = document.createElement("button");
            resetChannelBtn.className = "rc-curves-btn";
            resetChannelBtn.textContent = translate("ResetChannel", "Reset Channel");

            const resetAllBtn = document.createElement("button");
            resetAllBtn.className = "rc-curves-btn";
            resetAllBtn.textContent = translate("ResetAll", "Reset All");

            actionsRow.appendChild(resetChannelBtn);
            actionsRow.appendChild(resetAllBtn);

            const presetsRow = document.createElement("div");
            presetsRow.className = "rc-curves-presets-row";
            container.appendChild(presetsRow);

            const presetLabel = document.createElement("span");
            presetLabel.className = "rc-curves-label";
            presetLabel.textContent = translate("Presets", "Presets") + ":";

            const presetSelect = document.createElement("select");
            presetSelect.className = "rc-curves-select";
            PRESETS.forEach(preset => {
                const option = document.createElement("option");
                option.value = preset.id;
                option.textContent = translate(preset.key, preset.id);
                presetSelect.appendChild(option);
            });

            const applyPresetBtn = document.createElement("button");
            applyPresetBtn.className = "rc-curves-btn primary";
            applyPresetBtn.textContent = translate("ApplyPreset", "Apply");

            const hint = document.createElement("span");
            hint.className = "rc-curves-hint";
            hint.textContent = translate("Hint", "Click to add, drag to adjust, double-click to delete");

            presetsRow.appendChild(presetLabel);
            presetsRow.appendChild(presetSelect);
            presetsRow.appendChild(applyPresetBtn);
            presetsRow.appendChild(hint);

            const node = this;

            const getPoints = () => curves[activeChannel];
            const getSelectedIndex = () => {
                const points = getPoints();
                const stored = selectedPerChannel[activeChannel] ?? 0;
                return clamp(stored, 0, points.length - 1);
            };

            const setSelectedIndex = (idx) => {
                selectedPerChannel[activeChannel] = idx;
                updateInfo();
                draw();
            };

            const save = () => {
                const payloadToSave = {
                    channels: {},
                    active_channel: activeChannel,
                    editor: "RC_CurvesAdjust",
                    version: 2,
                    selection: selectedPerChannel
                };

                CHANNELS.forEach(ch => {
                    payloadToSave.channels[ch.id] = curves[ch.id].map(pt => ({
                        x: parseFloat(pt.x.toFixed(4)),
                        y: parseFloat(pt.y.toFixed(4))
                    }));
                });

                const serialized = JSON.stringify(payloadToSave);
                widget.value = serialized;
                lastWidgetValue = serialized;
                if (widget.callback) {
                    widget.callback(serialized, node, appInstance);
                }
                node?.graph?.setDirtyCanvas(true, true);

            };

            const updateChannelButtons = () => {
                CHANNELS.forEach(ch => {
                    const btn = channelButtons[ch.id];
                    if (!btn) return;
                    
                    // 更新激活状态
                    if (ch.id === activeChannel) {
                        btn.classList.add("active");
                    } else {
                        btn.classList.remove("active");
                    }
                    
                    // 更新修改状态标记
                    if (isCurveModified(curves[ch.id])) {
                        btn.classList.add("modified");
                    } else {
                        btn.classList.remove("modified");
                    }
                });
            };

            function draw() {
                const metrics = syncCanvasSize();
                const displayWidth = metrics.width;
                const displayHeight = metrics.height;
                const logicalWidth = metrics.logicalWidth;
                const logicalHeight = metrics.logicalHeight;
                const widthSpan = Math.max(1, logicalWidth - padding * 2);
                const heightSpan = Math.max(1, logicalHeight - padding * 2);
                const scaleX = logicalWidth ? displayWidth / logicalWidth : 1;
                const scaleY = logicalHeight ? displayHeight / logicalHeight : 1;

                ctx.clearRect(0, 0, displayWidth, displayHeight);

                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(0, 0, displayWidth, displayHeight);

                ctx.strokeStyle = "#2c2c2c";
                ctx.lineWidth = 1;

                const top = padding * scaleY;
                const bottom = (logicalHeight - padding) * scaleY;
                const left = padding * scaleX;
                const right = (logicalWidth - padding) * scaleX;

                // 绘制网格
                const steps = 4;
                for (let i = 1; i < steps; i++) {
                    const t = i / steps;
                    const x = (padding + t * widthSpan) * scaleX;
                    const y = (padding + t * heightSpan) * scaleY;

                    ctx.beginPath();
                    ctx.moveTo(x, top);
                    ctx.lineTo(x, bottom);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(left, y);
                    ctx.lineTo(right, y);
                    ctx.stroke();
                }


                // 绘制对角线
                ctx.strokeStyle = "#3c3c3c";
                ctx.beginPath();
                ctx.moveTo(left, bottom);
                ctx.lineTo(right, top);
                ctx.stroke();

                // 绘制其他被修改通道的曲线（虚线）
                CHANNELS.forEach(ch => {
                    if (ch.id === activeChannel) return; // 跳过当前通道
                    if (!isCurveModified(curves[ch.id])) return; // 跳过未修改的通道
                    
                    const otherSamples = sampleCurve(curves[ch.id]);
                    
                    ctx.save();
                    ctx.setLineDash([4, 4]); // 虚线样式
                    ctx.strokeStyle = ch.color;
                    ctx.globalAlpha = 0.3;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    otherSamples.forEach((pt, idx) => {
                        const x = (padding + pt.x * widthSpan) * scaleX;
                        const y = (padding + (1 - pt.y) * heightSpan) * scaleY;
                        if (idx === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    });
                    ctx.stroke();
                    ctx.restore();
                });

                // 绘制RGB参考曲线（非RGB通道时）
                const points = getPoints();

                if (activeChannel !== "RGB") {
                    const referenceSamples = sampleCurve(curves.RGB);
                    ctx.beginPath();
                    ctx.strokeStyle = "rgba(255,255,255,0.25)";
                    ctx.lineWidth = 1.5;
                    referenceSamples.forEach((pt, idx) => {
                        const x = (padding + pt.x * widthSpan) * scaleX;
                        const y = (padding + (1 - pt.y) * heightSpan) * scaleY;
                        if (idx === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    });
                    ctx.stroke();
                }

                // 绘制当前通道曲线
                const samples = sampleCurve(points);
                const strokeColor = CHANNELS.find(ch => ch.id === activeChannel)?.color || "#ffffff";
                ctx.beginPath();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 2.5;
                samples.forEach((pt, idx) => {
                    const x = (padding + pt.x * widthSpan) * scaleX;
                    const y = (padding + (1 - pt.y) * heightSpan) * scaleY;
                    if (idx === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });
                ctx.stroke();

                // 绘制控制点
                const selectedIdx = getSelectedIndex();
                ctx.lineWidth = 1;
                points.forEach((pt, idx) => {
                    const x = (padding + pt.x * widthSpan) * scaleX;
                    const y = (padding + (1 - pt.y) * heightSpan) * scaleY;
                    ctx.beginPath();
                    ctx.fillStyle = idx === selectedIdx ? "#ffffff" : "#f0a124";
                    ctx.strokeStyle = "#111";
                    ctx.arc(x, y, idx === selectedIdx ? 5 : 4, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                });
            }

            const updateInfo = () => {
                const points = getPoints();
                const idx = getSelectedIndex();
                const point = points[idx];
                inputNumber.value = Math.round(point.x * 255);
                outputNumber.value = Math.round(point.y * 255);

                const removable = points.length > 2 && idx > 0 && idx < points.length - 1;
                deleteBtn.disabled = !removable;
            };

            const addPoint = (x, y, { insertIndex } = {}) => {
                const points = getPoints();
                if (points.length >= MAX_POINTS) return -1;
                const newPoint = { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };

                const index = typeof insertIndex === "number" ? clamp(insertIndex, 0, points.length) : findInsertionIndex(points, newPoint.x);
                points.splice(index, 0, newPoint);
                sortPoints(points);
                const actualIdx = points.indexOf(newPoint);
                if (actualIdx === 0) newPoint.x = 0;
                else if (actualIdx === points.length - 1) newPoint.x = 1;
                else newPoint.x = clampPointX(points, actualIdx, newPoint.x);
                setSelectedIndex(actualIdx);
                updateChannelButtons();
                save();
                return actualIdx;
            };

            const removePoint = (idx) => {
                const points = getPoints();
                if (points.length <= 2 || idx <= 0 || idx >= points.length - 1) return;
                points.splice(idx, 1);
                const newIdx = clamp(idx - 1, 0, points.length - 1);
                setSelectedIndex(newIdx);
                updateChannelButtons();
                save();
            };

            let dragging = false;
            let dragIndex = -1;

            const handleMouseDown = (event) => {
                const rect = canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                const widthSpan = rect.width - padding * 2;
                const heightSpan = rect.height - padding * 2;

                const points = getPoints();
                const found = locatePointByPixel(points, padding, widthSpan, heightSpan, x, y);
                if (found !== -1) {
                    dragging = true;
                    dragIndex = found;
                    setSelectedIndex(found);
                    return;
                }

                const nx = clamp((x - padding) / widthSpan, 0, 1);
                const nyRaw = clamp(1 - (y - padding) / heightSpan, 0, 1);

                const ny = nyRaw;

                const insertIndex = findInsertionIndex(points, nx);
                const prevIdx = insertIndex - 1;
                const nextIdx = insertIndex;
                const prev = prevIdx >= 0 ? points[prevIdx] : null;
                const next = nextIdx < points.length ? points[nextIdx] : null;

                const gapThreshold = Math.max(EPSILON, 2 / Math.max(widthSpan, 1));

                if (prev && Math.abs(nx - prev.x) <= gapThreshold) {
                    dragging = true;
                    dragIndex = prevIdx;
                    setSelectedIndex(prevIdx);
                    return;
                }

                if (next && Math.abs(next.x - nx) <= gapThreshold) {
                    dragging = true;
                    dragIndex = nextIdx;
                    setSelectedIndex(nextIdx);
                    return;
                }

                const createdIdx = addPoint(nx, ny, { insertIndex });
                if (createdIdx !== -1) {
                    dragging = true;
                    dragIndex = createdIdx;
                }
            };

            const handleMouseMove = (event) => {
                if (!dragging || dragIndex === -1) return;
                const rect = canvas.getBoundingClientRect();
                const widthSpan = rect.width - padding * 2;
                const heightSpan = rect.height - padding * 2;
                const nx = clamp((event.clientX - rect.left - padding) / widthSpan, 0, 1);
                const ny = clamp(1 - (event.clientY - rect.top - padding) / heightSpan, 0, 1);

                const points = getPoints();
                const point = points[dragIndex];
                if (!point) return;

                point.y = ny;

                // 允许首尾锚点的水平移动
                if (dragIndex === 0) {
                    // 左锚点：允许向右移动但不能超过下一个点
                    if (points.length > 1) {
                        const nextPoint = points[1];
                        point.x = clamp(nx, 0, nextPoint.x - EPSILON);
                    } else {
                        point.x = 0;
                    }
                } else if (dragIndex === points.length - 1) {
                    // 右锚点：允许向左移动但不能超过前一个点
                    if (points.length > 1) {
                        const prevPoint = points[points.length - 2];
                        point.x = clamp(nx, prevPoint.x + EPSILON, 1);
                    } else {
                        point.x = 1;
                    }
                } else {
                    point.x = clampPointX(points, dragIndex, nx);
                }

                setSelectedIndex(dragIndex);
                save();
            };

            const handleMouseUp = () => {
                if (!dragging) return;
                dragging = false;
                dragIndex = -1;
                updateChannelButtons();
                draw();
            };

            const handleDoubleClick = (event) => {
                const rect = canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                const widthSpan = rect.width - padding * 2;
                const heightSpan = rect.height - padding * 2;
                const points = getPoints();
                const found = locatePointByPixel(points, padding, widthSpan, heightSpan, x, y);
                if (found !== -1) {
                    removePoint(found);
                }
            };

            canvas.addEventListener("mousedown", handleMouseDown);
            window.addEventListener("mousemove", handleMouseMove);
            window.addEventListener("mouseup", handleMouseUp);
            canvas.addEventListener("mouseleave", handleMouseUp);
            canvas.addEventListener("dblclick", handleDoubleClick);

            const cleanup = () => {
                canvas.removeEventListener("mousedown", handleMouseDown);
                canvas.removeEventListener("mouseleave", handleMouseUp);
                canvas.removeEventListener("dblclick", handleDoubleClick);
                window.removeEventListener("mousemove", handleMouseMove);
                window.removeEventListener("mouseup", handleMouseUp);
                window.removeEventListener("resize", requestRender);
                if (scheduledFrame !== null) {
                    cancelAnimationFrame(scheduledFrame);
                    scheduledFrame = null;
                }
                clearInterval(sizePoller);
                resizeObserver.disconnect();
            };

            inputNumber.addEventListener("input", () => {
                const points = getPoints();
                const idx = getSelectedIndex();
                const point = points[idx];
                const val = clamp(parseFloat(inputNumber.value) || 0, 0, 255) / 255;

                if (idx === 0) {
                    // 左锚点：允许向右移动但不能超过下一个点
                    if (points.length > 1) {
                        const nextPoint = points[1];
                        point.x = clamp(val, 0, nextPoint.x - EPSILON);
                    } else {
                        point.x = 0;
                    }
                } else if (idx === points.length - 1) {
                    // 右锚点：允许向左移动但不能超过前一个点
                    if (points.length > 1) {
                        const prevPoint = points[points.length - 2];
                        point.x = clamp(val, prevPoint.x + EPSILON, 1);
                    } else {
                        point.x = 1;
                    }
                } else {
                    point.x = clampPointX(points, idx, val);
                }
                setSelectedIndex(idx);
                updateChannelButtons();
                save();
            });

            outputNumber.addEventListener("input", () => {
                const points = getPoints();
                const idx = getSelectedIndex();
                const point = points[idx];
                point.y = clamp((parseFloat(outputNumber.value) || 0) / 255, 0, 1);
                setSelectedIndex(idx);
                updateChannelButtons();
                save();
            });

            deleteBtn.addEventListener("click", () => {
                const idx = getSelectedIndex();
                removePoint(idx);
            });

            resetChannelBtn.addEventListener("click", () => {
                curves[activeChannel] = normalizePoints(PRESETS[0].points);
                selectedPerChannel[activeChannel] = 0;
                updateChannelButtons();
                updateInfo();
                draw();
                save();
            });

            resetAllBtn.addEventListener("click", () => {
                CHANNELS.forEach(ch => {
                    curves[ch.id] = normalizePoints(PRESETS[0].points);
                    selectedPerChannel[ch.id] = 0;
                });
                activeChannel = "RGB";
                updateChannelButtons();
                updateInfo();
                draw();
                save();
            });


            applyPresetBtn.addEventListener("click", () => {
                const preset = PRESETS.find(p => p.id === presetSelect.value);
                if (!preset) return;

                // Handle multi-channel presets (for Color Negative and Cross Process)
                if (preset.channels) {
                    // Reset all channels to default first
                    CHANNELS.forEach(ch => {
                        curves[ch.id] = clonePoints(PRESETS[0].points); // Reset to default
                        selectedPerChannel[ch.id] = 0;
                    });

                    // Apply curves to specific channels (R, G, B)
                    Object.keys(preset.channels).forEach(channelId => {
                        if (curves[channelId]) {
                            curves[channelId] = normalizePoints(preset.channels[channelId]);
                            selectedPerChannel[channelId] = Math.min(1, curves[channelId].length - 1);
                        }
                    });
                } else {
                    // Handle single-channel presets - reset ALL channels first, then apply to active
                    CHANNELS.forEach(ch => {
                        curves[ch.id] = clonePoints(PRESETS[0].points); // Reset to default
                        selectedPerChannel[ch.id] = 0;
                    });

                    // Apply preset only to current active channel
                    curves[activeChannel] = normalizePoints(preset.points);
                    selectedPerChannel[activeChannel] = Math.min(1, curves[activeChannel].length - 1);
                }

                updateChannelButtons();
                updateInfo();
                draw();
                save();
            });

            updateChannelButtons();
            
            const htmlWidget = node.addDOMWidget("curves_editor", "div", container);
            htmlWidget.computeSize = function (width) {
                return [width, 500];
            };

            node.setSize([
                Math.max(node.size[0], 360),
                Math.max(node.size[1], 530)
            ]);

            requestAnimationFrame(() => {
                draw();
                updateInfo();
            });

            let lastWidgetValue = widget.value;
            const checkWidgetValueChange = () => {
                if (widget.value === lastWidgetValue) return;
                lastWidgetValue = widget.value;
                applyPayload(widget.value);
                updateChannelButtons();
                updateInfo();
                draw();
            };
            const valuePoller = setInterval(checkWidgetValueChange, 200);


            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                cleanup();
                clearInterval(valuePoller);
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            return result;
        };
    }
});
