import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LR_STATE = new Map();

const TEXT = {
    BrowserTitle: { en: "LR/ACR Preset Browser", zh: "LR/ACR 预设浏览器" },
    Refresh: { en: "Refresh", zh: "刷新" },
    RootLabel: { en: "Root", zh: "预设根目录" },
    NoRoot: { en: "Set a preset directory and run the workflow.", zh: "请设置预设目录后运行节点。" },
    Folders: { en: "Folders", zh: "文件夹" },
    Presets: { en: "Presets", zh: "预设" },
    Original: { en: "Original", zh: "原图" },
    Preview: { en: "Preview", zh: "预览" },
    SelectPreset: { en: "Select a preset to preview it.", zh: "选择预设以查看效果。" },
    NoImage: { en: "Run the workflow to load an image.", zh: "运行工作流以加载图像。" },
    PassForwardHint: {
        en: "Preview here, then enable Pass Forward to continue the workflow.",
        zh: "在此预览，满意后启用 Pass Forward 继续工作流。"
    },
    SelectedLabel: { en: "Selected", zh: "已选择" },
    Loading: { en: "Loading…", zh: "加载中…" },
    Error: { en: "Error", zh: "错误" },
    BackToRoot: { en: "Root", zh: "根目录" },
};

const getLocale = () => {
    try {
        const stored = localStorage.getItem("Comfy.Settings.Locale");
        if (stored) {
            if (stored.toLowerCase().startsWith("zh")) return "zh";
            return "en";
        }
    } catch (_) { /* ignore */ }
    const lang = navigator?.language?.toLowerCase() || "en";
    return lang.startsWith("zh") ? "zh" : "en";
};

const t = (key) => {
    const lang = getLocale();
    return TEXT[key]?.[lang] || TEXT[key]?.en || key;
};

const hideWidget = (node, widget) => {
    if (!widget || widget._rc_hidden) return;
    widget._rc_hidden = true;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
};

const markDirty = (node) => {
    if (!node) return;
    node.widgets_changed = true;
    node?.graph?.setDirtyCanvas(true, true);
    app?.graph?.setDirtyCanvas(true, true);
};

const ensureState = (node) => {
    if (!LR_STATE.has(node)) {
        LR_STATE.set(node, {
            baseImage: null,
            previewImage: null,
            directories: [],
            presets: [],
            breadcrumbs: [],
            currentPath: "",
            hasRoot: false,
            status: "",
            strength: 100,
            browserState: "{}",
            selectedPreset: "",
            lastSessionStamp: 0,
        });
    }
    return LR_STATE.get(node);
};

const safeParseState = (raw) => {
    if (!raw) return {};
    try {
        const parsed = JSON.parse(raw);
        return typeof parsed === "object" && parsed !== null ? parsed : {};
    } catch (_) {
        return {};
    }
};

const serializeBrowserState = (path) => JSON.stringify({ currentPath: path || "" });

const setStatus = (node, message, type = "info") => {
    const ui = node._rcPresetUI;
    if (!ui) return;
    ui.status.textContent = message || "";
    ui.status.dataset.type = type;
};

const updateImages = (node) => {
    const ui = node._rcPresetUI;
    const state = ensureState(node);
    if (!ui) return;
    if (state.baseImage) {
        ui.originalImg.src = state.baseImage;
        ui.originalImg.alt = "Base";
    } else {
        ui.originalImg.removeAttribute("src");
        ui.originalImg.alt = "Empty";
    }

    if (state.previewImage) {
        ui.previewImg.src = state.previewImage;
        ui.previewImg.alt = "Preview";
    } else if (state.baseImage) {
        ui.previewImg.src = state.baseImage;
        ui.previewImg.alt = "Base";
    } else {
        ui.previewImg.removeAttribute("src");
        ui.previewImg.alt = "Empty";
    }

    if (state.selectedPreset) {
        ui.selection.textContent = `${t("SelectedLabel")}: ${state.selectedPreset}`;
    } else {
        ui.selection.textContent = t("SelectPreset");
    }

    ui.hint.textContent = state.baseImage ? t("PassForwardHint") : t("NoImage");
};

const updateRootLabel = (node) => {
    const ui = node._rcPresetUI;
    const state = ensureState(node);
    if (!ui) return;
    ui.rootSpan.textContent = state.rootDir || "—";
};

const updateStrengthLabel = (node) => {
    const ui = node._rcPresetUI;
    const state = ensureState(node);
    if (!ui) return;
    ui.strengthValue.textContent = `${Math.round(state.strength ?? 0)}%`;
};

const updateBrowserStateWidget = (node, path) => {
    const widget = node.widgets?.find((w) => w.name === "browser_state");
    if (!widget) return;
    const newValue = serializeBrowserState(path);
    if (widget.value === newValue) return;
    widget.value = newValue;
    markDirty(node);
};

const updateSelectedPresetWidget = (node, presetPath) => {
    const widget = node.widgets?.find((w) => w.name === "selected_preset");
    if (!widget) return;
    widget.value = presetPath || "";
    markDirty(node);
};

const renderListing = (node) => {
    const ui = node._rcPresetUI;
    const state = ensureState(node);
    if (!ui) return;

    ui.folderList.innerHTML = "";
    ui.presetList.innerHTML = "";
    ui.breadcrumbs.innerHTML = "";

    if (!state.hasRoot) {
        setStatus(node, t("NoRoot"), "info");
        return;
    }

    setStatus(node, state.status || "", state.status ? "info" : "idle");

    const rootCrumb = document.createElement("button");
    rootCrumb.className = "rc-lr-breadcrumb";
    rootCrumb.textContent = t("BackToRoot");
    rootCrumb.addEventListener("click", () => requestListing(node, ""));
    ui.breadcrumbs.appendChild(rootCrumb);

    state.breadcrumbs.forEach((crumb) => {
        const el = document.createElement("button");
        el.className = "rc-lr-breadcrumb";
        el.textContent = crumb.label;
        el.addEventListener("click", () => requestListing(node, crumb.path));
        ui.breadcrumbs.appendChild(el);
    });

    const folderTitle = document.createElement("div");
    folderTitle.className = "rc-lr-section-title";
    folderTitle.textContent = t("Folders");
    ui.folderList.appendChild(folderTitle);

    if (!state.directories.length) {
        const empty = document.createElement("div");
        empty.className = "rc-lr-empty";
        empty.textContent = "—";
        ui.folderList.appendChild(empty);
    } else {
        state.directories.forEach((dir) => {
            const button = document.createElement("button");
            button.className = "rc-lr-item";
            button.textContent = dir.name;
            button.addEventListener("click", () => requestListing(node, dir.relative_path));
            ui.folderList.appendChild(button);
        });
    }

    const presetTitle = document.createElement("div");
    presetTitle.className = "rc-lr-section-title";
    presetTitle.textContent = t("Presets");
    ui.presetList.appendChild(presetTitle);

    if (!state.presets.length) {
        const empty = document.createElement("div");
        empty.className = "rc-lr-empty";
        empty.textContent = "—";
        ui.presetList.appendChild(empty);
    } else {
        state.presets.forEach((preset) => {
            const button = document.createElement("button");
            button.className = "rc-lr-item";
            button.textContent = preset.name;
            if (state.selectedPreset === preset.relative_path) {
                button.classList.add("active");
            }
            button.addEventListener("click", () => requestPreview(node, preset.relative_path));
            ui.presetList.appendChild(button);
        });
    }
};

const fetchJson = async (url, payload) => {
    const response = await api.fetchApi(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || data.status !== "success") {
        throw new Error(data.message || "Request failed");
    }
    return data;
};

const requestListing = async (node, path) => {
    const state = ensureState(node);
    if (!node.rcUniqueId) {
        setStatus(node, t("NoRoot"), "error");
        return;
    }
    if (!state.hasRoot) {
        setStatus(node, t("NoRoot"), "info");
        return;
    }

    try {
        setStatus(node, t("Loading"), "info");
        const data = await fetchJson("/rc_lr_presets/list", {
            node_id: node.rcUniqueId,
            path: path || "",
        });
        state.directories = data.directories || [];
        state.presets = data.presets || [];
        state.breadcrumbs = data.breadcrumbs || [];
        state.currentPath = data.path || "";
        updateBrowserStateWidget(node, state.currentPath);
        state.status = "";
        renderListing(node);
    } catch (err) {
        state.status = `${t("Error")}: ${err.message}`;
        setStatus(node, state.status, "error");
    }
};

const getStrengthValue = (node) => {
    const widget = node.widgets?.find((w) => w.name === "strength");
    return widget ? Number(widget.value) : 100;
};

const requestPreview = async (node, presetPath) => {
    const state = ensureState(node);
    if (!node.rcUniqueId) {
        setStatus(node, t("NoRoot"), "error");
        return;
    }
    try {
        setStatus(node, t("Loading"), "info");
        const strength = getStrengthValue(node);
        const data = await fetchJson("/rc_lr_presets/preview", {
            node_id: node.rcUniqueId,
            preset: presetPath,
            strength,
        });
        state.previewImage = data.preview;
        state.selectedPreset = data.preset?.relative_path || presetPath;
        updateSelectedPresetWidget(node, state.selectedPreset);
        updateImages(node);
        renderListing(node);
        setStatus(node, "", "idle");
    } catch (err) {
        setStatus(node, `${t("Error")}: ${err.message}`, "error");
    }
};

const setupDom = (node) => {
    const container = document.createElement("div");
    container.className = "rc-lr-panel";

    const header = document.createElement("div");
    header.className = "rc-lr-header";
    const title = document.createElement("div");
    title.className = "rc-lr-title";
    title.textContent = t("BrowserTitle");
    const refresh = document.createElement("button");
    refresh.className = "rc-lr-refresh";
    refresh.textContent = t("Refresh");
    refresh.addEventListener("click", () => {
        const state = ensureState(node);
        state.listLoaded = false;
        requestListing(node, state.currentPath || "");
    });
    header.appendChild(title);
    header.appendChild(refresh);

    const rootRow = document.createElement("div");
    rootRow.className = "rc-lr-root";
    const rootLabel = document.createElement("span");
    rootLabel.textContent = `${t("RootLabel")}: `;
    const rootSpan = document.createElement("span");
    rootSpan.className = "rc-lr-root-path";
    rootSpan.textContent = "—";
    rootRow.appendChild(rootLabel);
    rootRow.appendChild(rootSpan);

    const strengthRow = document.createElement("div");
    strengthRow.className = "rc-lr-strength-row";
    const strengthLabel = document.createElement("span");
    strengthLabel.textContent = "Strength:";
    const strengthValue = document.createElement("span");
    strengthValue.className = "rc-lr-strength-value";
    strengthValue.textContent = "100%";
    strengthRow.appendChild(strengthLabel);
    strengthRow.appendChild(strengthValue);

    const body = document.createElement("div");
    body.className = "rc-lr-body";

    const listing = document.createElement("div");
    listing.className = "rc-lr-listing";
    const breadcrumbs = document.createElement("div");
    breadcrumbs.className = "rc-lr-breadcrumbs";
    const folderList = document.createElement("div");
    folderList.className = "rc-lr-list rc-lr-scroll";
    const presetList = document.createElement("div");
    presetList.className = "rc-lr-list rc-lr-scroll";
    listing.appendChild(breadcrumbs);
    listing.appendChild(folderList);
    listing.appendChild(presetList);

    const preview = document.createElement("div");
    preview.className = "rc-lr-preview";
    const imgRow = document.createElement("div");
    imgRow.className = "rc-lr-preview-images";

    const originalBlock = document.createElement("div");
    originalBlock.className = "rc-lr-preview-block";
    const originalLabel = document.createElement("div");
    originalLabel.className = "rc-lr-preview-label";
    originalLabel.textContent = t("Original");
    const originalImg = document.createElement("img");
    originalImg.className = "rc-lr-preview-img";
    originalBlock.appendChild(originalLabel);
    originalBlock.appendChild(originalImg);

    const previewBlock = document.createElement("div");
    previewBlock.className = "rc-lr-preview-block";
    const previewLabel = document.createElement("div");
    previewLabel.className = "rc-lr-preview-label";
    previewLabel.textContent = t("Preview");
    const previewImg = document.createElement("img");
    previewImg.className = "rc-lr-preview-img";
    previewBlock.appendChild(previewLabel);
    previewBlock.appendChild(previewImg);

    imgRow.appendChild(originalBlock);
    imgRow.appendChild(previewBlock);

    const selection = document.createElement("div");
    selection.className = "rc-lr-selection";
    selection.textContent = t("SelectPreset");

    const hint = document.createElement("div");
    hint.className = "rc-lr-hint";
    hint.textContent = t("NoImage");

    const status = document.createElement("div");
    status.className = "rc-lr-status";

    preview.appendChild(imgRow);
    preview.appendChild(selection);
    preview.appendChild(hint);
    preview.appendChild(status);

    container.appendChild(header);
    container.appendChild(rootRow);
    container.appendChild(strengthRow);
    container.appendChild(body);
    body.appendChild(listing);
    body.appendChild(preview);

    node.addDOMWidget("rc_lr_browser", "div", container, {
        serialize: false,
        hideOnZoom: false
    });
    container.style.height = "100%";
    container.style.boxSizing = "border-box";

    const domWrapper = container.parentElement;
    if (domWrapper && domWrapper.classList?.contains("dom-widget")) {
        domWrapper.style.height = "100%";
        domWrapper.style.alignSelf = "stretch";
    }

    if (!node._rcPresetSizeInit) {
        const MIN_WIDTH = 540;
        const MIN_HEIGHT = 720;
        const originalSetSize = node.setSize ? node.setSize.bind(node) : null;
        node._rcOriginalSetSize = originalSetSize;
        node.setSize = function(size) {
            const target = [
                Math.max(size?.[0] ?? MIN_WIDTH, MIN_WIDTH),
                Math.max(size?.[1] ?? MIN_HEIGHT, MIN_HEIGHT)
            ];
            if (this._rcOriginalSetSize) {
                this._rcOriginalSetSize(target);
            } else if (originalSetSize) {
                originalSetSize(target);
            } else {
                this.size = target;
            }
        };
        node.setSize([MIN_WIDTH, MIN_HEIGHT]);
        node._rcPresetSizeInit = true;
    }

    node._rcPresetUI = {
        container,
        rootSpan,
        strengthValue,
        folderList,
        presetList,
        breadcrumbs,
        originalImg,
        previewImg,
        selection,
        hint,
        status,
    };
};

const setupNode = (node) => {
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        LR_STATE.delete(node);
        return onRemoved?.apply(this, arguments);
    };

    const originalOnWidgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function (name, value, widget) {
        const state = ensureState(this);
        if (name === "strength") {
            state.strength = Number(value);
            updateStrengthLabel(this);
        }
        if (name === "preset_directory") {
            state.rootDir = value || "";
            state.hasRoot = false;
            state.directories = [];
            state.presets = [];
            state.breadcrumbs = [];
            state.listLoaded = false;
            updateRootLabel(this);
            renderListing(this);
        }
        return originalOnWidgetChanged?.apply(this, arguments);
    };

    const selectedWidget = node.widgets?.find((w) => w.name === "selected_preset");
    const stateWidget = node.widgets?.find((w) => w.name === "browser_state");
    const directoryWidget = node.widgets?.find((w) => w.name === "preset_directory");
    const strengthWidget = node.widgets?.find((w) => w.name === "strength");

    hideWidget(node, selectedWidget);
    hideWidget(node, stateWidget);

    // Initialize state from saved widget values (for browser refresh)
    const state = ensureState(node);
    if (selectedWidget?.value) {
        state.selectedPreset = selectedWidget.value;
    }
    if (stateWidget?.value) {
        state.browserState = stateWidget.value;
        const parsed = safeParseState(stateWidget.value);
        state.currentPath = parsed.currentPath || "";
    }
    if (directoryWidget?.value) {
        state.rootDir = directoryWidget.value;
        state.hasRoot = !!directoryWidget.value;
    }
    if (strengthWidget?.value !== undefined) {
        state.strength = Number(strengthWidget.value);
    }

    setupDom(node);
    updateRootLabel(node);
    updateStrengthLabel(node);
    updateImages(node);
    renderListing(node);

    // Load directory listing if preset directory is set
    if (state.hasRoot && state.currentPath !== undefined) {
        // Delay to ensure DOM is ready and node has rcUniqueId
        setTimeout(() => {
            if (!node.rcUniqueId) {
                // Generate a temporary ID for API calls before workflow execution
                node.rcUniqueId = node.id;
            }
            requestListing(node, state.currentPath);
        }, 100);
    }
};

app.registerExtension({
    name: "RC.LRPresetBrowser",
    setup() {
        api.addEventListener("rc_lr_preset_session", (event) => {
            const { output, node: uniqueId } = event.detail;
            const target = app.graph.getNodeById(uniqueId);
            if (!target) return;

            target.rcUniqueId = uniqueId;
            const state = ensureState(target);
            state.baseImage = output.base_image?.[0] || null;
            state.selectedPreset = output.selected_preset?.[0] || "";
            state.browserState = output.browser_state?.[0] || "{}";
            state.rootDir = output.root_dir?.[0] || "";
            state.hasRoot = !!output.has_root?.[0];
            state.strength = output.strength?.[0] ?? getStrengthValue(target);
            state.listLoaded = false;
            state.status = state.hasRoot ? "" : t("NoRoot");
            state.lastSessionStamp = Date.now();
            const previewFromServer = output.preview_image?.[0];
            if (previewFromServer) {
                state.previewImage = previewFromServer;
            } else if (!state.previewImage) {
                state.previewImage = state.baseImage;
            }

            updateSelectedPresetWidget(target, state.selectedPreset);
            const parsedState = safeParseState(state.browserState);
            updateBrowserStateWidget(target, parsedState.currentPath || "");

            if (target._rcPresetUI) {
                updateRootLabel(target);
                updateStrengthLabel(target);
                updateImages(target);
                renderListing(target);
            }

            if (state.hasRoot) {
                const stored = safeParseState(state.browserState);
                requestListing(target, stored.currentPath || "");
            }
        });
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RC_LRPreset") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const res = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return res;
        };
    }
});

const style = document.createElement("style");
style.textContent = `
.rc-lr-panel {
    background: #1d1d1d;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: 100%;
    overflow: hidden;
    box-sizing: border-box;
}
.rc-lr-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rc-lr-title {
    font-weight: 600;
    color: #e6e6e6;
}
.rc-lr-refresh {
    background: #3f7de8;
    border: none;
    padding: 4px 12px;
    border-radius: 4px;
    color: #fff;
    cursor: pointer;
}
.rc-lr-refresh:hover {
    background: #518bf0;
}
.rc-lr-root {
    font-size: 12px;
    color: #bbb;
}
.rc-lr-root-path {
    color: #f5f5f5;
    word-break: break-all;
}
.rc-lr-strength-row {
    font-size: 12px;
    color: #bbb;
    display: flex;
    gap: 6px;
}
.rc-lr-strength-value {
    color: #6fb2ff;
    font-weight: 600;
}
.rc-lr-body {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    flex: 1;
    min-height: 0;
}
.rc-lr-listing {
    flex: 1;
    min-width: 260px;
    background: #232323;
    border-radius: 6px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-height: 0;
    overflow-y: auto;
}
.rc-lr-breadcrumbs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.rc-lr-breadcrumb {
    background: #2f2f2f;
    border: 1px solid #3a3a3a;
    color: #ddd;
    padding: 3px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
}
.rc-lr-breadcrumb:hover {
    border-color: #4a7dff;
}
.rc-lr-list {
    background: #1b1b1b;
    border-radius: 6px;
    padding: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-height: 0;
}
.rc-lr-scroll {
    max-height: 220px;
    overflow-y: auto;
}
.rc-lr-section-title {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.rc-lr-item {
    background: #262626;
    border: 1px solid #333;
    color: #e0e0e0;
    padding: 6px 8px;
    border-radius: 4px;
    text-align: left;
    cursor: pointer;
    font-size: 12px;
}
.rc-lr-item:hover {
    border-color: #4a7dff;
}
.rc-lr-item.active {
    border-color: #4a7dff;
    background: #3050a0;
}
.rc-lr-empty {
    color: #555;
    font-size: 12px;
    padding: 6px 0;
}
.rc-lr-preview {
    flex: 1;
    min-width: 260px;
    background: #232323;
    border-radius: 6px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-height: 0;
    overflow-y: auto;
}
.rc-lr-preview-images {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.rc-lr-preview-block {
    flex: 1;
    background: #151515;
    border-radius: 6px;
    padding: 6px;
}
.rc-lr-preview-img {
    width: 100%;
    border-radius: 4px;
    object-fit: contain;
    background: #000;
}
.rc-lr-preview-label {
    font-size: 12px;
    color: #999;
    margin-bottom: 4px;
}
.rc-lr-selection {
    color: #ddd;
    font-size: 12px;
}
.rc-lr-hint {
    font-size: 12px;
    color: #7bb9ff;
}
.rc-lr-status {
    font-size: 11px;
    color: #bbb;
}
.rc-lr-status[data-type="error"] {
    color: #ff7171;
}
`;
document.head.appendChild(style);
