import { app } from "../../scripts/app.js";
import { gradientStyles, GradientEditor } from "./rc_gradient_common.js";

// 注入样式
if (!document.getElementById("rc-gradient-map-styles")) {
    const styleEl = document.createElement("style");
    styleEl.id = "rc-gradient-map-styles";
    styleEl.textContent = gradientStyles;
    document.head.appendChild(styleEl);
}

app.registerExtension({
    name: "RC.GradientMap",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RC_GradientMap") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated?.apply(this, arguments);

                const widget = this.widgets.find(w => w.name === "gradient_data");
                if (!widget) return ret;

                // Hide the original widget
                widget.type = "converted-widget";
                widget.serializeValue = () => widget.value;

                // Create gradient editor container
                const container = document.createElement("div");
                container.style.margin = "4px 0";

                // Parse initial gradient data
                let initialStops;
                try {
                    const data = JSON.parse(widget.value);
                    initialStops = data.stops;
                } catch {
                    initialStops = [
                        { position: 0.0, color: [0, 0, 0, 255] },
                        { position: 1.0, color: [255, 255, 255, 255] }
                    ];
                }

                // Create gradient editor
                const editor = new GradientEditor(container, initialStops);

                // Update widget when gradient changes
                editor.onUpdate = (stops) => {
                    const data = { stops: stops };
                    widget.value = JSON.stringify(data);
                    if (widget.callback) {
                        widget.callback(widget.value, this, app);
                    }
                    this.graph?.setDirtyCanvas(true, true);
                    lastWidgetValue = widget.value;
                };

                // Listen for external widget value changes (e.g., loading saved workflows)
                let lastWidgetValue = widget.value;
                const checkWidgetValueChange = () => {
                    if (widget.value !== lastWidgetValue) {
                        lastWidgetValue = widget.value;
                        try {
                            const newData = JSON.parse(widget.value);
                            if (newData.stops) {
                                editor.setStops(newData.stops);
                            }
                        } catch {
                            // If parsing fails, ignore
                        }
                    }
                };

                // Check for widget value changes periodically
                const valuePoller = setInterval(checkWidgetValueChange, 200);

                // Add the editor to the node
                const htmlWidget = this.addDOMWidget("gradient_map_editor", "div", container);
                htmlWidget.computeSize = function(width) {
                    return [width, 260];
                };

                // Increase node size to accommodate the gradient editor
                this.setSize([
                    Math.max(this.size[0], 320),
                    Math.max(this.size[1], this.size[1] + 280)
                ]);

                const onRemoved = this.onRemoved;
                this.onRemoved = function () {
                    clearInterval(valuePoller);
                    return onRemoved ? onRemoved.apply(this, arguments) : undefined;
                };

                return ret;
            };
        }
    }
});
