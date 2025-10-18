/**
 * RC Interactive Compositor - Frontend
 * Native Canvas API implementation - Photoshop-like interactive compositor
 * Features: Move, Scale, Rotate foreground, Eraser tool
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const debugLog = () => {};

// Helper: Hide widget
function hideWidget(node, widget) {
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
}

// Compositor Editor Class
class RCCompositorEditor {
    constructor(node, containerDiv) {
        this.node = node;
        this.containerDiv = containerDiv;
        this.canvas = null;
        this.ctx = null;
        this.toolbar = null;

        // Images
        this.backgroundImg = null;
        this.foregroundImg = null;
        this.foregroundMask = null; // For eraser

        // Canvas dimensions
        this.canvasWidth = 1024;
        this.canvasHeight = 1024;

        // Foreground transform
        this.fgTransform = {
            x: 100,
            y: 100,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
            width: 0,
            height: 0
        };

        // Interaction state
        this.currentTool = "transform"; // transform or eraser
        this.isDragging = false;
        this.isRotating = false;
        this.isScaling = false;
        this.isErasing = false;
        this.dragStart = { x: 0, y: 0 };
        this.eraserSize = 30;

        // Control handles
        this.showHandles = true;
        this.hoveredHandle = null;

        this.needsUpload = false;
        this.scalingCorner = null;
        this.scalingEdge = null;
        this.pendingTransform = null;
        this.uploadTimeout = null;
        this.pendingMaskData = null;
        this.lastMaskData = null;
        this.maskDirty = false;
        this.isInitializing = false;
        this.animationFrameId = null;
        this.lastEraseCanvasPos = null;
    }

    init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");

        // Mouse events
        canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
        canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
        canvas.addEventListener("mouseup", this.onMouseUp.bind(this));
        canvas.addEventListener("wheel", this.onWheel.bind(this));

        // Start continuous render loop
        this.startRenderLoop();
    }

    startRenderLoop() {
        const renderFrame = () => {
            this.render();
            this.animationFrameId = requestAnimationFrame(renderFrame);
        };
        renderFrame();
    }

    stopRenderLoop() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    setCanvasSize(width, height) {
        this.canvasWidth = width;
        this.canvasHeight = height;
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.canvas.style.width = `${width}px`;
            this.canvas.style.height = `${height}px`;
            
            this.updateNodeSize();
        }
    }

    updateNodeSize() {
        const horizontalPadding = 80;
        const widgetReserve = 110; // Space reserved for downstream widgets
        const baseMinWidth = 520;
        const baseMinHeight = 880;

        let chromeHeight = widgetReserve;

        if (this.toolbar) {
            chromeHeight += this.toolbar.offsetHeight || 0;
        } else {
            chromeHeight += 70;
        }

        let containerExtra = 30;
        if (this.containerDiv) {
            const styles = window.getComputedStyle(this.containerDiv);
            const verticalPadding = parseFloat(styles.paddingTop || 0) + parseFloat(styles.paddingBottom || 0);
            const verticalMargin = parseFloat(styles.marginTop || 0) + parseFloat(styles.marginBottom || 0);
            containerExtra = verticalPadding + verticalMargin;
        }

        const targetWidth = Math.max(baseMinWidth, this.canvasWidth + horizontalPadding);
        const targetHeight = Math.max(
            baseMinHeight,
            this.canvasHeight + chromeHeight + containerExtra
        );

        this.node.setSize([targetWidth, targetHeight]);
    }

    loadBackground(imageDataUrl) {
        const img = new Image();
        img.onload = () => {
            this.backgroundImg = img;
            debugLog("Background loaded", { width: img.width, height: img.height });
            this.render();
        };
        img.src = imageDataUrl;
    }

    loadForeground(imageDataUrl) {
        const img = new Image();
        img.onload = () => {
            this.foregroundImg = img;

            const hasPendingTransform = !!this.pendingTransform;
            debugLog("Foreground loaded", {
                width: img.width,
                height: img.height,
                hasPendingTransform,
                hasPendingMask: !!this.pendingMaskData
            });

            // Always refresh intrinsic dimensions to match the loaded asset
            this.fgTransform.width = img.width;
            this.fgTransform.height = img.height;

            // Only reset default placement when no saved transform is pending
            if (!hasPendingTransform) {
                this.fgTransform.x = (this.canvasWidth - img.width) / 2;
                this.fgTransform.y = (this.canvasHeight - img.height) / 2;
                this.fgTransform.scaleX = 1;
                this.fgTransform.scaleY = 1;
                this.fgTransform.rotation = 0;
            }

            // Prepare mask canvas before applying any saved mask
            this.initMask(img.width, img.height);

            if (this.pendingMaskData) {
                debugLog("Applying pending mask data");
                this.applyMaskData(this.pendingMaskData);
                this.pendingMaskData = null;
            }

            if (hasPendingTransform) {
                debugLog("Applying pending transform", this.pendingTransform);
                this.applyForegroundTransform(this.pendingTransform);
                this.pendingTransform = null;
            } else {
                const initialPayload = {
                    foreground: this.getTransformSnapshot(),
                    saved: false
                };
                debugLog("Committing initial transform", initialPayload.foreground);
                this.commitCompositionData(initialPayload);
                this.render();
            }

            if (this.isInitializing) {
                this.isInitializing = false;
                debugLog("Initialization complete");
            }
        };
        img.src = imageDataUrl;
    }

    setCompositionState(data) {
        if (!data || typeof data !== "object") {
            this.pendingTransform = null;
            this.pendingMaskData = null;
            return;
        }

        if (data.foreground && typeof data.foreground === "object") {
            this.pendingTransform = { ...data.foreground };
            debugLog("Received pending transform", this.pendingTransform);
            if (!this.isInitializing && this.foregroundImg) {
                this.applyForegroundTransform(this.pendingTransform);
                this.pendingTransform = null;
            }
        }

        if (data.mask) {
            this.pendingMaskData = data.mask;
            this.lastMaskData = data.mask;
            this.maskDirty = false;
            debugLog("Received pending mask");
            if (!this.isInitializing && this.foregroundMask) {
                this.applyMaskData(data.mask);
                this.pendingMaskData = null;
            }
        } else {
            this.pendingMaskData = null;
            this.lastMaskData = null;
            this.maskDirty = false;
        }

        if (!this.isInitializing) {
            this.needsUpload = false;
            if (this.uploadTimeout) {
                clearTimeout(this.uploadTimeout);
                this.uploadTimeout = null;
            }
        }
    }

    applyForegroundTransform(transform) {
        if (!transform || !this.foregroundImg) return;

        const t = this.fgTransform;
        const toNumber = (value, fallback) => {
            const num = Number(value);
            return Number.isFinite(num) ? num : fallback;
        };

        t.x = toNumber(transform.x, t.x);
        t.y = toNumber(transform.y, t.y);
        t.scaleX = toNumber(transform.scaleX, t.scaleX);
        t.scaleY = toNumber(transform.scaleY, t.scaleY);
        t.rotation = toNumber(transform.rotation, t.rotation);
        t.width = Math.max(1, toNumber(transform.width, t.width || this.foregroundImg.width));
        t.height = Math.max(1, toNumber(transform.height, t.height || this.foregroundImg.height));
        debugLog("Applied transform", this.getTransformSnapshot());
        this.render();
    }

    beginInitialize() {
        debugLog("Begin initialization");
        this.isInitializing = true;
    }

    applyMaskData(maskDataUrl) {
        if (!maskDataUrl || !this.foregroundMask) return;

        const img = new Image();
        img.onload = () => {
            const maskCtx = this.foregroundMask.getContext("2d");
            maskCtx.save();
            maskCtx.globalCompositeOperation = "copy";
            maskCtx.drawImage(img, 0, 0, this.foregroundMask.width, this.foregroundMask.height);
            maskCtx.restore();
            this.lastMaskData = maskDataUrl;
            this.maskDirty = false;
            debugLog("Applied mask data");
            this.render();
        };
        img.onerror = () => {
            console.warn("Failed to load saved mask data");
        };
        img.src = maskDataUrl;
    }

    initMask(width, height) {
        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = width;
        maskCanvas.height = height;
        const maskCtx = maskCanvas.getContext("2d");

        // Fill with white (fully visible)
        maskCtx.fillStyle = "white";
        maskCtx.fillRect(0, 0, width, height);

        this.foregroundMask = maskCanvas;
        this.maskDirty = false;
        this.lastMaskData = null;
    }

    render() {
        if (!this.ctx) return;

        // Clear canvas
        this.ctx.fillStyle = "#1a1a1a";
        this.ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);

        // Draw background (scaled to fit canvas)
        if (this.backgroundImg) {
            this.ctx.drawImage(
                this.backgroundImg,
                0, 0,
                this.canvasWidth,
                this.canvasHeight
            );
        }

        // Draw foreground with transform and mask
        if (this.foregroundImg) {
            this.ctx.save();

            const t = this.fgTransform;
            const centerX = t.x + (t.width * t.scaleX) / 2;
            const centerY = t.y + (t.height * t.scaleY) / 2;

            // Apply transform
            this.ctx.translate(centerX, centerY);
            this.ctx.rotate((t.rotation * Math.PI) / 180);
            this.ctx.scale(t.scaleX, t.scaleY);
            this.ctx.translate(-t.width / 2, -t.height / 2);

            // Apply mask
            if (this.foregroundMask) {
                // Create temp canvas with masked foreground
                const tempCanvas = document.createElement("canvas");
                tempCanvas.width = t.width;
                tempCanvas.height = t.height;
                const tempCtx = tempCanvas.getContext("2d");

                // Draw foreground
                tempCtx.drawImage(this.foregroundImg, 0, 0);

                // Apply mask
                tempCtx.globalCompositeOperation = "destination-in";
                tempCtx.drawImage(this.foregroundMask, 0, 0);

                this.ctx.drawImage(tempCanvas, 0, 0);
            } else {
                this.ctx.drawImage(this.foregroundImg, 0, 0);
            }

            this.ctx.restore();

            // Draw control handles
            if (this.showHandles && this.currentTool === "transform") {
                this.drawHandles();
            }
        }

        // Draw eraser cursor (size reflects actual erase size on canvas)
        if (this.currentTool === "eraser" && this.mousePos && this.foregroundImg) {
            const t = this.fgTransform;
            // Eraser size on canvas = eraser size in image space * current scale
            const displaySize = this.eraserSize * Math.min(t.scaleX, t.scaleY);

            this.ctx.save();
            this.ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
            this.ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(this.mousePos.x, this.mousePos.y, displaySize, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
            this.ctx.restore();
        }
    }

    drawHandles() {
        const corners = this.getTransformedCorners();
        const edges = this.getEdgeHandles();

        this.ctx.strokeStyle = "#4a9eff";
        this.ctx.lineWidth = 2;

        // Draw bounding box
        this.ctx.beginPath();
        this.ctx.moveTo(corners[0].x, corners[0].y);
        for (let i = 1; i < 4; i++) {
            this.ctx.lineTo(corners[i].x, corners[i].y);
        }
        this.ctx.closePath();
        this.ctx.stroke();

        // Draw corner handles (for scaling)
        corners.forEach((corner, i) => {
            this.ctx.fillStyle = this.hoveredHandle === `corner${i}` ? "#6abaff" : "#4a9eff";
            this.ctx.fillRect(corner.x - 5, corner.y - 5, 10, 10);
        });

        // Draw edge handles (for axis scaling)
        edges.forEach((edge, i) => {
            this.ctx.fillStyle = this.hoveredHandle === `edge${i}` ? "#6abaff" : "#4a9eff";
            this.ctx.fillRect(edge.x - 5, edge.y - 5, 10, 10);
        });

        // No separate rotation handle - rotate by dragging outside bounding box
    }

    getTransformedCorners() {
        const t = this.fgTransform;
        const w = t.width * t.scaleX;
        const h = t.height * t.scaleY;
        const centerX = t.x + w / 2;
        const centerY = t.y + h / 2;
        const rad = (t.rotation * Math.PI) / 180;

        const corners = [
            { x: -w/2, y: -h/2 }, // top-left
            { x: w/2, y: -h/2 },  // top-right
            { x: w/2, y: h/2 },   // bottom-right
            { x: -w/2, y: h/2 }   // bottom-left
        ];

        return corners.map(corner => {
            const rotX = corner.x * Math.cos(rad) - corner.y * Math.sin(rad);
            const rotY = corner.x * Math.sin(rad) + corner.y * Math.cos(rad);
            return {
                x: centerX + rotX,
                y: centerY + rotY
            };
        });
    }

    getEdgeHandles() {
        const t = this.fgTransform;
        const corners = this.getTransformedCorners();
        const centerX = t.x + (t.width * t.scaleX) / 2;
        const centerY = t.y + (t.height * t.scaleY) / 2;

        const edgePairs = [
            { points: [corners[0], corners[1]], orientation: "vertical" },
            { points: [corners[1], corners[2]], orientation: "horizontal" },
            { points: [corners[2], corners[3]], orientation: "vertical" },
            { points: [corners[3], corners[0]], orientation: "horizontal" }
        ];

        return edgePairs.map((edge, index) => {
            const x = (edge.points[0].x + edge.points[1].x) / 2;
            const y = (edge.points[0].y + edge.points[1].y) / 2;
            const vecX = x - centerX;
            const vecY = y - centerY;
            const length = Math.hypot(vecX, vecY) || 1;
            return {
                x,
                y,
                index,
                orientation: edge.orientation,
                vector: {
                    x: vecX / length,
                    y: vecY / length
                }
            };
        });
    }

    getPointerPosition(e) {
        if (!this.canvas) {
            return { x: 0, y: 0 };
        }

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = rect.width ? this.canvas.width / rect.width : 1;
        const scaleY = rect.height ? this.canvas.height / rect.height : 1;

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    getInteractionZone(x, y) {
        if (!this.foregroundImg) return null;

        const t = this.fgTransform;
        const corners = this.getTransformedCorners();
        const centerX = t.x + (t.width * t.scaleX) / 2;
        const centerY = t.y + (t.height * t.scaleY) / 2;

        const edges = this.getEdgeHandles();

        for (let i = 0; i < corners.length; i++) {
            const corner = corners[i];
            const dist = Math.hypot(x - corner.x, y - corner.y);
            if (dist < 15) {
                const angle = Math.atan2(corner.y - centerY, corner.x - centerX) * 180 / Math.PI;
                return { type: "corner", index: i, angle };
            }
        }

        for (let edge of edges) {
            const dist = Math.hypot(x - edge.x, y - edge.y);
            if (dist < 12) {
                return {
                    type: "edge",
                    index: edge.index,
                    orientation: edge.orientation,
                    vector: edge.vector
                };
            }
        }

        const rotationInner = 16;
        const rotationOuter = 44;
        for (let i = 0; i < corners.length; i++) {
            const corner = corners[i];
            const dist = Math.hypot(x - corner.x, y - corner.y);
            if (dist >= rotationInner && dist < rotationOuter) {
                const angle = Math.atan2(corner.y - centerY, corner.x - centerX) * 180 / Math.PI;
                return { type: "rotate", index: i, angle };
            }
        }

        return null;
    }

    onMouseDown(e) {
        const { x, y } = this.getPointerPosition(e);
        this.mousePos = { x, y };

        if (this.currentTool === "transform" && this.foregroundImg) {
            const zone = this.getInteractionZone(x, y);
            const t = this.fgTransform;

            if (zone?.type === "corner") {
                this.isScaling = true;
                this.scalingCorner = zone.index;
                this.scalingEdge = null;
                this.dragStart = { x, y };
                this.scaleStartTransform = {
                    x: t.x,
                    y: t.y,
                    scaleX: t.scaleX,
                    scaleY: t.scaleY,
                    width: t.width,
                    height: t.height
                };
                return;
            }

            if (zone?.type === "edge") {
                this.isScaling = true;
                this.scalingEdge = zone;
                this.scalingCorner = null;
                this.dragStart = { x, y };
                this.scaleStartTransform = {
                    x: t.x,
                    y: t.y,
                    scaleX: t.scaleX,
                    scaleY: t.scaleY,
                    width: t.width,
                    height: t.height
                };
                return;
            }

            if (zone?.type === "rotate") {
                this.isRotating = true;
                const centerX = t.x + (t.width * t.scaleX) / 2;
                const centerY = t.y + (t.height * t.scaleY) / 2;
                this.dragStart = { x, y };
                this.rotationCenter = { x: centerX, y: centerY };
                this.startAngle = Math.atan2(y - centerY, x - centerX) * 180 / Math.PI;
                this.startRotation = t.rotation;
                return;
            }

            if (this.isPointInForeground(x, y)) {
                this.isDragging = true;
                this.dragStart = { x, y };
                this.dragStartTransform = { x: t.x, y: t.y };
            }
        } else if (this.currentTool === "eraser") {
            this.isErasing = true;
            this.mousePos = { x, y };
            this.lastEraseCanvasPos = null;
            this.eraseAt(x, y);
        }
    }

    onMouseMove(e) {
        const { x, y } = this.getPointerPosition(e);

        this.mousePos = { x, y };

        if (this.isDragging) {
            // Make dragging follow mouse 1:1
            const dx = x - this.dragStart.x;
            const dy = y - this.dragStart.y;
            this.fgTransform.x = this.dragStartTransform.x + dx;
            this.fgTransform.y = this.dragStartTransform.y + dy;
            
            this.scheduleUpload();
        } else if (this.isRotating) {
            // Calculate rotation angle from center
            const currentAngle = Math.atan2(y - this.rotationCenter.y, x - this.rotationCenter.x) * 180 / Math.PI;
            const deltaAngle = currentAngle - this.startAngle;
            this.fgTransform.rotation = this.startRotation + deltaAngle;
            
            this.scheduleUpload();
        } else if (this.isScaling) {
            const t = this.scaleStartTransform;
            const centerX = t.x + (t.width * t.scaleX) / 2;
            const centerY = t.y + (t.height * t.scaleY) / 2;

            if (this.scalingEdge) {
                const baseExtent = this.scalingEdge.orientation === "vertical"
                    ? (t.height * t.scaleY) / 2
                    : (t.width * t.scaleX) / 2;

                if (baseExtent > 0) {
                    const vector = this.scalingEdge.vector;
                    const component = ((x - centerX) * vector.x) + ((y - centerY) * vector.y);
                    const ratio = Math.abs(component) / baseExtent;
                    const minScale = 0.05;

                    if (this.scalingEdge.orientation === "vertical") {
                        this.fgTransform.scaleY = Math.max(minScale, t.scaleY * ratio);
                        this.fgTransform.scaleX = t.scaleX;
                    } else {
                        this.fgTransform.scaleX = Math.max(minScale, t.scaleX * ratio);
                        this.fgTransform.scaleY = t.scaleY;
                    }

                    this.fgTransform.x = centerX - (t.width * this.fgTransform.scaleX) / 2;
                    this.fgTransform.y = centerY - (t.height * this.fgTransform.scaleY) / 2;
                }

                
                this.scheduleUpload();
            } else {
                const startDist = Math.hypot(this.dragStart.x - centerX, this.dragStart.y - centerY);
                const currentDist = Math.hypot(x - centerX, y - centerY);

                if (startDist > 0) {
                    const scaleRatio = currentDist / startDist;

                    this.fgTransform.scaleX = Math.max(0.05, t.scaleX * scaleRatio);
                    this.fgTransform.scaleY = Math.max(0.05, t.scaleY * scaleRatio);

                    // Adjust position to keep center fixed
                    this.fgTransform.x = centerX - (t.width * this.fgTransform.scaleX) / 2;
                    this.fgTransform.y = centerY - (t.height * this.fgTransform.scaleY) / 2;
                }

                
                this.scheduleUpload();
            }
        } else if (this.isErasing) {
            this.eraseAt(x, y);
        } else if (this.currentTool === "transform") {
            // Update hovered handle and cursor
            this.updateHoveredHandle(x, y);
            this.updateCursor(x, y);
            
        } else if (this.currentTool === "eraser") {
            // Just update cursor position for eraser preview
            
        }
    }

    onMouseUp() {
        const wasErasing = this.isErasing;

        this.isDragging = false;
        this.isRotating = false;
        this.isScaling = false;
        this.scalingCorner = null;
        this.scalingEdge = null;
        this.isErasing = false;
        this.lastEraseCanvasPos = null;

        // Upload after erasing is done
        if (wasErasing) {
            this.scheduleUpload();
        }

        this.flushCompositionData();
    }

    onWheel(e) {
        // Disable wheel scaling - use corner handles instead
        e.preventDefault();
    }

    updateHoveredHandle(x, y) {
        const zone = this.getInteractionZone(x, y);
        if (zone?.type === "corner") {
            this.hoveredHandle = `corner${zone.index}`;
        } else if (zone?.type === "edge") {
            this.hoveredHandle = `edge${zone.index}`;
        } else {
            this.hoveredHandle = null;
        }
    }

    updateCursor(x, y) {
        if (!this.canvas) return;

        const zone = this.getInteractionZone(x, y);

        if (zone?.type === "corner") {
            this.canvas.style.cursor = zone.index % 2 === 0 ? "nwse-resize" : "nesw-resize";
            return;
        }

        if (zone?.type === "edge") {
            this.canvas.style.cursor = zone.orientation === "horizontal" ? "col-resize" : "row-resize";
            return;
        }

        if (zone?.type === "rotate") {
            this.canvas.style.cursor = "crosshair";
            return;
        }

        if (this.isPointInForeground(x, y)) {
            this.canvas.style.cursor = "move";
        } else {
            this.canvas.style.cursor = "default";
        }
    }

    isPointInForeground(x, y) {
        if (!this.foregroundImg) return false;

        const t = this.fgTransform;
        const centerX = t.x + (t.width * t.scaleX) / 2;
        const centerY = t.y + (t.height * t.scaleY) / 2;

        // Apply inverse rotation to transform point to local space
        const rad = -(t.rotation * Math.PI) / 180;

        // Translate to center-origin
        const dx = x - centerX;
        const dy = y - centerY;

        // Rotate back
        const rotX = dx * Math.cos(rad) - dy * Math.sin(rad);
        const rotY = dx * Math.sin(rad) + dy * Math.cos(rad);

        // Scale back and check bounds
        const localX = rotX / t.scaleX;
        const localY = rotY / t.scaleY;

        const halfW = t.width / 2;
        const halfH = t.height / 2;

        return localX >= -halfW && localX <= halfW &&
               localY >= -halfH && localY <= halfH;
    }

    eraseAt(x, y) {
        if (!this.foregroundMask || !this.foregroundImg) return;

        this.mousePos = { x, y };
        const current = this.canvasToMask(x, y);
        if (!current.inside) {
            this.lastEraseCanvasPos = { x, y };
            return;
        }

        const maskCtx = this.foregroundMask.getContext("2d");
        maskCtx.globalCompositeOperation = "destination-out";
        maskCtx.fillStyle = "rgba(0,0,0,1)";

        const stampAt = (canvasPoint) => {
            const local = this.canvasToMask(canvasPoint.x, canvasPoint.y);
            if (!local.inside) return;
            maskCtx.beginPath();
            maskCtx.arc(local.x, local.y, this.eraserSize, 0, Math.PI * 2);
            maskCtx.fill();
        };

        if (this.lastEraseCanvasPos) {
            const last = this.lastEraseCanvasPos;
            const dx = x - last.x;
            const dy = y - last.y;
            const dist = Math.hypot(dx, dy);
            const step = Math.max(this.eraserSize * 0.4, 1);
            const steps = Math.max(1, Math.ceil(dist / step));
            for (let i = 1; i <= steps; i++) {
                const t = i / steps;
                const point = {
                    x: last.x + dx * t,
                    y: last.y + dy * t
                };
                stampAt(point);
            }
        } else {
            stampAt({ x, y });
        }

        this.lastEraseCanvasPos = { x, y };

        // Render immediately for responsive feedback
        this.render();
        this.maskDirty = true;
    }

    resetMask() {
        if (!this.foregroundMask) return;

        const maskCtx = this.foregroundMask.getContext("2d");
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.fillStyle = "white";
        maskCtx.fillRect(0, 0, this.foregroundMask.width, this.foregroundMask.height);

        this.render();
        const resetTransform = {
            foreground: {
                x: (this.canvasWidth - this.foregroundImg.width) / 2,
                y: (this.canvasHeight - this.foregroundImg.height) / 2,
                scaleX: 1,
                scaleY: 1,
                rotation: 0,
                width: this.foregroundImg.width,
                height: this.foregroundImg.height
            },
            mask: null,
            saved: false
        };

        this.fgTransform.x = resetTransform.foreground.x;
        this.fgTransform.y = resetTransform.foreground.y;
        this.fgTransform.scaleX = resetTransform.foreground.scaleX;
        this.fgTransform.scaleY = resetTransform.foreground.scaleY;
        this.fgTransform.rotation = resetTransform.foreground.rotation;
        this.fgTransform.width = resetTransform.foreground.width;
        this.fgTransform.height = resetTransform.foreground.height;

        this.lastMaskData = null;
        this.maskDirty = false;
        this.lastEraseCanvasPos = null;

        this.commitCompositionData(resetTransform);
        this.render();
    }

    canvasToMask(canvasX, canvasY) {
        const t = this.fgTransform;
        const centerX = t.x + (t.width * t.scaleX) / 2;
        const centerY = t.y + (t.height * t.scaleY) / 2;
        const rad = -(t.rotation * Math.PI) / 180;

        const dx = canvasX - centerX;
        const dy = canvasY - centerY;

        const rotX = dx * Math.cos(rad) - dy * Math.sin(rad);
        const rotY = dx * Math.sin(rad) + dy * Math.cos(rad);

        const localX = (rotX / t.scaleX) + t.width / 2;
        const localY = (rotY / t.scaleY) + t.height / 2;

        const inside = localX >= 0 && localX < t.width && localY >= 0 && localY < t.height;

        return { x: localX, y: localY, inside };
    }

    setTool(tool) {
        this.currentTool = tool;
        this.canvas.style.cursor = tool === "eraser" ? "crosshair" : "default";
        this.render();
    }

    scheduleUpload() {
        this.needsUpload = true;
        if (this.uploadTimeout) clearTimeout(this.uploadTimeout);

        this.uploadTimeout = setTimeout(() => {
            this.uploadCompositionData();
        }, 250);
    }

    uploadCompositionData() {
        if (!this.needsUpload) return;

        const transformSnapshot = this.getTransformSnapshot();
        let maskData = this.lastMaskData;
        debugLog("Scheduling upload", {
            needsUpload: this.needsUpload,
            snapshot: transformSnapshot,
            maskDirty: this.maskDirty
        });
        if (this.foregroundMask) {
            if (this.maskDirty || !maskData) {
                maskData = this.foregroundMask.toDataURL("image/png");
                this.lastMaskData = maskData;
                this.maskDirty = false;
                debugLog("Captured mask data from canvas");
            }
        }

        const payload = {
            foreground: transformSnapshot,
            saved: false
        };

        if (maskData) {
            payload.mask = maskData;
        }

        this.commitCompositionData(payload);
        this.needsUpload = false;
        this.uploadTimeout = null;
    }

    flushCompositionData() {
        if (this.uploadTimeout) {
            clearTimeout(this.uploadTimeout);
            this.uploadTimeout = null;
        }
        if (this.needsUpload) {
            this.uploadCompositionData();
        }
    }

    getTransformSnapshot() {
        const t = this.fgTransform;
        return {
            x: t.x,
            y: t.y,
            scaleX: t.scaleX,
            scaleY: t.scaleY,
            rotation: t.rotation,
            width: t.width,
            height: t.height
        };
    }

    commitCompositionData(payload) {
        const data = JSON.stringify(payload);
        const widget = this.node.widgets.find(w => w.name === "composition_data");
        if (widget) {
            widget.value = data;
            if (typeof widget.callback === "function") {
                widget.callback(widget.value);
            }
            if (typeof this.node.onWidgetChanged === "function") {
                this.node.onWidgetChanged(widget);
            }
        }
        this.node.widgets_changed = true;
        if (this.node.graph && typeof this.node.graph.setDirtyCanvas === "function") {
            this.node.graph.setDirtyCanvas(true, true);
        }
        if (app && app.graph && typeof app.graph.setDirtyCanvas === "function") {
            app.graph.setDirtyCanvas(true, true);
        }
        debugLog("Committed composition data", payload);
        return data;
    }

    async saveComposition() {
        if (!this.canvas) return;

        const dataURL = this.canvas.toDataURL("image/png");
        const maskURL = this.foregroundMask ? this.foregroundMask.toDataURL("image/png") : null;
        const filename = this.node.widgets.find(w => w.name === "output_filename")?.value || "rc_composition.png";

        try {
            const response = await api.fetchApi("/rc_compositor/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: dataURL,
                    mask: maskURL,
                    filename: filename
                })
            });

            const result = await response.json();

            if (result.status === "success") {
                const transformSnapshot = this.getTransformSnapshot();
                if (maskURL) {
                    this.lastMaskData = maskURL;
                    this.maskDirty = false;
                } else {
                    this.lastMaskData = null;
                    this.maskDirty = false;
                }

                const payload = {
                    foreground: transformSnapshot,
                    saved: true
                };

                if (maskURL) {
                    payload.mask = maskURL;
                }

                this.commitCompositionData(payload);

                this.needsUpload = false;
                if (this.uploadTimeout) {
                    clearTimeout(this.uploadTimeout);
                    this.uploadTimeout = null;
                }

                const autoContinue = this.node.widgets.find(w => w.name === "auto_continue")?.value;
                if (autoContinue) {
                    await app.queuePrompt(0, 1);
                }

                console.log("Composition saved:", filename);
            }
        } catch (error) {
            console.error("Save failed:", error);
        }
    }

    createToolbar() {
        const toolbar = document.createElement("div");
        toolbar.className = "rc-compositor-toolbar";
        toolbar.innerHTML = `
            <button class="rc-tool-btn active" data-tool="transform" title="Transform (T)">
                <svg viewBox="0 0 1024 1024" width="20" height="20">
                    <path d="M486.4 776.533333v-213.333333H247.466667v106.666667L85.333333 512l162.133334-162.133333V512h238.933333V247.466667H349.866667L512 85.333333l162.133333 162.133334h-132.266666V512h238.933333V349.866667L938.666667 512l-162.133334 162.133333v-106.666666h-238.933333v213.333333h132.266667L512 938.666667l-162.133333-162.133334h136.533333z" fill="currentColor"/>
                </svg>
            </button>
            <button class="rc-tool-btn" data-tool="eraser" title="Eraser (E)">
                <svg viewBox="0 0 1024 1024" width="20" height="20">
                    <path d="M567.024534 149.331607m60.339779 60.339778l199.422968 199.422969q60.339779 60.339779 0 120.679557l-184.941421 184.941422q-60.339779 60.339779-120.679558 0l-199.422968-199.422969q-60.339779-60.339779 0-120.679557l184.941421-184.941422q60.339779-60.339779 120.679558 0Z" fill="currentColor"/>
                    <path d="M557.653333 256l211.2 213.333333-302.08 298.666667H346.88l-151.466667-151.466667L557.653333 256m0-85.333333a85.333333 85.333333 0 0 0-60.586666 24.746666L135.253333 554.666667a85.333333 85.333333 0 0 0 0 120.746666L311.466667 853.333333h190.293333l327.253333-327.253333a85.333333 85.333333 0 0 0 0-120.746667l-211.2-211.2A85.333333 85.333333 0 0 0 557.653333 170.666667z" fill="currentColor"/>
                    <path d="M332.8 768m42.666667 0l469.333333 0q42.666667 0 42.666667 42.666667l0 0q0 42.666667-42.666667 42.666666l-469.333333 0q-42.666667 0-42.666667-42.666666l0 0q0-42.666667 42.666667-42.666667Z" fill="currentColor"/>
                </svg>
            </button>
            <label class="rc-toolbar-label">
                Size: <input type="range" class="rc-eraser-size" min="10" max="100" value="30" step="5">
                <span class="rc-size-value">30</span>
            </label>
            <button class="rc-tool-btn" data-action="reset" title="Reset Mask">
                <span>↺</span>
            </button>
        `;

        toolbar.querySelectorAll(".rc-tool-btn[data-tool]").forEach(btn => {
            btn.addEventListener("click", () => {
                this.setTool(btn.dataset.tool);
                toolbar.querySelectorAll(".rc-tool-btn[data-tool]").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
            });
        });

        const sizeSlider = toolbar.querySelector(".rc-eraser-size");
        const sizeValue = toolbar.querySelector(".rc-size-value");
        sizeSlider.addEventListener("input", (e) => {
            this.eraserSize = parseInt(e.target.value);
            sizeValue.textContent = e.target.value;
        });

        toolbar.querySelector("[data-action='reset']").addEventListener("click", () => {
            this.resetMask();
        });

        this.toolbar = toolbar;

        return toolbar;
    }
}

// ComfyUI Extension
app.registerExtension({
    name: "RC.InteractiveCompositor",

    async setup() {
        api.addEventListener("rc_compositor_init", (event) => {
            const { output, node: nodeId } = event.detail;
            const node = app.graph.getNodeById(nodeId);

            if (!node || !node.compositorEditor) return;

            const editor = node.compositorEditor;

            editor.beginInitialize();

            editor.setCanvasSize(
                output.canvas_width[0],
                output.canvas_height[0]
            );

            if (output.background_image[0]) {
                editor.loadBackground(output.background_image[0]);
            }
            if (output.foreground_image[0]) {
                editor.loadForeground(output.foreground_image[0]);
            }

            if (output.composition_data && output.composition_data[0]) {
                try {
                    const state = JSON.parse(output.composition_data[0]);
                    editor.setCompositionState(state);
                } catch (err) {
                    console.warn("Failed to parse composition state", err);
                }
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "RC_InteractiveCompositor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Hide composition_data widget
                const dataWidget = this.widgets.find(w => w.name === "composition_data");
                if (dataWidget) hideWidget(this, dataWidget);

                const containerDiv = document.createElement("div");
                containerDiv.className = "rc-compositor-container";

                const editor = new RCCompositorEditor(this, containerDiv);
                this.compositorEditor = editor;

                const toolbar = editor.createToolbar();
                containerDiv.appendChild(toolbar);

                const canvas = document.createElement("canvas");
                canvas.id = `rc_compositor_${this.id}`;
                canvas.width = 1024;
                canvas.height = 900;
                containerDiv.appendChild(canvas);

                this.addDOMWidget("compositor_canvas", "canvas", containerDiv, {
                    serialize: false,
                    hideOnZoom: false
                });

                const domWrapper = containerDiv.parentElement;
                if (domWrapper && domWrapper.classList?.contains("dom-widget")) {
                    domWrapper.style.top = "16px";
                }

                // Set minimum node size and prevent collapsing
                this.size = [600, 900];

                // Override setSize to enforce minimum dimensions
                const originalSetSize = this.setSize.bind(this);
                this.setSize = function(size) {
                    const minWidth = 500;
                    const minHeight = 900; // Increased to show more canvas
                    const newSize = [
                        Math.max(minWidth, size[0]),
                        Math.max(minHeight, size[1])
                    ];
                    originalSetSize(newSize);
                };

                setTimeout(() => {
                    editor.init(canvas);
                    editor.setCanvasSize(1024, 900);
                }, 100);

                return result;
            };
        }
    }
});

// CSS
const style = document.createElement("style");
style.textContent = `
.rc-compositor-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    background: #1a1a1a;
    padding: 12px 12px 16px;
    padding-top: 16px;
    border-radius: 8px;
    margin-top: 6px;
}

.rc-compositor-container canvas {
    border: 2px solid #333;
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}

.rc-compositor-toolbar {
    display: flex;
    gap: 10px;
    padding: 10px;
    background: #2b2b2b;
    border-radius: 6px;
    align-items: center;
    flex-wrap: wrap;
}

.rc-tool-btn {
    padding: 10px 18px;
    background: #3a3a3a;
    border: 1px solid #555;
    color: #fff;
    cursor: pointer;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
}

.rc-tool-btn:hover {
    background: #4a4a4a;
    border-color: #777;
    transform: translateY(-1px);
}

.rc-tool-btn.active {
    background: #5a5aff;
    border-color: #7a7aff;
    box-shadow: 0 2px 8px rgba(90,90,255,0.3);
}

.rc-save-btn {
    background: #4a9eff !important;
    border-color: #6abaff !important;
    margin-left: auto;
}

.rc-save-btn:hover {
    background: #6abaff !important;
    box-shadow: 0 2px 8px rgba(74,158,255,0.4);
}

.rc-toolbar-label {
    color: #fff;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.rc-eraser-size {
    width: 120px;
}

.rc-size-value {
    min-width: 30px;
    text-align: center;
    font-weight: 600;
    color: #4a9eff;
}
`;
document.head.appendChild(style);
