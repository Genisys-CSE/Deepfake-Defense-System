"use strict";

document.addEventListener("DOMContentLoaded", () => {
    const state = {
        screen: "landing",
        redTool: "protect",
        busy: {
            protect: false,
            swap: false,
            detect: false,
        },
        pipeline: {
            timerId: null,
            stageId: null,
            startedAt: 0,
            currentStage: -1,
            active: false,
        },
        taskProgress: {
            protect: { timerId: null, value: 0 },
            swap: { timerId: null, value: 0 },
            detect: { timerId: null, value: 0 },
        },
    };

    const el = {
        screens: Array.from(document.querySelectorAll(".screen")),
        redToolButtons: Array.from(document.querySelectorAll("[data-red-tool]")),
        redToolViews: Array.from(document.querySelectorAll("[data-red-view]")),
        openScreenButtons: Array.from(document.querySelectorAll("[data-open-screen]")),
        navScrollButtons: Array.from(document.querySelectorAll("[data-scroll-target]")),
        footerLinks: Array.from(document.querySelectorAll(".footer-links a")),
        missionGrid: document.getElementById("mission-grid"),
        startSystemBtn: document.getElementById("start-system-btn"),
        brandHomeBtn: document.getElementById("brand-home"),

        protectFile: document.getElementById("protect-file"),
        detectFile: document.getElementById("detect-file"),
        swapOrigFile: document.getElementById("swap-orig-file"),
        swapProtFile: document.getElementById("swap-prot-file"),
        swapTargetFile: document.getElementById("swap-target-file"),

        protectBtn: document.getElementById("protect-btn"),
        swapBtn: document.getElementById("swap-btn"),
        detectBtn: document.getElementById("detect-btn"),

        protectError: document.getElementById("protect-error"),
        swapError: document.getElementById("swap-error"),
        detectError: document.getElementById("detect-error"),

        protectResults: document.getElementById("protect-results"),
        swapResults: document.getElementById("swap-results"),
        detectResults: document.getElementById("detect-results"),

        verdictCard: document.getElementById("verdict-card"),
        verdictLabel: document.getElementById("verdict-label"),
        verdictConfidence: document.getElementById("verdict-confidence"),
        detectSummary: document.getElementById("detect-summary"),
        detectReasons: document.getElementById("detect-reasons"),

        pipelineStages: Array.from(document.querySelectorAll("#detect-pipeline .pipeline-stage")),
        pipeProgressCount: document.getElementById("pipe-progress-count"),
        pipeProgressFill: document.getElementById("pipe-progress-fill"),
        pipeTimer: document.getElementById("pipe-timer"),

        protectProgressWrap: document.getElementById("protect-progress-wrap"),
        protectProgressFill: document.getElementById("protect-progress-fill"),
        protectProgressLabel: document.getElementById("protect-progress-label"),
        protectProgressValue: document.getElementById("protect-progress-value"),

        swapProgressWrap: document.getElementById("swap-progress-wrap"),
        swapProgressFill: document.getElementById("swap-progress-fill"),
        swapProgressLabel: document.getElementById("swap-progress-label"),
        swapProgressValue: document.getElementById("swap-progress-value"),

        detectProgressWrap: document.getElementById("detect-progress-wrap"),
        detectProgressFill: document.getElementById("detect-progress-fill"),
        detectProgressLabel: document.getElementById("detect-progress-label"),
        detectProgressValue: document.getElementById("detect-progress-value"),
    };

    init();

    function init() {
        bindNavigation();
        bindMissionTools();
        bindUploads();
        bindActions();
        showScreen("landing");
        setRedTool("protect");
        resetPipeline();
        refreshActionButtons();
    }

    function bindNavigation() {
        el.openScreenButtons.forEach((btn) => {
            btn.addEventListener("click", () => showScreen(btn.dataset.openScreen));
        });

        if (el.brandHomeBtn) {
            el.brandHomeBtn.addEventListener("click", () => {
                showScreen("landing");
                window.scrollTo({ top: 0, behavior: "smooth" });
            });
        }

        if (el.startSystemBtn && el.missionGrid) {
            el.startSystemBtn.addEventListener("click", () => {
                el.missionGrid.scrollIntoView({ behavior: "smooth", block: "start" });
            });
        }

        el.navScrollButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                const targetId = btn.dataset.scrollTarget;
                if (!targetId) return;
                if (state.screen !== "landing") {
                    showScreen("landing");
                    window.setTimeout(() => scrollToSection(targetId), 260);
                    return;
                }
                scrollToSection(targetId);
            });
        });

        el.footerLinks.forEach((anchor) => {
            anchor.addEventListener("click", (event) => {
                const href = anchor.getAttribute("href") || "";
                if (!href.startsWith("#")) return;
                event.preventDefault();
                const targetId = href.slice(1);
                if (!targetId) return;
                if (state.screen !== "landing") {
                    showScreen("landing");
                    window.setTimeout(() => scrollToSection(targetId), 260);
                    return;
                }
                scrollToSection(targetId);
            });
        });
    }

    function bindMissionTools() {
        el.redToolButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                const nextTool = btn.dataset.redTool;
                if (!nextTool) return;
                setRedTool(nextTool);
            });
        });
    }

    function bindUploads() {
        setupUploadZone({
            zoneId: "protect-upload",
            input: el.protectFile,
            previewId: "protect-preview",
            onFileChange: () => {
                clearInlineError(el.protectError);
                refreshActionButtons();
            },
        });

        setupUploadZone({
            zoneId: "swap-orig-upload",
            input: el.swapOrigFile,
            previewId: "swap-orig-preview",
            onFileChange: () => {
                clearInlineError(el.swapError);
                refreshActionButtons();
            },
        });

        setupUploadZone({
            zoneId: "swap-prot-upload",
            input: el.swapProtFile,
            previewId: "swap-prot-preview",
            onFileChange: () => {
                clearInlineError(el.swapError);
                refreshActionButtons();
            },
        });

        setupUploadZone({
            zoneId: "swap-target-upload",
            input: el.swapTargetFile,
            previewId: "swap-target-preview",
            onFileChange: () => {
                clearInlineError(el.swapError);
                refreshActionButtons();
            },
        });

        setupUploadZone({
            zoneId: "detect-upload",
            input: el.detectFile,
            previewId: "detect-preview",
            onFileChange: () => {
                clearInlineError(el.detectError);
                refreshActionButtons();
            },
        });
    }

    function bindActions() {
        if (el.protectBtn) {
            el.protectBtn.addEventListener("click", () => {
                void runProtect();
            });
        }

        if (el.swapBtn) {
            el.swapBtn.addEventListener("click", () => {
                void runSwap();
            });
        }

        if (el.detectBtn) {
            el.detectBtn.addEventListener("click", () => {
                void runDetect();
            });
        }
    }

    function showScreen(nextScreen) {
        if (!nextScreen) return;
        state.screen = nextScreen;
        el.screens.forEach((screenEl) => {
            const isActive = screenEl.dataset.screen === nextScreen;
            screenEl.classList.toggle("active", isActive);
        });
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function setRedTool(toolName) {
        state.redTool = toolName;
        el.redToolButtons.forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.redTool === toolName);
        });
        el.redToolViews.forEach((view) => {
            view.classList.toggle("active", view.dataset.redView === toolName);
        });
    }

    function scrollToSection(sectionId) {
        const target = document.getElementById(sectionId);
        if (!target) return;
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function setupUploadZone({ zoneId, input, previewId, onFileChange }) {
        const zone = document.getElementById(zoneId);
        const preview = document.getElementById(previewId);
        if (!zone || !input || !preview) return;

        zone.addEventListener("click", (event) => {
            if (event.target instanceof HTMLElement && event.target.tagName.toLowerCase() === "label") {
                return;
            }
            input.click();
        });

        zone.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                input.click();
            }
        });

        zone.addEventListener("dragover", (event) => {
            event.preventDefault();
            zone.classList.add("dragover");
        });

        zone.addEventListener("dragleave", () => {
            zone.classList.remove("dragover");
        });

        zone.addEventListener("drop", (event) => {
            event.preventDefault();
            zone.classList.remove("dragover");
            const droppedFile = event.dataTransfer?.files?.[0];
            if (!droppedFile) return;
            assignInputFile(input, droppedFile);
            renderPreview(zone, preview, droppedFile);
            onFileChange();
        });

        input.addEventListener("change", () => {
            const selected = input.files?.[0];
            if (!selected) {
                clearPreview(zone, preview);
                onFileChange();
                return;
            }
            renderPreview(zone, preview, selected);
            onFileChange();
        });
    }

    function assignInputFile(input, file) {
        try {
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
        } catch (_) {
            /* no-op: file picker remains the fallback path */
        }
    }

    function renderPreview(zone, preview, file) {
        const oldUrl = preview.dataset.objectUrl;
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        const objectUrl = URL.createObjectURL(file);
        preview.src = objectUrl;
        preview.dataset.objectUrl = objectUrl;
        zone.classList.add("has-image");
    }

    function clearPreview(zone, preview) {
        const oldUrl = preview.dataset.objectUrl;
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        preview.removeAttribute("src");
        delete preview.dataset.objectUrl;
        zone.classList.remove("has-image");
    }

    function refreshActionButtons() {
        if (el.protectBtn) {
            el.protectBtn.disabled = state.busy.protect || !hasFile(el.protectFile);
        }
        if (el.swapBtn) {
            const ready = hasFile(el.swapOrigFile) && hasFile(el.swapProtFile) && hasFile(el.swapTargetFile);
            el.swapBtn.disabled = state.busy.swap || !ready;
        }
        if (el.detectBtn) {
            el.detectBtn.disabled = state.busy.detect || !hasFile(el.detectFile);
        }
    }

    function hasFile(input) {
        return !!(input && input.files && input.files.length > 0);
    }

    async function runProtect() {
        if (!hasFile(el.protectFile) || state.busy.protect) return;

        clearInlineError(el.protectError);
        setBusy("protect", true);
        startTaskProgress("protect");

        const form = new FormData();
        form.append("image", el.protectFile.files[0]);

        try {
            const data = await postForm("/api/protect", form);
            const metrics = data.metrics || {};

            setImage("result-original", data.original);
            setImage("result-protected", data.protected);

            const psnr = toFinite(metrics.psnr);
            const ssim = toFinite(metrics.ssim);
            const arcface = toFinite(metrics.arcface_similarity);
            const facenet = toFinite(metrics.facenet_similarity);

            setBar("bar-psnr", psnr === null ? 0 : (psnr / 50) * 100);
            setText("val-psnr", psnr === null ? "--" : `${psnr.toFixed(2)} dB`);

            setBar("bar-ssim", ssim === null ? 0 : ssim * 100);
            setText("val-ssim", ssim === null ? "--" : ssim.toFixed(4));

            setBar("bar-arcface", arcface === null ? 0 : similarityToPercent(arcface));
            setText("val-arcface", arcface === null ? "--" : arcface.toFixed(4));

            setBar("bar-facenet", facenet === null ? 0 : similarityToPercent(facenet));
            setText("val-facenet", facenet === null ? "--" : facenet.toFixed(4));

            setText("protect-elapsed", formatElapsed(data.elapsed));
            revealResult(el.protectResults);
            completeTaskProgress("protect", true);
        } catch (error) {
            setInlineError(el.protectError, error.message);
            completeTaskProgress("protect", false);
        } finally {
            setBusy("protect", false);
        }
    }

    async function runSwap() {
        const ready = hasFile(el.swapOrigFile) && hasFile(el.swapProtFile) && hasFile(el.swapTargetFile);
        if (!ready || state.busy.swap) return;

        clearInlineError(el.swapError);
        setBusy("swap", true);
        startTaskProgress("swap");

        const form = new FormData();
        form.append("source_original", el.swapOrigFile.files[0]);
        form.append("source_protected", el.swapProtFile.files[0]);
        form.append("target", el.swapTargetFile.files[0]);

        try {
            const data = await postForm("/api/swap", form);
            const clean = data.clean_swap || {};
            const protectedSwap = data.protected_swap || {};

            setImage("swap-clean-img", clean.image);
            setImage("swap-prot-img", protectedSwap.image);

            const cleanConf = confidenceToPercent(clean.identity_confidence);
            const protConf = confidenceToPercent(protectedSwap.identity_confidence);

            setBar("conf-clean", cleanConf);
            setBar("conf-prot", protConf);
            setText("conf-clean-val", `${cleanConf.toFixed(1)}%`);
            setText("conf-prot-val", `${protConf.toFixed(1)}%`);

            setSwapBadge("swap-clean-header", clean.quality);
            setSwapBadge("swap-prot-header", protectedSwap.quality);

            const errors = [clean.error, protectedSwap.error].filter(Boolean);
            if (errors.length > 0) {
                setInlineError(el.swapError, errors.join(" | "));
            }

            setText("swap-elapsed", formatElapsed(data.elapsed));
            revealResult(el.swapResults);
            completeTaskProgress("swap", true);
        } catch (error) {
            setInlineError(el.swapError, error.message);
            completeTaskProgress("swap", false);
        } finally {
            setBusy("swap", false);
        }
    }

    async function runDetect() {
        if (!hasFile(el.detectFile) || state.busy.detect) return;

        clearInlineError(el.detectError);
        resetPipeline();
        startPipeline();
        setBusy("detect", true);
        startTaskProgress("detect");

        const form = new FormData();
        form.append("image", el.detectFile.files[0]);

        try {
            const data = await postForm("/api/detect", form);
            completePipelineSuccess();
            renderDetectResult(data);
            revealResult(el.detectResults);
            completeTaskProgress("detect", true);
        } catch (error) {
            completePipelineFailure();
            setInlineError(el.detectError, error.message);
            completeTaskProgress("detect", false);
        } finally {
            stopPipelineTimers();
            setBusy("detect", false);
        }
    }

    function renderDetectResult(data) {
        const label = String(data.label || "UNKNOWN").toUpperCase();
        const confidence = toFinite(data.confidence);
        const fakeProbability = toFinite(data.fake_probability);
        const analysis = data.analysis || {};
        const explanation = data.explanation || {};

        setText("verdict-label", label);
        const confidenceText = confidence === null ? "--" : `${(confidence * 100).toFixed(1)}%`;
        const fakeProbText = fakeProbability === null ? "--" : `${(fakeProbability * 100).toFixed(1)}%`;
        setText("verdict-confidence", `Confidence: ${confidenceText} | Fake Probability: ${fakeProbText}`);

        el.verdictCard.classList.remove("real", "fake");
        if (label.includes("FAKE")) {
            el.verdictCard.classList.add("fake");
        } else if (label.includes("REAL")) {
            el.verdictCard.classList.add("real");
        }

        if (typeof data.spectrum === "string" && data.spectrum.length > 0) {
            setImage("spectrum-img", data.spectrum);
        }

        renderAnalysisBar("frequency", analysis.frequency);
        renderAnalysisBar("noise", analysis.noise);
        renderAnalysisBar("boundary", analysis.boundary);
        renderAnalysisBar("sharpness", analysis.sharpness);
        renderAnalysisBar("compression", analysis.compression);
        renderAnalysisBar("color", analysis.color);

        setText("detect-summary", explanation.summary || "Signal breakdown is complete.");
        renderReasonList(explanation.reasons || []);

        setText("detect-elapsed", formatElapsed(data.elapsed));
    }

    function renderAnalysisBar(key, value) {
        const numeric = toFinite(value);
        const pct = numeric === null ? 0 : clamp(numeric * 100);
        setBar(`abar-${key}`, pct);
        setText(`aval-${key}`, numeric === null ? "--" : `${pct.toFixed(0)}%`);

        const bar = document.getElementById(`abar-${key}`);
        if (!bar) return;
        if (pct >= 60) {
            bar.style.background = "linear-gradient(90deg, #ff6f82, #ff9a66)";
            return;
        }
        if (pct >= 35) {
            bar.style.background = "linear-gradient(90deg, #ffad63, #ffd06a)";
            return;
        }
        bar.style.background = "linear-gradient(90deg, #2ad881, #44f0aa)";
    }

    function resetPipeline() {
        stopPipelineTimers();
        state.pipeline.currentStage = -1;
        state.pipeline.active = false;
        updatePipelineProgress(0);
        if (el.pipeTimer) el.pipeTimer.textContent = "00:00";

        el.pipelineStages.forEach((stage) => {
            stage.classList.remove("running", "done", "failed");
            stage.classList.add("pending");
            const status = stage.querySelector(".stage-status");
            if (status) status.textContent = "PENDING";
        });
    }

    function startPipeline() {
        if (el.pipelineStages.length === 0) return;
        state.pipeline.active = true;
        state.pipeline.startedAt = Date.now();
        state.pipeline.currentStage = -1;

        advancePipelineStep();
        state.pipeline.stageId = window.setInterval(advancePipelineStep, 900);
        state.pipeline.timerId = window.setInterval(updatePipelineTimer, 1000);
        updatePipelineTimer();
    }

    function advancePipelineStep() {
        if (!state.pipeline.active || el.pipelineStages.length === 0) return;
        const lastIndex = el.pipelineStages.length - 1;
        const previousIndex = state.pipeline.currentStage;

        if (previousIndex >= 0 && previousIndex < lastIndex) {
            markPipelineStage(previousIndex, "done");
        }

        if (state.pipeline.currentStage < lastIndex) {
            state.pipeline.currentStage += 1;
            markPipelineStage(state.pipeline.currentStage, "running");
        } else {
            markPipelineStage(lastIndex, "running");
        }

        const shownProgress = Math.min(state.pipeline.currentStage + 1, el.pipelineStages.length);
        updatePipelineProgress(shownProgress);
    }

    function completePipelineSuccess() {
        if (el.pipelineStages.length === 0) return;
        el.pipelineStages.forEach((_, index) => markPipelineStage(index, "done"));
        updatePipelineProgress(el.pipelineStages.length);
        state.pipeline.active = false;
    }

    function completePipelineFailure() {
        if (el.pipelineStages.length === 0) return;
        const idx = state.pipeline.currentStage >= 0 ? state.pipeline.currentStage : 0;
        markPipelineStage(idx, "failed");
        state.pipeline.active = false;
    }

    function markPipelineStage(index, statusName) {
        const stage = el.pipelineStages[index];
        if (!stage) return;

        stage.classList.remove("pending", "running", "done", "failed");
        stage.classList.add(statusName);

        const statusLabel = stage.querySelector(".stage-status");
        if (!statusLabel) return;

        if (statusName === "done") {
            statusLabel.textContent = "DONE";
            return;
        }
        if (statusName === "running") {
            statusLabel.textContent = "RUNNING";
            return;
        }
        if (statusName === "failed") {
            statusLabel.textContent = "FAILED";
            return;
        }
        statusLabel.textContent = "PENDING";
    }

    function updatePipelineProgress(progressCount) {
        const total = el.pipelineStages.length || 7;
        if (el.pipeProgressCount) {
            el.pipeProgressCount.textContent = `${progressCount} / ${total} stages`;
        }
        if (el.pipeProgressFill) {
            el.pipeProgressFill.style.width = `${clamp((progressCount / total) * 100)}%`;
        }
    }

    function updatePipelineTimer() {
        if (!el.pipeTimer) return;
        const elapsedSec = Math.max(0, Math.floor((Date.now() - state.pipeline.startedAt) / 1000));
        const minutes = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
        const seconds = String(elapsedSec % 60).padStart(2, "0");
        el.pipeTimer.textContent = `${minutes}:${seconds}`;
    }

    function stopPipelineTimers() {
        if (state.pipeline.timerId) {
            window.clearInterval(state.pipeline.timerId);
            state.pipeline.timerId = null;
        }
        if (state.pipeline.stageId) {
            window.clearInterval(state.pipeline.stageId);
            state.pipeline.stageId = null;
        }
    }

    function setSwapBadge(headerId, qualityRaw) {
        const header = document.getElementById(headerId);
        if (!header) return;

        let badge = header.querySelector(".badge");
        if (!badge) {
            badge = document.createElement("span");
            badge.className = "badge";
            header.innerHTML = "";
            header.appendChild(badge);
        }

        badge.classList.remove("high", "degraded", "failed");

        const quality = String(qualityRaw || "FAILED").toUpperCase();
        if (quality === "HIGH") {
            badge.classList.add("high");
            badge.textContent = "HIGH QUALITY";
            return;
        }
        if (quality === "DEGRADED") {
            badge.classList.add("degraded");
            badge.textContent = "DEGRADED";
            return;
        }
        badge.classList.add("failed");
        badge.textContent = "FAILED";
    }

    async function postForm(url, formData) {
        const response = await fetch(url, {
            method: "POST",
            body: formData,
        });

        let payload;
        try {
            payload = await response.json();
        } catch (_) {
            payload = {};
        }

        if (!response.ok) {
            const errorMessage = payload.error || payload.message || `Server error: ${response.status}`;
            throw new Error(errorMessage);
        }
        if (payload.error) {
            throw new Error(payload.error);
        }
        return payload;
    }

    function renderReasonList(reasons) {
        if (!el.detectReasons) return;
        el.detectReasons.innerHTML = "";
        const entries = Array.isArray(reasons) ? reasons.slice(0, 4) : [];
        if (entries.length === 0) {
            const item = document.createElement("li");
            item.textContent = "No additional reasoning available for this scan.";
            el.detectReasons.appendChild(item);
            return;
        }
        entries.forEach((reason) => {
            const item = document.createElement("li");
            item.textContent = String(reason);
            el.detectReasons.appendChild(item);
        });
    }

    function startTaskProgress(flow) {
        const ui = getTaskProgressUi(flow);
        const tracker = state.taskProgress[flow];
        if (!ui || !tracker) return;
        stopTaskProgress(flow);
        tracker.value = 2;
        ui.wrap.hidden = false;
        updateTaskProgressUi(flow, tracker.value);
        const profile = getTaskProfile(flow);
        let stageIndex = 0;
        updateTaskLabel(flow, profile.labels[stageIndex]);
        tracker.timerId = window.setInterval(() => {
            if (tracker.value >= profile.cap) return;
            tracker.value = Math.min(
                profile.cap,
                tracker.value + profile.stepMin + Math.random() * (profile.stepMax - profile.stepMin)
            );
            const nextStage = Math.min(
                profile.labels.length - 1,
                Math.floor((tracker.value / profile.cap) * profile.labels.length)
            );
            if (nextStage !== stageIndex) {
                stageIndex = nextStage;
                updateTaskLabel(flow, profile.labels[stageIndex]);
            }
            updateTaskProgressUi(flow, tracker.value);
        }, 420);
    }

    function completeTaskProgress(flow, success) {
        const ui = getTaskProgressUi(flow);
        const tracker = state.taskProgress[flow];
        if (!ui || !tracker) return;
        stopTaskProgress(flow);
        tracker.value = success ? 100 : Math.max(12, tracker.value);
        updateTaskProgressUi(flow, tracker.value);
        updateTaskLabel(flow, success ? "Completed." : "Request failed.");
        window.setTimeout(() => {
            if (!state.busy[flow]) {
                ui.wrap.hidden = true;
                tracker.value = 0;
                updateTaskProgressUi(flow, 0);
            }
        }, success ? 900 : 1400);
    }

    function stopTaskProgress(flow) {
        const tracker = state.taskProgress[flow];
        if (!tracker) return;
        if (tracker.timerId) {
            window.clearInterval(tracker.timerId);
            tracker.timerId = null;
        }
    }

    function updateTaskProgressUi(flow, value) {
        const ui = getTaskProgressUi(flow);
        if (!ui) return;
        const safe = clamp(value);
        ui.fill.style.width = `${safe}%`;
        ui.value.textContent = `${safe.toFixed(0)}%`;
    }

    function updateTaskLabel(flow, text) {
        const ui = getTaskProgressUi(flow);
        if (!ui) return;
        ui.label.textContent = text;
    }

    function getTaskProfile(flow) {
        if (flow === "protect") {
            return {
                cap: 94,
                stepMin: 0.6,
                stepMax: 1.8,
                labels: [
                    "Initializing protection graph...",
                    "Running adversarial optimization...",
                    "Applying frequency-layer shielding...",
                    "Calibrating identity disruption...",
                    "Final quality validation..."
                ],
            };
        }
        if (flow === "swap") {
            return {
                cap: 92,
                stepMin: 1.2,
                stepMax: 3.2,
                labels: [
                    "Preparing source/target faces...",
                    "Running clean source swap...",
                    "Running protected source swap...",
                    "Scoring identity transfer..."
                ],
            };
        }
        return {
            cap: 90,
            stepMin: 1.0,
            stepMax: 2.6,
            labels: [
                "Loading media and normalizing input...",
                "Computing forensic feature maps...",
                "Scoring anomaly signals...",
                "Assembling verdict explanation..."
            ],
        };
    }

    function getTaskProgressUi(flow) {
        if (flow === "protect") {
            return {
                wrap: el.protectProgressWrap,
                fill: el.protectProgressFill,
                label: el.protectProgressLabel,
                value: el.protectProgressValue,
            };
        }
        if (flow === "swap") {
            return {
                wrap: el.swapProgressWrap,
                fill: el.swapProgressFill,
                label: el.swapProgressLabel,
                value: el.swapProgressValue,
            };
        }
        if (flow === "detect") {
            return {
                wrap: el.detectProgressWrap,
                fill: el.detectProgressFill,
                label: el.detectProgressLabel,
                value: el.detectProgressValue,
            };
        }
        return null;
    }

    function setBusy(flow, isBusy) {
        state.busy[flow] = isBusy;
        if (!isBusy) {
            stopTaskProgress(flow);
        }
        const btnMap = {
            protect: el.protectBtn,
            swap: el.swapBtn,
            detect: el.detectBtn,
        };
        const button = btnMap[flow];
        if (button) {
            button.classList.toggle("loading", isBusy);
        }
        refreshActionButtons();
    }

    function revealResult(container) {
        if (!container) return;
        container.hidden = false;
        container.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function setBar(id, percentage) {
        const node = document.getElementById(id);
        if (!node) return;
        node.style.width = `${clamp(percentage)}%`;
    }

    function setText(id, value) {
        const node = document.getElementById(id);
        if (!node) return;
        node.textContent = value;
    }

    function setImage(id, source) {
        const node = document.getElementById(id);
        if (!node || typeof source !== "string" || source.length === 0) return;
        node.src = source;
    }

    function setInlineError(node, message) {
        if (!node) return;
        node.textContent = message || "An unexpected error occurred.";
    }

    function clearInlineError(node) {
        if (!node) return;
        node.textContent = "";
    }

    function toFinite(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return null;
        return numeric;
    }

    function confidenceToPercent(value) {
        const numeric = toFinite(value);
        if (numeric === null) return 0;
        return clamp(numeric * 100);
    }

    function similarityToPercent(value) {
        if (value >= -1 && value <= 1) {
            return clamp(((value + 1) / 2) * 100);
        }
        return clamp(value * 100);
    }

    function formatElapsed(elapsed) {
        const numeric = toFinite(elapsed);
        if (numeric === null) return "";
        return `Completed in ${numeric.toFixed(2)}s`;
    }

    function clamp(value) {
        return Math.min(100, Math.max(0, Number.isFinite(value) ? value : 0));
    }
});
