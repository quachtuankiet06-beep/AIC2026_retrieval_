// ==========================================
// CLIENT-SIDE APPLICATION LOGIC - AIC 2026
// ==========================================

document.addEventListener("DOMContentLoaded", () => {
    initSidebarControls();
    initSearchEvents();
    initConverter();
});

// Format seconds -> HH:MM:SS or MM:SS
function formatHms(seconds) {
    const totalSec = Math.max(0, Number(seconds) || 0);
    const hrs = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    const secs = Math.floor(totalSec % 60);

    if (hrs > 0) {
        return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

// Copy text to clipboard and show toast
window.copyText = function(text) {
    if (!text) return;
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showToast(`Đã sao chép: ${text}`, "success");
        }).catch(() => {
            fallbackCopyText(text);
        });
    } else {
        fallbackCopyText(text);
    }
};

function fallbackCopyText(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
        document.execCommand("copy");
        showToast(`Đã sao chép: ${text}`, "success");
    } catch (err) {
        showToast("Không thể sao chép tự động", "error");
    }
    document.body.removeChild(textArea);
}

window.showToast = function(msg, type = "success") {
    const toast = document.getElementById("toast");
    const toastMsg = document.getElementById("toastMsg");
    if (!toast || !toastMsg) return;

    toastMsg.innerText = msg;
    const icon = toast.querySelector("i");
    if (icon) {
        if (type === "info") {
            icon.className = "fa-solid fa-circle-info";
        } else if (type === "error") {
            icon.className = "fa-solid fa-triangle-exclamation";
        } else {
            icon.className = "fa-solid fa-circle-check";
        }
    }

    toast.classList.add("show");
    clearTimeout(window._toastTimeout);
    window._toastTimeout = setTimeout(() => {
        toast.classList.remove("show");
    }, 2500);
};

// ==========================================
// 1. SIDEBAR CONTROLS
// ==========================================
function initSidebarControls() {
    const searchType = document.getElementById("searchType");
    const queryInput = document.getElementById("queryInput");
    const modeDesc = document.getElementById("modeDescription");
    const temporalSettings = document.getElementById("temporalSettings");
    
    // Sliders
    const topKSlider = document.getElementById("topKSlider");
    const topKValue = document.getElementById("topKValue");
    topKSlider.addEventListener("input", () => { topKValue.innerText = topKSlider.value; });

    const maxKfGap = document.getElementById("maxKfGap");
    const maxKfGapVal = document.getElementById("maxKfGapVal");
    maxKfGap.addEventListener("input", () => { maxKfGapVal.innerText = maxKfGap.value; });

    const minKfGap = document.getElementById("minKfGap");
    const minKfGapVal = document.getElementById("minKfGapVal");
    minKfGap.addEventListener("input", () => { minKfGapVal.innerText = minKfGap.value; });

    const beamWidth = document.getElementById("beamWidth");
    const beamWidthVal = document.getElementById("beamWidthVal");
    beamWidth.addEventListener("input", () => { beamWidthVal.innerText = beamWidth.value; });

    // Search Type change
    searchType.addEventListener("change", () => {
        if (searchType.value === "temporal") {
            temporalSettings.style.display = "flex";
            modeDesc.innerHTML = `💡 <b>Chế độ Temporal đang bật:</b> Sử dụng dấu <code>/</code> để phân tách các bước theo thời gian. Ví dụ: <code>Cảnh mở cửa / Người đàn ông ngồi vào bàn</code>`;
            queryInput.placeholder = "Cảnh mở cửa / Người đàn ông ngồi vào bàn";
            if (!queryInput.value || queryInput.value === "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh") {
                queryInput.value = "Cảnh mở cửa / Người đàn ông ngồi vào bàn";
            }
        } else {
            temporalSettings.style.display = "none";
            modeDesc.innerText = "Nhập câu truy vấn của bạn bằng tiếng Việt hoặc tiếng Anh để tìm kiếm các khung hình (keyframe) chính xác nhất.";
            queryInput.placeholder = "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh";
            if (!queryInput.value || queryInput.value === "Cảnh mở cửa / Người đàn ông ngồi vào bàn") {
                queryInput.value = "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh";
            }
        }
    });

    // Default query
    queryInput.value = "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh";

    // Retrieval Mode radio change
    const radioModes = document.querySelectorAll('input[name="retrievalMode"]');
    const multiModelSettings = document.getElementById("multiModelSettings");
    const singleModelSettings = document.getElementById("singleModelSettings");

    radioModes.forEach(radio => {
        radio.addEventListener("change", () => {
            if (radio.value === "single") {
                multiModelSettings.style.display = "none";
                singleModelSettings.style.display = "flex";
            } else {
                multiModelSettings.style.display = "flex";
                singleModelSettings.style.display = "none";
            }
        });
    });
}

// ==========================================
// 2. CONVERTER
// ==========================================
function initConverter() {
    const btnConvert = document.getElementById("btnConvert");
    const converterInput = document.getElementById("converterInput");
    const converterResult = document.getElementById("converterResult");
    const convertedString = document.getElementById("convertedString");
    const convertedDetails = document.getElementById("convertedDetails");

    btnConvert.addEventListener("click", async () => {
        const val = converterInput.value.trim();
        if (!val) return;

        try {
            const resp = await fetch("/api/convert_time", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ converter_input: val })
            });

            const data = await resp.json();
            if (!resp.ok) {
                alert(data.detail || "Lỗi chuyển đổi thời gian");
                return;
            }

            convertedString.innerText = data.submission_string;
            convertedDetails.innerText = `FPS: ${data.fps} | Thời gian: ${data.total_seconds}s (${data.formatted_time})`;
            converterResult.style.display = "flex";
        } catch (err) {
            alert("Lỗi kết nối converter: " + err.message);
        }
    });
}

// ==========================================
// 3. SEARCH EXECUTION & RENDERING
// ==========================================
function initSearchEvents() {
    const btnSearch = document.getElementById("btnSearch");
    const queryInput = document.getElementById("queryInput");
    const btnClearQuery = document.getElementById("btnClearQuery");

    btnClearQuery.addEventListener("click", () => {
        queryInput.value = "";
        queryInput.focus();
    });

    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            performSearch();
        }
    });

    btnSearch.addEventListener("click", performSearch);
}

async function performSearch() {
    const queryInput = document.getElementById("queryInput");
    const query = queryInput.value.trim();
    if (!query) {
        alert("Vui lòng nhập nội dung truy vấn!");
        return;
    }

    const searchType = document.getElementById("searchType").value;
    const retrievalMode = document.querySelector('input[name="retrievalMode"]:checked').value;
    const singleModelChoice = document.getElementById("singleModelChoice").value;
    const useSiglip2 = document.getElementById("useSiglip2").checked;
    const useDfn5b = document.getElementById("useDfn5b").checked;
    const topK = parseInt(document.getElementById("topKSlider").value, 10);
    const maxKfGap = parseInt(document.getElementById("maxKfGap").value, 10);
    const minKfGap = parseInt(document.getElementById("minKfGap").value, 10);
    const beamWidth = parseInt(document.getElementById("beamWidth").value, 10);

    const payload = {
        query: query,
        search_type: searchType,
        retrieval_mode: retrievalMode,
        single_model_choice: singleModelChoice,
        use_siglip2: useSiglip2,
        use_dfn5b: useDfn5b,
        top_k: topK,
        max_kf_gap: maxKfGap,
        min_kf_gap: minKfGap,
        beam_width: beamWidth
    };

    const loading = document.getElementById("loadingIndicator");
    const resultsArea = document.getElementById("resultsArea");

    loading.style.display = "flex";
    resultsArea.innerHTML = "";

    try {
        const resp = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await resp.json();
        loading.style.display = "none";

        if (!resp.ok) {
            resultsArea.innerHTML = `
                <div class="result-card" style="border-color: var(--danger);">
                    <h3 style="color: var(--danger);"><i class="fa-solid fa-triangle-exclamation"></i> Lỗi tìm kiếm</h3>
                    <p style="margin-top: 8px;">${data.detail || "Đã xảy ra lỗi không xác định"}</p>
                </div>
            `;
            return;
        }

        renderSearchResults(data);

    } catch (err) {
        loading.style.display = "none";
        resultsArea.innerHTML = `
            <div class="result-card" style="border-color: var(--danger);">
                <h3 style="color: var(--danger);"><i class="fa-solid fa-triangle-exclamation"></i> Lỗi kết nối máy chủ</h3>
                <p style="margin-top: 8px;">${err.message}</p>
            </div>
        `;
    }
}

// Quản lý danh sách các Video ID bị vô hiệu hóa
window.dismissedVideos = new Set();
window.hideDismissedMode = false;

// Render search results list
function renderSearchResults(data) {
    const resultsArea = document.getElementById("resultsArea");
    const results = data.results || [];
    const total = data.total || 0;
    const isTemporal = data.search_type === "temporal";

    // Reset danh sách video bị loại khi thực hiện tìm kiếm mới
    window.dismissedVideos.clear();

    if (results.length === 0) {
        resultsArea.innerHTML = `
            <div class="result-card">
                <h3>🔍 Không tìm thấy kết quả</h3>
                <p style="color: var(--text-secondary); margin-top: 8px;">
                    Không tìm thấy keyframe nào phù hợp với câu truy vấn hoặc điều kiện lọc. Hãy thử câu truy vấn khác.
                </p>
            </div>
        `;
        return;
    }

    let html = `
        <div class="results-header">
            <div class="results-count">
                Tìm thấy <span>${total}</span> kết quả. Đang hiển thị Top <span>${results.length}</span>:
            </div>
            <div class="dismiss-controls">
                <label class="checkbox-label" style="font-size: 0.85rem; margin: 0; cursor: pointer;" title="Tự động ẩn hẳn các card thuộc video đã loại">
                    <input type="checkbox" id="hideDismissedToggle" onchange="toggleHideDismissed(this.checked)">
                    <span>Ẩn hẳn video đã loại</span>
                </label>
                <span id="dismissedSummaryBadge" class="badge-dismissed" style="display: none;">Đã loại: 0 video</span>
                <button id="btnRestoreAll" class="btn btn-sm btn-secondary" onclick="restoreAllVideos()" style="display: none;">
                    <i class="fa-solid fa-rotate-left"></i> Khôi phục tất cả
                </button>
            </div>
        </div>
    `;

    results.forEach((item, idx) => {
        if (isTemporal && item.sequence_path) {
            html += renderTemporalCard(item, idx);
        } else {
            html += renderStandardCard(item, idx);
        }
    });

    resultsArea.innerHTML = html;

    // Attach video player interaction listeners
    initVideoPlayerListeners();
}

// Render 1 Standard Keyframe Result Card
function renderStandardCard(item, idx) {
    const videoId = item.video_id;
    const rank = item.rank;
    const fps = item.fps || 25.0;
    const ptsTime = item.pts_time || 0.0;
    const kfIdx = item.keyframe_index;
    const defaultSub = item.submission_string;
    const playerId = `vid_${rank}_${videoId}_${kfIdx}`;

    const metadata = item.metadata || {};
    const title = metadata.title ? `<div class="meta-item" style="width: 100%;"><b>Tiêu đề:</b> <span>${metadata.title}</span></div>` : '';
    const objects = (item.object_entities && item.object_entities.length > 0)
        ? `<div class="meta-item" style="width: 100%;"><b>Detected Objects:</b> <span>${item.object_entities.slice(0, 10).join(', ')}</span></div>`
        : '';
    const sources = (item.sources && item.sources.length > 0)
        ? `<div class="meta-item"><b>Models:</b> <span>${item.sources.join(', ')}</span></div>`
        : '';

    return `
        <div class="result-card" id="card_${rank}" data-video-id="${videoId}">
            <div class="dismissed-overlay-bar">
                <span><i class="fa-solid fa-ban"></i> Đã vô hiệu hóa video <b>${videoId}</b> (toàn bộ candidates cùng video này)</span>
                <button class="btn-undo-dismiss" onclick="restoreVideo('${videoId}')">
                    <i class="fa-solid fa-rotate-left"></i> Khôi phục video này
                </button>
            </div>

            <div class="card-top">
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <span class="rank-badge"><i class="fa-solid fa-trophy"></i> Top Rank ${rank}</span>
                    <button class="btn-dismiss-video" onclick="dismissVideo('${videoId}')" title="Vô hiệu hóa toàn bộ candidates thuộc video ${videoId}">
                        <i class="fa-solid fa-ban"></i> Loại Video (${videoId})
                    </button>
                </div>
                <span class="submission-code">${defaultSub}</span>
            </div>

            <div class="card-grid">
                <!-- LEFT: Keyframe Image -->
                <div class="keyframe-preview-box">
                    <img src="${item.keyframe_url}" alt="${videoId} - Frame ${kfIdx}" class="keyframe-img" loading="lazy"
                         onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'200\\'><rect fill=\\'%231e293b\\' width=\\'100%25\\' height=\\'100%25\\'/><text fill=\\'%2394a3b8\\' x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\'>Không tải được ảnh</text></svg>'">
                    <div class="keyframe-caption">Rank ${rank} - ${videoId} / ${item.img_filename}</div>
                </div>

                <!-- RIGHT: Details & Scores -->
                <div class="card-details">
                    <div class="meta-row">
                        <div class="meta-item"><b>Video ID:</b> <span>${videoId}</span></div>
                        <div class="meta-item"><b>Keyframe Index:</b> <span>${kfIdx}</span></div>
                        <div class="meta-item"><b>PTS Time:</b> <span>${ptsTime.toFixed(2)}s (${item.formatted_time})</span></div>
                        <div class="meta-item"><b>FPS:</b> <span>${fps}</span></div>
                        ${sources}
                    </div>

                    ${title}
                    ${objects}

                    <!-- SCORES BREAKDOWN -->
                    <div class="scores-container">
                        <div class="score-pill">
                            <span class="score-label">Retrieval Score</span>
                            <span class="score-value">${item.retrieval_score.toFixed(4)}</span>
                        </div>
                        <div class="score-pill">
                            <span class="score-label">Object Score</span>
                            <span class="score-value">${item.object_score.toFixed(4)}</span>
                        </div>
                        <div class="score-pill">
                            <span class="score-label">Metadata Score</span>
                            <span class="score-value">${item.metadata_score.toFixed(4)}</span>
                        </div>
                        <div class="score-pill">
                            <span class="score-label">OCR Score</span>
                            <span class="score-value">${item.ocr_score.toFixed(4)}</span>
                        </div>
                        <div class="score-pill">
                            <span class="score-label">ASR Score</span>
                            <span class="score-value">${item.asr_score.toFixed(4)}</span>
                        </div>
                        <div class="score-pill">
                            <span class="score-label">RRF Final Score</span>
                            <span class="score-value highlight">${item.final_score.toFixed(6)}</span>
                        </div>
                    </div>

                    <!-- DEFAULT SUBMISSION ROW -->
                    <div class="submission-banner">
                        <div>
                            <div class="submission-label">🎯 Định dạng nộp bài mặc định (tại Keyframe PTS):</div>
                            <div class="submission-code">${defaultSub}</div>
                        </div>
                        <button class="btn btn-copy" onclick="copyText('${defaultSub}')">
                            <i class="fa-solid fa-copy"></i> Copy mã này
                        </button>
                    </div>

                    <!-- VIDEO EXPANDER & LIVE CAPTURE -->
                    ${renderVideoModule(item, playerId)}
                </div>
            </div>
        </div>
    `;
}

// Render 1 Temporal Sequence Result Card
function renderTemporalCard(item, idx) {
    const videoId = item.video_id;
    const rank = item.rank;
    const steps = item.sequence_path || [];

    let stepsHtml = '';
    steps.forEach((step, sIdx) => {
        const sPlayerId = `vid_temp_${rank}_${step.video_id}_${step.keyframe_index}`;
        stepsHtml += `
            <div class="step-card">
                <div class="step-badge"><i class="fa-solid fa-arrow-right"></i> Bước ${step.step_idx}</div>
                <img src="${step.keyframe_url}" alt="${step.video_id}" class="keyframe-img" loading="lazy" style="height: 140px;">
                <div style="font-size: 0.82rem; display: flex; flex-direction: column; gap: 4px;">
                    <div><b>Video:</b> <code>${step.video_id}</code></div>
                    <div><b>KF Index:</b> <code>${step.keyframe_index}</code></div>
                    <div><b>Step Score:</b> <code>${step.score.toFixed(4)}</code></div>
                    <div><b>PTS:</b> <code>${step.pts_time.toFixed(2)}s (${step.formatted_time})</code></div>
                </div>
                <div class="submission-banner" style="padding: 8px 12px; margin-top: auto;">
                    <div class="submission-code" style="font-size: 0.95rem;">${step.submission_string}</div>
                    <button class="btn btn-sm btn-copy" onclick="copyText('${step.submission_string}')">
                        <i class="fa-solid fa-copy"></i>
                    </button>
                </div>
                ${renderVideoModule(step, sPlayerId)}
            </div>
        `;
    });

    return `
        <div class="result-card" id="card_temp_${rank}" data-video-id="${videoId}">
            <div class="dismissed-overlay-bar">
                <span><i class="fa-solid fa-ban"></i> Đã vô hiệu hóa chuỗi video <b>${videoId}</b> (toàn bộ candidates cùng video này)</span>
                <button class="btn-undo-dismiss" onclick="restoreVideo('${videoId}')">
                    <i class="fa-solid fa-rotate-left"></i> Khôi phục video này
                </button>
            </div>

            <div class="card-top">
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <span class="rank-badge"><i class="fa-solid fa-film"></i> Rank ${rank} (Chuỗi ${steps.length} bước)</span>
                    <button class="btn-dismiss-video" onclick="dismissVideo('${videoId}')" title="Vô hiệu hóa toàn bộ candidates thuộc video ${videoId}">
                        <i class="fa-solid fa-ban"></i> Loại Video (${videoId})
                    </button>
                </div>
                <div class="meta-item"><b>Final Score TB:</b> <span style="color: var(--warning);">${item.final_score.toFixed(6)}</span></div>
            </div>
            <div style="margin-bottom: 8px; font-size: 0.95rem;"><b>Video ID chung:</b> <code>${videoId}</code></div>
            <div class="temporal-sequence-row">
                ${stepsHtml}
            </div>
        </div>
    `;
}

// Render Video Module Component
function renderVideoModule(item, playerId) {
    if (!item.has_video) {
        return `
            <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 8px;">
                <i class="fa-solid fa-circle-exclamation"></i> Không tìm thấy file video gốc <code>${item.video_id}</code> trong Public Documents.
            </div>
        `;
    }

    const fps = item.fps || 25.0;
    const initialPts = item.pts_time || 0.0;

    return `
        <div class="video-module">
            <div class="video-module-header" onclick="toggleVideoAccordion('${playerId}')">
                <span class="video-module-title">
                    <i class="fa-solid fa-circle-play"></i> 🎬 Mở Trình phát Video & Bắt Live PTS_TIME (${item.video_id})
                </span>
                <i class="fa-solid fa-chevron-down" id="chevron_${playerId}"></i>
            </div>

            <div class="video-module-body" id="body_${playerId}" style="display: none;">
                <!-- HTML5 Video Player -->
                <video id="${playerId}" class="video-player-element" controls preload="metadata"
                       data-video-id="${item.video_id}" data-fps="${fps}" data-initial-pts="${initialPts}">
                    <source src="/api/video/${item.video_id}" type="video/mp4">
                    Trình duyệt không hỗ trợ thẻ video.
                </video>

                <!-- LIVE CAPTURE & SUBMISSION PANEL -->
                <div class="live-capture-panel">
                    <div class="live-metrics">
                        <div class="live-metric-item">
                            <span class="live-metric-label">⏱️ PTS Time hiện tại</span>
                            <span id="live_time_${playerId}" class="live-metric-val">${initialPts.toFixed(2)}s (${formatHms(initialPts)})</span>
                        </div>

                        <div class="live-metric-item">
                            <span class="live-metric-label">🎞️ Frame Index</span>
                            <span id="live_frame_${playerId}" class="live-metric-val">${item.calc_frame} (FPS: ${fps})</span>
                        </div>

                        <div class="live-metric-item">
                            <span class="live-metric-label">🎯 Mã nộp bài (Live Submission)</span>
                            <span id="live_sub_${playerId}" class="live-metric-val green">${item.submission_string}</span>
                        </div>

                        <button id="btn_copy_live_${playerId}" class="btn btn-primary" style="margin-left: auto;"
                                onclick="copyText(document.getElementById('live_sub_${playerId}').innerText)">
                            <i class="fa-solid fa-copy"></i> Sao chép mã này
                        </button>
                    </div>

                    <!-- STEPPING CONTROLS -->
                    <div class="video-stepping-bar">
                        <button class="btn-step" onclick="seekVideo('${playerId}', -5.0)" title="Lùi 5 giây">⏪ -5s</button>
                        <button class="btn-step" onclick="seekVideo('${playerId}', -1.0)" title="Lùi 1 giây">◀ -1s</button>
                        <button class="btn-step" onclick="stepFrame('${playerId}', -1)" title="Lùi 1 frame (1/FPS)">⏮ -1 Frame</button>
                        <button class="btn-step" onclick="togglePlayPause('${playerId}')" style="background: #2563eb;" title="Phát / Tạm dừng">⏯ Pause / Play</button>
                        <button class="btn-step" onclick="stepFrame('${playerId}', 1)" title="Tiến 1 frame (1/FPS)">⏭ +1 Frame</button>
                        <button class="btn-step" onclick="seekVideo('${playerId}', 1.0)" title="Tiến 1 giây">▶ +1s</button>
                        <button class="btn-step" onclick="seekVideo('${playerId}', 5.0)" title="Tiến 5 giây">⏩ +5s</button>
                        <button class="btn-step" onclick="jumpToInitialPts('${playerId}')" style="background: #475569;" title="Về mốc Keyframe PTS">🎯 Về mốc KF</button>
                    </div>
                </div>

                <div style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">
                    💡 <b>Mẹo thi đấu:</b> Bấm phát video, sau đó bấm <b>Pause</b> hoặc dùng nút <b>⏮ -1 Frame / +1 Frame ⏭</b> để căn chỉnh chính xác khung hình. Hệ thống sẽ tự động bắt <code>current_time</code> và sinh chuỗi <b><code>video_id / frame_idx</code></b> để bạn copy nộp bài ngay lập tức!
                </div>
            </div>
        </div>
    `;
}

// Toggle accordion
window.toggleVideoAccordion = function(playerId) {
    const body = document.getElementById(`body_${playerId}`);
    const chevron = document.getElementById(`chevron_${playerId}`);
    const vid = document.getElementById(playerId);

    if (body.style.display === "none" || !body.style.display) {
        body.style.display = "flex";
        chevron.className = "fa-solid fa-chevron-up";
        
        // Tua tới initial PTS khi mở nếu chưa tua
        if (vid && vid.dataset.initialPts) {
            const initialPts = parseFloat(vid.dataset.initialPts);
            if (vid.currentTime === 0 && initialPts > 0) {
                vid.currentTime = initialPts;
            }
        }
    } else {
        body.style.display = "none";
        chevron.className = "fa-solid fa-chevron-down";
        if (vid && !vid.paused) {
            vid.pause();
        }
    }
};

// Seek by delta seconds
window.seekVideo = function(playerId, deltaSeconds) {
    const vid = document.getElementById(playerId);
    if (!vid) return;
    vid.currentTime = Math.max(0, vid.currentTime + deltaSeconds);
};

// Step by frame
window.stepFrame = function(playerId, frameDelta) {
    const vid = document.getElementById(playerId);
    if (!vid) return;
    const fps = parseFloat(vid.dataset.fps) || 25.0;
    const delta = frameDelta * (1.0 / fps);
    vid.currentTime = Math.max(0, vid.currentTime + delta);
};

// Toggle play/pause
window.togglePlayPause = function(playerId) {
    const vid = document.getElementById(playerId);
    if (!vid) return;
    if (vid.paused) {
        vid.play();
    } else {
        vid.pause();
    }
};

// Jump to initial PTS
window.jumpToInitialPts = function(playerId) {
    const vid = document.getElementById(playerId);
    if (!vid) return;
    const initialPts = parseFloat(vid.dataset.initialPts) || 0.0;
    vid.currentTime = initialPts;
};

// Initialize listeners on all video tags
function initVideoPlayerListeners() {
    const videos = document.querySelectorAll("video.video-player-element");

    videos.forEach(vid => {
        const playerId = vid.id;
        const videoId = vid.dataset.videoId;
        const fps = parseFloat(vid.dataset.fps) || 25.0;

        const updateLiveInfo = () => {
            const currentTime = vid.currentTime;
            const frameIdx = Math.round(currentTime * fps);
            const subString = `${videoId} / ${frameIdx}`;

            const timeEl = document.getElementById(`live_time_${playerId}`);
            const frameEl = document.getElementById(`live_frame_${playerId}`);
            const subEl = document.getElementById(`live_sub_${playerId}`);

            if (timeEl) timeEl.innerText = `${currentTime.toFixed(2)}s (${formatHms(currentTime)})`;
            if (frameEl) frameEl.innerText = `${frameIdx} (FPS: ${fps})`;
            if (subEl) subEl.innerText = subString;
        };

        // Bắt sự kiện thời gian thực khi chạy, dừng hoặc tua
        vid.addEventListener("timeupdate", updateLiveInfo);
        vid.addEventListener("pause", updateLiveInfo);
        vid.addEventListener("seeked", updateLiveInfo);
        vid.addEventListener("play", updateLiveInfo);
    });
}

// ==========================================
// TÍNH NĂNG: VÔ HIỆU HÓA TOÀN BỘ CANDIDATES THEO VIDEO ID
// ==========================================

// Cập nhật thống kê và trạng thái nút điều khiển trên header
function updateDismissStats() {
    const badge = document.getElementById("dismissedSummaryBadge");
    const btnRestoreAll = document.getElementById("btnRestoreAll");
    const dismissedCards = document.querySelectorAll(".result-card.card-dismissed");

    const dismissedCount = window.dismissedVideos.size;
    const cardCount = dismissedCards.length;

    if (badge) {
        if (dismissedCount > 0) {
            badge.style.display = "inline-block";
            badge.innerHTML = `<i class="fa-solid fa-ban"></i> Đã loại: <b>${dismissedCount}</b> video (${cardCount} thẻ)`;
        } else {
            badge.style.display = "none";
        }
    }

    if (btnRestoreAll) {
        btnRestoreAll.style.display = dismissedCount > 0 ? "inline-flex" : "none";
    }
}

// Vô hiệu hóa một video (toàn bộ thẻ có cùng data-video-id)
window.dismissVideo = function(videoId) {
    if (!videoId) return;

    window.dismissedVideos.add(videoId);

    // Tìm tất cả các thẻ có videoId này
    const cards = document.querySelectorAll(`.result-card[data-video-id="${videoId}"]`);
    cards.forEach(card => {
        card.classList.add("card-dismissed");
        if (window.hideDismissedMode) {
            card.classList.add("dismissed-hidden");
        }
        // Dừng video player nếu đang phát
        const player = card.querySelector("video");
        if (player && !player.paused) {
            player.pause();
        }
    });

    updateDismissStats();
    showToast(`Đã loại toàn bộ candidates của video ${videoId} (${cards.length} thẻ)`, "info");
};

// Khôi phục một video cụ thể
window.restoreVideo = function(videoId) {
    if (!videoId) return;

    window.dismissedVideos.delete(videoId);

    const cards = document.querySelectorAll(`.result-card[data-video-id="${videoId}"]`);
    cards.forEach(card => {
        card.classList.remove("card-dismissed");
        card.classList.remove("dismissed-hidden");
    });

    updateDismissStats();
    showToast(`Đã khôi phục video ${videoId}`, "success");
};

// Khôi phục tất cả các video đã bị loại
window.restoreAllVideos = function() {
    if (window.dismissedVideos.size === 0) return;

    window.dismissedVideos.clear();

    const allDismissedCards = document.querySelectorAll(".result-card.card-dismissed");
    allDismissedCards.forEach(card => {
        card.classList.remove("card-dismissed");
        card.classList.remove("dismissed-hidden");
    });

    updateDismissStats();
    showToast("Đã khôi phục lại toàn bộ video đã loại", "success");
};

// Bật/tắt chế độ ẩn hẳn các card đã bị loại
window.toggleHideDismissed = function(isChecked) {
    window.hideDismissedMode = isChecked;

    const allDismissedCards = document.querySelectorAll(".result-card.card-dismissed");
    allDismissedCards.forEach(card => {
        if (isChecked) {
            card.classList.add("dismissed-hidden");
        } else {
            card.classList.remove("dismissed-hidden");
        }
    });
};



