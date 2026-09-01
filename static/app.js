// ============================================
// KINETIC OVERSIGHT — App Controller
// Municipal ANPR Intelligence Dashboard
// ============================================

let currentTab = 'dashboard';
let jobPollInterval = null;
let currentJobId = null;
let dashboardRefreshInterval = null;
let mapZoomLevel = 1;

// === INITIALIZATION ===
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupFileUpload();
    setupSliders();
    setupMapControls();
    switchTab('dashboard');
});

// === NAVIGATION ===
function setupNavigation() {
    // Get all nav links with data-tab attribute
    const navLinks = document.querySelectorAll('[data-tab]');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = link.getAttribute('data-tab');
            switchTab(tab);
        });
    });
    
    // Alert banner "Go to Alert Panel" button
    const alertBannerBtn = document.getElementById('alertBannerBtn');
    if (alertBannerBtn) {
        alertBannerBtn.addEventListener('click', () => switchTab('alerts'));
    }
    
    // Header Blacklist button
    const headerBlacklistBtn = document.getElementById('headerBlacklistBtn');
    if (headerBlacklistBtn) {
        headerBlacklistBtn.addEventListener('click', () => switchTab('alerts'));
    }
    
    // Global search in top header
    const globalSearch = document.getElementById('globalSearch');
    if (globalSearch) {
        globalSearch.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const plate = globalSearch.value.trim().toUpperCase();
                if (plate) {
                    document.getElementById('plateSearchInput').value = plate;
                    switchTab('lookup');
                    lookupPlate();
                }
            }
        });
    }
}

function switchTab(tabName) {
    currentTab = tabName;
    
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show target tab
    const targetTab = document.getElementById('tab-' + tabName);
    if (targetTab) targetTab.classList.add('active');
    
    // Update nav styles - remove active from all
    const navLinks = document.querySelectorAll('[data-tab]');
    navLinks.forEach(link => {
        // Reset to inactive style
        link.className = link.className
            .replace(/bg-secondary/g, '')
            .replace(/text-on-secondary/g, '')
            .replace(/font-bold/g, '')
            .replace(/shadow-\[2px_2px_0px_0px_rgba\(0,0,0,1\)\]/g, '');
        link.classList.add('text-on-surface-variant');
        link.classList.add('hover:bg-surface-container-high');
        link.classList.add('border-transparent');
        // Reset icon fill
        const icon = link.querySelector('.material-symbols-outlined');
        if (icon) icon.style.fontVariationSettings = '';
    });
    
    // Set active style on current tab
    const activeLink = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeLink) {
        activeLink.classList.remove('text-on-surface-variant', 'hover:bg-surface-container-high', 'border-transparent');
        activeLink.classList.add('bg-secondary', 'text-on-secondary', 'font-bold');
        activeLink.style.boxShadow = '2px 2px 0px 0px rgba(0,0,0,1)';
        activeLink.style.borderColor = '#000000';
        const icon = activeLink.querySelector('.material-symbols-outlined');
        if (icon) icon.style.fontVariationSettings = "'FILL' 1";
    }
    
    // Load tab data
    if (tabName === 'dashboard') loadDashboard();
    if (tabName === 'processing') loadCameras();
    if (tabName === 'routes') loadRoutes();
    if (tabName === 'alerts') loadAlerts();
    if (tabName === 'heatmap') loadHeatmap();
    
    // Dashboard auto-refresh
    if (dashboardRefreshInterval) clearInterval(dashboardRefreshInterval);
    if (tabName === 'dashboard') {
        dashboardRefreshInterval = setInterval(loadDashboard, 10000);
    }
}

// === DASHBOARD TAB ===
async function loadDashboard() {
    try {
        // Load stats
        const stats = await fetch('/api/stats').then(r => r.json());
        
        document.getElementById('statTotalDetections').textContent = Number(stats.total_detections).toLocaleString();
        document.getElementById('statUniqueVehicles').textContent = Number(stats.unique_plates).toLocaleString();
        document.getElementById('statActiveCameras').textContent = stats.unique_cameras;
        
        // Alert banner
        const banner = document.getElementById('alertBanner');
        if (banner) {
            if (stats.unacknowledged_alerts > 0) {
                banner.classList.remove('hidden');
                document.getElementById('alertBannerText').textContent = 
                    `🚨 ${stats.unacknowledged_alerts} UNACKNOWLEDGED ALERT(S) — Blacklisted vehicle(s) detected!`;
            } else {
                banner.classList.add('hidden');
            }
        }
        
        // Load charts
        loadTimelineChart();
        loadCameraChart();
        loadRecentActivity();
    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

async function loadTimelineChart() {
    try {
        const data = await fetch('/api/detections-timeline').then(r => r.json());
        const container = document.getElementById('chartTimeline');
        if (!container || !data.length) return;
        
        const maxCount = Math.max(...data.map(d => d.count));
        
        let html = '<div class="flex flex-col gap-1 h-full justify-end">';
        // Show last 24 bars
        const displayData = data.slice(-24);
        html += '<div class="flex items-end gap-[2px] h-full min-h-[250px]">';
        displayData.forEach(d => {
            const pct = maxCount > 0 ? (d.count / maxCount * 100) : 0;
            const time = d.bucket_time ? d.bucket_time.split(' ')[1]?.substring(0,5) : '';
            html += `<div class="flex-1 flex flex-col items-center justify-end h-full">`;
            html += `<div class="w-full bg-primary border border-primary transition-all hover:bg-secondary" style="height: ${pct}%" title="${d.count} detections at ${time}"></div>`;
            html += `</div>`;
        });
        html += '</div>';
        // Time labels
        html += '<div class="flex gap-[2px] mt-2">';
        displayData.forEach((d, i) => {
            if (i % 4 === 0) {
                const time = d.bucket_time ? d.bucket_time.split(' ')[1]?.substring(0,5) : '';
                html += `<div class="flex-1 text-center font-label-mono text-[10px] text-on-surface-variant">${time}</div>`;
            } else {
                html += `<div class="flex-1"></div>`;
            }
        });
        html += '</div></div>';
        
        container.innerHTML = html;
    } catch (err) {
        console.error('Timeline chart error:', err);
    }
}

async function loadCameraChart() {
    try {
        const data = await fetch('/api/camera-heatmap').then(r => r.json());
        const container = document.getElementById('chartCameras');
        if (!container || !data.length) return;
        
        // Sort by count descending, take top 6
        const sorted = data.sort((a, b) => b.count - a.count).slice(0, 6);
        const maxCount = sorted[0]?.count || 1;
        
        let html = '';
        sorted.forEach((cam, i) => {
            const pct = (cam.count / maxCount * 100).toFixed(0);
            const countStr = cam.count >= 1000 ? (cam.count / 1000).toFixed(1) + 'K' : cam.count;
            const isHot = i === 0;
            const barColor = isHot ? 'bg-secondary' : 'bg-primary';
            const textColor = isHot ? 'text-secondary' : 'text-on-surface-variant';
            html += `<div class="flex flex-col gap-1">`;
            html += `<div class="flex justify-between font-label-mono text-label-mono ${textColor}"><span>${cam.name}${isHot ? ' (HOT)' : ''}</span><span>${countStr}</span></div>`;
            html += `<div class="h-4 w-full bg-surface-variant border border-primary"><div class="h-full ${barColor} border-r border-primary" style="width: ${pct}%"></div></div>`;
            html += `</div>`;
        });
        
        container.innerHTML = html;
    } catch (err) {
        console.error('Camera chart error:', err);
    }
}

async function loadRecentActivity() {
    try {
        const recent = await fetch('/api/recent-activity').then(r => r.json());
        const container = document.getElementById('recentActivityBody');
        if (!container) return;
        
        if (!recent || !recent.length) {
            container.innerHTML = `<tr><td colspan="4" class="p-3 text-center text-on-surface-variant">No recent activity recorded yet</td></tr>`;
            return;
        }

        container.innerHTML = '';
        recent.forEach(item => {
            const timeStr = item.timestamp ? item.timestamp.replace('T', ' ') : 'Just now';
            const ocrConfStr = item.ocr_conf ? Math.round(item.ocr_conf * 100) + '%' : 'N/A';
            const detConfStr = item.detection_conf ? Math.round(item.detection_conf * 100) + '%' : 'N/A';
            const isHigh = item.ocr_conf && item.ocr_conf > 0.85;

            container.innerHTML += `
                <tr class="border-b border-primary/20 hover:bg-surface-container-high transition-colors">
                    <td class="p-3 border-r border-primary/20 font-mono text-[11px]">${timeStr}</td>
                    <td class="p-3 border-r border-primary/20 font-bold text-primary tracking-wider">${item.plate_text}</td>
                    <td class="p-3 border-r border-primary/20">${item.camera_name || item.camera_id}</td>
                    <td class="p-3 ${isHigh ? 'text-secondary font-bold' : ''}">OCR: ${ocrConfStr} · Det: ${detConfStr}</td>
                </tr>`;
        });
    } catch (err) {
        console.error('Recent activity load error:', err);
    }
}

// === VIDEO PROCESSING TAB ===
async function loadCameras() {
    try {
        const cameras = await fetch('/api/cameras').then(r => r.json());
        const select = document.getElementById('cameraSelect');
        if (!select) return;
        
        select.innerHTML = '<option value="">-- Select Existing Source --</option>';
        cameras.forEach(cam => {
            select.innerHTML += `<option value="${cam.camera_id}">${cam.name}</option>`;
        });
    } catch (err) {
        console.error('Load cameras error:', err);
    }
}

function setupFileUpload() {
    const dropzone = document.getElementById('fileDropzone');
    const fileInput = document.getElementById('videoFileInput');
    const browseBtn = document.getElementById('browseFilesBtn');
    const fileNameDisplay = document.getElementById('selectedFileName');
    
    if (!dropzone || !fileInput) return;
    
    // Click to browse
    if (browseBtn) {
        browseBtn.addEventListener('click', () => fileInput.click());
    }
    dropzone.addEventListener('click', () => fileInput.click());
    
    // Drag & drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('bg-surface-container');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('bg-surface-container');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('bg-surface-container');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            if (fileNameDisplay) fileNameDisplay.textContent = e.dataTransfer.files[0].name;
        }
    });
    
    // File selected
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0 && fileNameDisplay) {
            fileNameDisplay.textContent = fileInput.files[0].name;
        }
    });
    
    // Enqueue job button
    const enqueueBtn = document.getElementById('enqueueJobBtn');
    if (enqueueBtn) {
        enqueueBtn.addEventListener('click', submitVideo);
    }
}

function setupSliders() {
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');
    if (confSlider && confValue) {
        confSlider.addEventListener('input', () => {
            confValue.textContent = confSlider.value + '%';
        });
    }
    
    const ocrSlider = document.getElementById('ocrSlider');
    const ocrValue = document.getElementById('ocrValue');
    if (ocrSlider && ocrValue) {
        ocrSlider.addEventListener('input', () => {
            ocrValue.textContent = 'Every ' + ocrSlider.value + ' Frames';
        });
    }
}

async function submitVideo() {
    const fileInput = document.getElementById('videoFileInput');
    const cameraSelect = document.getElementById('cameraSelect');
    const newCameraName = document.getElementById('newCameraName');
    const newCameraLat = document.getElementById('newCameraLat');
    const newCameraLon = document.getElementById('newCameraLon');
    const confSlider = document.getElementById('confSlider');
    const ocrSlider = document.getElementById('ocrSlider');
    
    if (!fileInput || !fileInput.files.length) {
        alert('Please select a video file.');
        return;
    }
    
    let cameraId = cameraSelect ? cameraSelect.value : '';
    
    // Create new camera if needed
    if (!cameraId && newCameraName && newCameraName.value.trim()) {
        try {
            const camRes = await fetch('/api/cameras', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: newCameraName.value.trim(),
                    lat: parseFloat(newCameraLat?.value) || 12.97,
                    lon: parseFloat(newCameraLon?.value) || 77.59,
                    description: 'Created via dashboard'
                })
            });
            const camData = await camRes.json();
            cameraId = camData.camera_id;
        } catch (err) {
            alert('Failed to create camera: ' + err.message);
            return;
        }
    }
    
    if (!cameraId) {
        alert('Please select or create a camera.');
        return;
    }
    
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('camera_id', cameraId);
    formData.append('conf', (confSlider ? confSlider.value / 100 : 0.35).toString());
    formData.append('ocr_n', (ocrSlider ? ocrSlider.value : '3'));
    formData.append('max_frames', '0');
    
    try {
        const res = await fetch('/api/upload-video', { method: 'POST', body: formData });
        const data = await res.json();
        currentJobId = data.job_id;
        
        // Add job to tracker table
        addJobToTracker(currentJobId, fileInput.files[0].name, 'PENDING', 0);
        
        // Start polling
        startJobPolling(currentJobId);
    } catch (err) {
        alert('Upload failed: ' + err.message);
    }
}

function addJobToTracker(jobId, source, status, progress) {
    const tbody = document.getElementById('jobTrackerBody');
    if (!tbody) return;
    
    const shortId = '#JOB-' + jobId.substring(0, 4).toUpperCase();
    const row = document.createElement('tr');
    row.id = 'job-' + jobId;
    row.className = 'border-b border-primary/20 hover:bg-surface-container-low transition-colors';
    row.innerHTML = `
        <td class="p-3 border-r border-primary/20 font-bold">${shortId}</td>
        <td class="p-3 border-r border-primary/20">${source}</td>
        <td class="p-3 border-r border-primary/20" id="job-status-${jobId}">
            <span class="inline-flex items-center gap-1 bg-surface-variant text-on-surface-variant px-2 py-1 border border-primary text-[10px]">
                <span class="material-symbols-outlined text-[12px]">hourglass_empty</span> PENDING
            </span>
        </td>
        <td class="p-3 border-r border-primary/20" id="job-progress-${jobId}">
            <div class="flex items-center gap-2">
                <div class="flex-grow h-3 border-2 border-primary bg-surface w-full overflow-hidden">
                    <div class="h-full bg-primary" style="width: 0%" id="job-bar-${jobId}"></div>
                </div>
                <span class="font-bold" id="job-pct-${jobId}">0%</span>
            </div>
        </td>
        <td class="p-3 text-center">
            <button class="p-1 border-2 border-primary/50 text-primary/50 cursor-not-allowed opacity-50" disabled>
                <span class="material-symbols-outlined text-[16px]">download</span>
            </button>
        </td>
    `;
    
    // Insert at top
    tbody.insertBefore(row, tbody.firstChild);
}

function startJobPolling(jobId) {
    if (jobPollInterval) clearInterval(jobPollInterval);
    jobPollInterval = setInterval(() => trackJob(jobId), 2000);
}

async function trackJob(jobId) {
    try {
        const data = await fetch(`/api/job/${jobId}`).then(r => r.json());
        
        const statusEl = document.getElementById(`job-status-${jobId}`);
        const barEl = document.getElementById(`job-bar-${jobId}`);
        const pctEl = document.getElementById(`job-pct-${jobId}`);
        
        if (barEl) barEl.style.width = data.progress + '%';
        if (pctEl) pctEl.textContent = data.progress + '%';
        
        if (statusEl) {
            if (data.status === 'processing') {
                statusEl.innerHTML = `<span class="inline-flex items-center gap-1 bg-primary text-on-primary px-2 py-1 border border-primary text-[10px]"><span class="material-symbols-outlined text-[12px] animate-spin">sync</span> PROCESSING</span>`;
            } else if (data.status === 'completed') {
                statusEl.innerHTML = `<span class="inline-flex items-center gap-1 bg-secondary text-on-secondary px-2 py-1 border border-secondary text-[10px] font-bold"><span class="material-symbols-outlined text-[12px]">check_circle</span> COMPLETED</span>`;
                if (barEl) { barEl.style.width = '100%'; barEl.classList.remove('bg-primary'); barEl.classList.add('bg-secondary'); }
                if (pctEl) pctEl.textContent = '100%';
                
                // Add detection count
                const progressEl = document.getElementById(`job-progress-${jobId}`);
                if (progressEl && data.detections_found) {
                    progressEl.innerHTML += `<div class="text-[10px] mt-1 opacity-70">${data.detections_found} plates detected</div>`;
                }

                // Update download action button
                const row = document.getElementById(`job-${jobId}`);
                if (row) {
                    const actionCell = row.cells[row.cells.length - 1];
                    if (actionCell) {
                        actionCell.innerHTML = `
                            <a href="/api/job/${jobId}/download" class="p-1 border-2 border-primary bg-primary text-on-primary hover:bg-secondary hover:text-on-secondary font-bold text-[10px] px-2 py-1 uppercase inline-flex items-center gap-1 transition-colors" title="Download tracked video" download>
                                <span class="material-symbols-outlined text-[14px]">download</span> Save
                            </a>`;
                    }
                }
                
                clearInterval(jobPollInterval);
            } else if (data.status === 'failed') {
                statusEl.innerHTML = `<span class="inline-flex items-center gap-1 bg-error text-on-error px-2 py-1 border border-error text-[10px] font-bold"><span class="material-symbols-outlined text-[12px]">error</span> FAILED</span>`;
                clearInterval(jobPollInterval);
            }
        }
    } catch (err) {
        console.error('Job tracking error:', err);
    }
}

// === VEHICLE LOOKUP TAB ===
async function lookupPlate() {
    const input = document.getElementById('plateSearchInput');
    if (!input || !input.value.trim()) return;
    
    const plate = input.value.trim().toUpperCase();
    
    try {
        const data = await fetch(`/api/plate/${plate}`).then(r => r.json());
        
        // Status card
        const statusCard = document.getElementById('vehicleStatusCard');
        if (statusCard) {
            if (data.blacklist_reason) {
                statusCard.innerHTML = `
                    <div class="p-stack-md border-b-2 border-primary bg-error-container">
                        <div class="flex items-center gap-3">
                            <span class="material-symbols-outlined text-secondary text-display-lg" style="font-variation-settings: 'FILL' 1;">warning</span>
                            <div>
                                <p class="font-label-mono text-label-mono text-secondary uppercase">Alert Status</p>
                                <h2 class="font-headline-md text-headline-md font-extrabold text-secondary uppercase">BLACKLISTED: ${data.blacklist_reason}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="p-stack-md grid grid-cols-2 gap-4">
                        <div class="border-heavy p-3 bg-surface-container">
                            <p class="font-label-mono text-label-mono text-on-surface-variant uppercase mb-1">Sightings</p>
                            <p class="font-metric-xl text-metric-xl text-primary">${data.sightings}</p>
                        </div>
                        <div class="border-heavy p-3 bg-surface-container">
                            <p class="font-label-mono text-label-mono text-on-surface-variant uppercase mb-1">Cameras</p>
                            <p class="font-metric-xl text-metric-xl text-primary">${data.cameras}</p>
                        </div>
                    </div>`;
            } else {
                statusCard.innerHTML = `
                    <div class="p-stack-md border-b-2 border-primary bg-surface-container-high">
                        <div class="flex items-center gap-3">
                            <span class="material-symbols-outlined text-primary text-display-lg" style="font-variation-settings: 'FILL' 1;">verified</span>
                            <div>
                                <p class="font-label-mono text-label-mono text-on-surface-variant uppercase">Alert Status</p>
                                <h2 class="font-headline-md text-headline-md font-extrabold text-primary uppercase">CLEAR — No Alerts</h2>
                            </div>
                        </div>
                    </div>
                    <div class="p-stack-md grid grid-cols-2 gap-4">
                        <div class="border-heavy p-3 bg-surface-container">
                            <p class="font-label-mono text-label-mono text-on-surface-variant uppercase mb-1">Sightings</p>
                            <p class="font-metric-xl text-metric-xl text-primary">${data.sightings}</p>
                        </div>
                        <div class="border-heavy p-3 bg-surface-container">
                            <p class="font-label-mono text-label-mono text-on-surface-variant uppercase mb-1">Cameras</p>
                            <p class="font-metric-xl text-metric-xl text-primary">${data.cameras}</p>
                        </div>
                    </div>`;
            }
        }
        
        // Detection log table
        const logBody = document.getElementById('detectionLogBody');
        if (logBody && data.history) {
            logBody.innerHTML = '';
            data.history.forEach(h => {
                const time = h.timestamp ? h.timestamp.split(' ')[1]?.substring(0, 8) || h.timestamp : '';
                const conf = h.ocr_conf ? Math.round(h.ocr_conf * 100) + '%' : 'N/A';
                const isHigh = h.ocr_conf && h.ocr_conf > 0.95;
                logBody.innerHTML += `
                    <tr class="border-b border-outline-variant hover:bg-surface-container-high transition-colors">
                        <td class="p-3 border-r border-outline-variant">${time}</td>
                        <td class="p-3 border-r border-outline-variant">${h.camera_name || h.camera_id}</td>
                        <td class="p-3 ${isHigh ? 'text-secondary font-bold' : ''}">${conf}</td>
                    </tr>`;
            });
        }
        
        // Trajectory map
        const trajLabel = document.getElementById('trajectoryLabel');
        if (trajLabel) trajLabel.textContent = 'Live Trajectory: ' + plate;
        
        const mapContainer = document.getElementById('mapOuterContainer');
        if (mapContainer && data.history && data.history.length > 0) {
            mapContainer.innerHTML = `
                <div class="relative w-full h-full min-h-[600px]">
                    <div class="absolute top-4 left-4 z-30 bg-surface border-heavy px-4 py-2 shadow-hard">
                        <span class="font-label-mono text-label-mono font-bold uppercase text-primary">Live Trajectory: ${plate}</span>
                    </div>
                    <iframe src="/api/map/trajectory/${encodeURIComponent(plate)}" class="w-full h-full min-h-[600px] border-none" title="Vehicle Trajectory Map"></iframe>
                </div>`;
        } else {
            buildTrajectoryMap(data.history);
        }
        
    } catch (err) {
        console.error('Lookup error:', err);
    }
}

function buildTrajectoryMap(history) {
    const mapContainer = document.getElementById('trajectoryMapNodes');
    const svgPath = document.getElementById('trajectorySvgPath');
    const svgPathAnimated = document.getElementById('trajectorySvgPathAnimated');
    if (!mapContainer || !history || !history.length) return;
    
    mapContainer.innerHTML = '';
    
    // Map coordinates to pixel positions within 800x600 viewbox
    const lats = history.map(h => h.latitude).filter(Boolean);
    const lons = history.map(h => h.longitude).filter(Boolean);
    if (!lats.length) return;
    
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const minLon = Math.min(...lons), maxLon = Math.max(...lons);
    const padding = 100;
    
    const points = history.filter(h => h.latitude && h.longitude).map((h, i) => {
        const x = lons.length > 1 ? padding + ((h.longitude - minLon) / (maxLon - minLon || 1)) * (800 - 2 * padding) : 400;
        const y = lats.length > 1 ? (800 - 2 * padding) - ((h.latitude - minLat) / (maxLat - minLat || 1)) * (600 - 2 * padding) + padding : 300;
        return { x, y, ...h, index: i };
    });
    
    if (!points.length) return;
    
    // Build SVG path
    const pathD = points.map((p, i) => (i === 0 ? 'M' : 'L') + ` ${p.x} ${p.y}`).join(' ');
    if (svgPath) svgPath.setAttribute('d', pathD);
    if (svgPathAnimated) {
        svgPathAnimated.setAttribute('d', pathD);
        // Restart animation
        svgPathAnimated.setAttribute('stroke-dasharray', '1000');
        svgPathAnimated.setAttribute('stroke-dashoffset', '1000');
        const anim = svgPathAnimated.querySelector('animate');
        if (anim) {
            anim.beginElement();
        }
    }
    
    // Add node markers
    points.forEach((p, i) => {
        const isLatest = i === points.length - 1;
        const time = p.timestamp ? p.timestamp.split(' ')[1]?.substring(0, 8) || '' : '';
        const camName = p.camera_name || p.camera_id || '';
        
        // Convert SVG viewbox coords to percentage positions
        const leftPct = (p.x / 800 * 100);
        const topPct = (p.y / 600 * 100);
        
        const node = document.createElement('div');
        node.className = `absolute z-20 -translate-x-1/2 -translate-y-1/2 group/node cursor-pointer`;
        node.style.left = leftPct + '%';
        node.style.top = topPct + '%';
        
        if (isLatest) {
            node.innerHTML = `
                <div class="w-6 h-6 bg-secondary border-heavy transform rotate-45 marker-pulse relative shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]"></div>
                <div class="absolute top-full left-1/2 -translate-x-1/2 mt-2 bg-primary text-on-primary border-heavy p-2 whitespace-nowrap shadow-hard z-30">
                    <span class="font-label-mono text-label-mono font-bold uppercase text-secondary">LATEST: ${time}</span><br/>
                    <span class="font-label-mono text-[10px] uppercase">${camName}</span>
                </div>`;
        } else {
            node.innerHTML = `
                <div class="w-4 h-4 bg-surface border-heavy transform rotate-45 group-hover/node:bg-primary transition-colors"></div>
                <div class="absolute top-full left-1/2 -translate-x-1/2 mt-2 bg-surface border-heavy p-1 opacity-0 group-hover/node:opacity-100 transition-opacity pointer-events-none whitespace-nowrap shadow-hard z-30">
                    <span class="font-label-mono text-[10px] uppercase">${time} ${camName}</span>
                </div>`;
        }
        
        mapContainer.appendChild(node);
    });
}

// === HEATMAP TAB ===
function loadHeatmap() {
    const container = document.getElementById('heatmapContainer');
    if (!container) return;
    container.innerHTML = `
        <div class="relative w-full h-full min-h-[600px]">
            <iframe src="/api/map/overview" class="w-full h-full min-h-[600px] border-none" title="Camera Traffic Network Map" onerror="loadCameraGrid()"></iframe>
        </div>`;
}

async function loadCameraGrid() {
    try {
        const cameras = await fetch('/api/camera-heatmap').then(r => r.json());
        const container = document.getElementById('heatmapContainer');
        if (!container) return;
        
        let html = '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-gutter p-gutter">';
        cameras.sort((a, b) => b.count - a.count).forEach(cam => {
            html += `
                <div class="bg-surface border-heavy shadow-hard p-gutter">
                    <div class="flex items-center gap-2 mb-3">
                        <span class="material-symbols-outlined text-secondary">videocam</span>
                        <h3 class="font-headline-md text-headline-md font-bold uppercase">${cam.name}</h3>
                    </div>
                    <div class="font-metric-xl text-metric-xl text-primary mb-2">${Number(cam.count).toLocaleString()}</div>
                    <div class="font-label-mono text-label-mono text-on-surface-variant uppercase">Detections</div>
                    <div class="font-label-mono text-label-mono text-outline mt-2">
                        ${cam.latitude?.toFixed(4) || 'N/A'}, ${cam.longitude?.toFixed(4) || 'N/A'}
                    </div>
                </div>`;
        });
        html += '</div>';
        container.innerHTML = html;
    } catch (err) {
        console.error('Camera grid error:', err);
    }
}

// === ROUTES TAB ===
async function loadRoutes() {
    try {
        const routes = await fetch('/api/routes').then(r => r.json());
        const tbody = document.getElementById('routesBody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        if (!routes || !routes.length) {
            tbody.innerHTML = `<tr><td colspan="6" class="p-4 text-center text-on-surface-variant">No inter-camera vehicle routes recorded yet</td></tr>`;
            return;
        }

        routes.forEach((r, i) => {
            const travelMinStr = (r.avg_travel_seconds && Number(r.avg_travel_seconds) > 0)
                ? (Number(r.avg_travel_seconds) / 60).toFixed(1) + ' min'
                : 'N/A';
            tbody.innerHTML += `
                <tr class="border-b border-outline-variant hover:bg-surface-container-high transition-colors">
                    <td class="p-3 border-r border-outline-variant font-bold">${i + 1}</td>
                    <td class="p-3 border-r border-outline-variant">
                        <span class="font-label-mono">${r.from_name || r.from_camera}</span>
                    </td>
                    <td class="p-3 border-r border-outline-variant text-center">
                        <span class="material-symbols-outlined text-secondary">arrow_forward</span>
                    </td>
                    <td class="p-3 border-r border-outline-variant">
                        <span class="font-label-mono">${r.to_name || r.to_camera}</span>
                    </td>
                    <td class="p-3 border-r border-outline-variant font-metric-xl text-2xl text-primary font-bold">${r.count}</td>
                    <td class="p-3 font-label-mono">${travelMinStr}</td>
                </tr>`;
        });
    } catch (err) {
        console.error('Routes load error:', err);
    }
}

// === ALERTS TAB ===
async function loadAlerts() {
    try {
        // Load blacklist
        const blacklist = await fetch('/api/blacklist').then(r => r.json());
        const blacklistContainer = document.getElementById('blacklistList');
        if (blacklistContainer) {
            blacklistContainer.innerHTML = '';
            blacklist.forEach(item => {
                blacklistContainer.innerHTML += `
                    <div class="flex justify-between items-center p-3 border-b border-outline-variant">
                        <div>
                            <span class="font-label-mono text-label-mono font-bold text-primary text-lg tracking-widest">${item.plate}</span>
                            <span class="font-label-mono text-label-mono text-on-surface-variant ml-2">${item.reason || 'Flagged'}</span>
                        </div>
                        <button onclick="removeFromBlacklist('${item.plate}')" class="bg-surface text-secondary font-label-mono text-label-mono py-1 px-3 border-2 border-secondary hover:bg-secondary hover:text-on-secondary transition-colors">
                            REMOVE
                        </button>
                    </div>`;
            });
        }
        
        // Load alerts
        const alerts = await fetch('/api/alerts').then(r => r.json());
        const alertsContainer = document.getElementById('alertsContainer');
        if (alertsContainer) {
            alertsContainer.innerHTML = '';
            
            // Count unacknowledged
            const unackCount = alerts.filter(a => !a.acknowledged).length;
            const unackBadge = document.getElementById('unackAlertCount');
            if (unackBadge) unackBadge.textContent = unackCount + ' UNACKNOWLEDGED';
            
            alerts.forEach(alert => {
                const isAcked = alert.acknowledged;
                const borderClass = isAcked ? 'border-outline-variant' : 'border-secondary';
                const time = alert.timestamp ? alert.timestamp.split(' ')[1]?.substring(0, 8) || alert.timestamp : alert.created_at || '';
                
                alertsContainer.innerHTML += `
                    <div class="bg-surface border-2 ${borderClass} p-stack-md neo-shadow relative mb-4">
                        ${!isAcked ? '<div class="absolute top-0 left-0 w-full h-1 bg-secondary"></div>' : ''}
                        <div class="flex justify-between items-start mb-4">
                            <div>
                                <span class="font-label-mono text-label-mono ${isAcked ? 'bg-surface-container text-on-surface-variant' : 'bg-secondary text-on-secondary'} px-2 py-0.5 text-xs mb-2 inline-block">${alert.reason || 'ALERT'}</span>
                                <h4 class="font-metric-xl text-metric-xl text-primary font-black uppercase leading-none">${alert.plate_text}</h4>
                            </div>
                            <div class="text-right">
                                <span class="font-label-mono text-label-mono text-outline block text-xs">TIME DETECTED</span>
                                <span class="font-headline-md text-headline-md font-bold ${isAcked ? 'text-outline' : 'text-secondary'}">${time}</span>
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-4 mb-4 border-t border-outline-variant pt-4">
                            <div>
                                <span class="font-label-mono text-label-mono text-outline block text-xs mb-1">CAMERA ID</span>
                                <p class="font-body-md text-body-md font-medium flex items-center gap-1">
                                    <span class="material-symbols-outlined text-sm">location_on</span> ${alert.camera_id}
                                </p>
                            </div>
                            <div>
                                <span class="font-label-mono text-label-mono text-outline block text-xs mb-1">STATUS</span>
                                <p class="font-body-md text-body-md font-medium ${isAcked ? 'text-outline' : 'text-secondary'}">${isAcked ? 'ACKNOWLEDGED' : 'UNACKNOWLEDGED'}</p>
                            </div>
                        </div>
                        <div class="flex gap-4">
                            ${!isAcked ? `<button onclick="acknowledgeAlert(${alert.id})" class="bg-primary text-on-primary font-label-mono text-label-mono py-2 px-4 neo-border neo-shadow-sm hover-lift active-press transition-all uppercase flex-grow text-center">ACKNOWLEDGE</button>` : ''}
                            <button onclick="document.getElementById('plateSearchInput').value='${alert.plate_text}'; switchTab('lookup'); lookupPlate();" class="bg-surface text-primary font-label-mono text-label-mono py-2 px-4 border-2 border-primary hover:bg-surface-container transition-colors flex items-center justify-center">
                                <span class="material-symbols-outlined">visibility</span>
                            </button>
                        </div>
                    </div>`;
            });
        }
    } catch (err) {
        console.error('Alerts load error:', err);
    }
}

async function addToBlacklist() {
    const plateInput = document.getElementById('blacklistPlateInput');
    const reasonSelect = document.getElementById('blacklistReasonSelect');
    
    if (!plateInput || !plateInput.value.trim()) {
        alert('Please enter a license plate.');
        return;
    }
    
    try {
        await fetch('/api/blacklist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plate: plateInput.value.trim().toUpperCase(),
                reason: reasonSelect ? reasonSelect.value : 'Flagged'
            })
        });
        plateInput.value = '';
        loadAlerts();
    } catch (err) {
        alert('Failed to add to blacklist: ' + err.message);
    }
}

async function removeFromBlacklist(plate) {
    try {
        await fetch(`/api/blacklist/${plate}`, { method: 'DELETE' });
        loadAlerts();
    } catch (err) {
        alert('Failed to remove from blacklist: ' + err.message);
    }
}

async function acknowledgeAlert(alertId) {
    try {
        await fetch(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' });
        loadAlerts();
        loadDashboard(); // Refresh alert count
    } catch (err) {
        alert('Failed to acknowledge: ' + err.message);
    }
}

// === MAP ZOOM CONTROLS ===
function setupMapControls() {
    const zoomIn = document.getElementById('mapZoomIn');
    const zoomOut = document.getElementById('mapZoomOut');
    const reset = document.getElementById('mapReset');
    const zoomLayer = document.getElementById('mapZoomLayer');

    if (!zoomLayer) return;

    if (zoomIn) {
        zoomIn.addEventListener('click', () => {
            mapZoomLevel = Math.min(mapZoomLevel + 0.25, 3);
            zoomLayer.style.transform = `scale(${mapZoomLevel})`;
        });
    }

    if (zoomOut) {
        zoomOut.addEventListener('click', () => {
            mapZoomLevel = Math.max(mapZoomLevel - 0.25, 0.5);
            zoomLayer.style.transform = `scale(${mapZoomLevel})`;
        });
    }

    if (reset) {
        reset.addEventListener('click', () => {
            mapZoomLevel = 1;
            zoomLayer.style.transform = 'scale(1)';
        });
    }

    // Mouse wheel zoom
    const container = document.getElementById('mapOuterContainer');
    if (container) {
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY < 0) {
                mapZoomLevel = Math.min(mapZoomLevel + 0.1, 3);
            } else {
                mapZoomLevel = Math.max(mapZoomLevel - 0.1, 0.5);
            }
            zoomLayer.style.transform = `scale(${mapZoomLevel})`;
        }, { passive: false });
    }
}
