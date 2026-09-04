// CITY-ANPR Surveillance & Traffic Intelligence Platform Controller (Phase 7D)

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initGlobalSearch();
    initMapControls();
    initAnalyticsView();
    initVehicleSearch();
    initSliders();
    initVideoUpload();
    initBlacklist();

    // Initial Data Fetch
    loadHotspots();
    loadNetworkAnalytics();
    loadDashboardStats();
    loadRecentActivity();
    loadCameras();
    updateSystemHealthStatus();
    loadJobQueue();

    // Polling intervals: jobs every 2s, telemetry & health every 5s
    setInterval(loadJobQueue, 2000);
    setInterval(() => {
        loadDashboardStats();
        loadRecentActivity();
        loadCameras();
        updateSystemHealthStatus();
    }, 5000);
});

// ---------------------------------------------------------------------------
// 1. SPA Tab Navigation System
// ---------------------------------------------------------------------------
function initNavigation() {
    const tabs = [
        { btnId: 'navTrafficMap', tabId: 'tabTrafficMap', onShow: loadHotspots },
        { btnId: 'navAnalytics', tabId: 'tabAnalytics', onShow: loadNetworkAnalytics },
        { btnId: 'navVehicleLookup', tabId: 'tabVehicleLookup' },
        { btnId: 'navDashboard', tabId: 'tabDashboard', onShow: loadDashboardStats },
        { btnId: 'navVideoHub', tabId: 'tabVideoHub' },
        { btnId: 'navAlertCenter', tabId: 'tabAlertCenter', onShow: loadBlacklistAndAlerts }
    ];

    tabs.forEach(t => {
        const btn = document.getElementById(t.btnId);
        if (btn) {
            btn.addEventListener('click', () => {
                switchTab(t.tabId, t.btnId);
                if (t.onShow) t.onShow();
            });
        }
    });

    const brandLogo = document.getElementById('brandLogoBtn');
    if (brandLogo) {
        brandLogo.addEventListener('click', () => switchTab('tabTrafficMap', 'navTrafficMap'));
    }

    const headerThreatBtn = document.getElementById('headerBlacklistBtn');
    if (headerThreatBtn) {
        headerThreatBtn.addEventListener('click', () => {
            switchTab('tabAlertCenter', 'navAlertCenter');
            loadBlacklistAndAlerts();
        });
    }

    const resetBtn = document.getElementById('resetSystemDataBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to reset all telemetry and detections to zero?')) return;
            try {
                const res = await fetch('/api/reset-data', { method: 'POST' });
                if (res.ok) {
                    loadHotspots();
                    loadNetworkAnalytics();
                    loadDashboardStats();
                    loadRecentActivity();
                    loadJobQueue();
                    alert('System telemetry reset to zero.');
                }
            } catch (err) {
                console.error('Error resetting system data:', err);
            }
        });
    }
}

function switchTab(targetTabId, activeBtnId) {
    const allTabIds = ['tabTrafficMap', 'tabAnalytics', 'tabVehicleLookup', 'tabDashboard', 'tabVideoHub', 'tabAlertCenter'];
    const allBtnIds = ['navTrafficMap', 'navAnalytics', 'navVehicleLookup', 'navDashboard', 'navVideoHub', 'navAlertCenter'];

    allTabIds.forEach(tid => {
        const tabEl = document.getElementById(tid);
        if (tabEl) {
            if (tid === targetTabId) {
                tabEl.classList.remove('hidden');
            } else {
                tabEl.classList.add('hidden');
            }
        }
    });

    allBtnIds.forEach(bid => {
        const btnEl = document.getElementById(bid);
        if (btnEl) {
            if (bid === activeBtnId) {
                btnEl.classList.add('nav-link-active');
            } else {
                btnEl.classList.remove('nav-link-active');
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 2. Top Global Search Bar
// ---------------------------------------------------------------------------
function initGlobalSearch() {
    const input = document.getElementById('globalSearchInput');
    if (!input) return;

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const val = input.value.trim();
            if (val) {
                switchTab('tabVehicleLookup', 'navVehicleLookup');
                const pInput = document.getElementById('plateSearchInput');
                if (pInput) pInput.value = val;
                performVehicleSearch(val);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 3. LAYER 1: GIS Map View Controls & Hotspots
// ---------------------------------------------------------------------------
function initMapControls() {
    const reloadBtn = document.getElementById('reloadMapBtn');
    if (reloadBtn) {
        reloadBtn.addEventListener('click', () => {
            const iframe = document.getElementById('cityMapIframe');
            if (iframe) {
                iframe.src = '/api/v1/gis/folium-map?t=' + Date.now();
            }
            loadHotspots();
        });
    }
}

async function loadHotspots() {
    const listEl = document.getElementById('hotspotsList');
    const badgeEl = document.getElementById('hotspotsCountBadge');
    if (!listEl) return;

    try {
        const res = await fetch('/api/v1/analytics/hotspots');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const hotspots = await res.json();

        if (badgeEl) badgeEl.textContent = hotspots.length;

        if (!hotspots || hotspots.length === 0) {
            listEl.innerHTML = `
                <div class="p-3 rounded bg-surface-subtle border border-border-light text-center text-text-muted text-xs">
                    <span class="material-symbols-outlined text-success text-lg block mb-1">verified</span>
                    <span>No congested hotspots detected. All monitored corridors operating under threshold.</span>
                </div>
            `;
            return;
        }

        listEl.innerHTML = hotspots.map((h, i) => `
            <div class="p-2.5 rounded bg-error-bg/30 border border-red-200 hover:bg-error-bg/50 transition-colors cursor-pointer space-y-1" onclick="focusHotspot('${h.corridor}')">
                <div class="flex justify-between items-center text-xs">
                    <span class="font-bold text-text-main flex items-center gap-1">
                        <span class="w-4 h-4 rounded-full bg-error text-white text-[10px] flex items-center justify-center font-bold">${i + 1}</span>
                        <span>${h.corridor}</span>
                    </span>
                    <span class="los-badge los-${h.los_proxy || 'D'}">LOS ${h.los_proxy || 'D'}</span>
                </div>
                <div class="flex justify-between text-[11px] text-text-muted">
                    <span>TTI: <b class="text-error font-mono">${Number(h.tti || 0).toFixed(2)}</b></span>
                    <span>Degradation: <b class="text-text-main font-mono">${Number(h.speed_degradation_pct || 0).toFixed(1)}%</b></span>
                    <span>Flow: <b class="text-text-main font-mono">${Number(h.transit_rate_veh_hr || 0).toFixed(0)}/hr</b></span>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('Error loading hotspots:', err);
        listEl.innerHTML = `<p class="text-xs text-error py-2 text-center">Failed to load hotspots.</p>`;
    }
}

function focusHotspot(corridorId) {
    const iframe = document.getElementById('cityMapIframe');
    if (iframe) {
        iframe.src = `/api/v1/gis/folium-map?corridor=${encodeURIComponent(corridorId)}&t=` + Date.now();
    }
}

// ---------------------------------------------------------------------------
// 4. LAYER 2: Network Analytics View
// ---------------------------------------------------------------------------
function initAnalyticsView() {
    const refreshBtn = document.getElementById('refreshAnalyticsBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadNetworkAnalytics);
    }
}

async function loadNetworkAnalytics() {
    await Promise.all([
        loadAnalyticsSummary(),
        loadCorridorsTable(),
        loadOdMatrixTable(),
        loadTimeWindows()
    ]);
}

async function loadAnalyticsSummary() {
    try {
        const res = await fetch('/api/v1/analytics/summary');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        document.getElementById('analyticsTotalVehicles').textContent = data.total_vehicles_observed ?? 0;
        document.getElementById('analyticsTransitObs').textContent = data.total_transit_observations ?? 0;
        document.getElementById('analyticsActiveCorridors').textContent = data.active_corridors_count ?? 0;

        const ttiEl = document.getElementById('analyticsAverageTti');
        const badgeEl = document.getElementById('analyticsAverageTtiBadge');

        if (data.network_average_tti !== null && data.network_average_tti !== undefined) {
            ttiEl.textContent = Number(data.network_average_tti).toFixed(2);
            let los = 'A';
            const val = Number(data.network_average_tti);
            if (val > 2.50) los = 'F';
            else if (val > 2.00) los = 'E';
            else if (val > 1.50) los = 'D';
            else if (val > 1.25) los = 'C';
            else if (val > 1.10) los = 'B';

            badgeEl.className = `los-badge los-${los}`;
            badgeEl.textContent = `LOS ${los}`;
        } else {
            ttiEl.textContent = '--';
            badgeEl.className = 'los-badge los-UNKNOWN';
            badgeEl.textContent = 'N/A';
        }

        // Modal breakdown
        const modal = data.modal_flow_breakdown || {};
        document.getElementById('modalCountCar').textContent = modal.car || 0;
        document.getElementById('modalCountMotorcycle').textContent = modal.motorcycle || 0;
        document.getElementById('modalCountBus').textContent = modal.bus || 0;
        document.getElementById('modalCountTruck').textContent = modal.truck || 0;

    } catch (err) {
        console.error('Error loading analytics summary:', err);
    }
}

async function loadCorridorsTable() {
    const tbody = document.getElementById('corridorsTableBody');
    if (!tbody) return;

    try {
        const res = await fetch('/api/v1/analytics/corridors');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const corridors = await res.json();

        if (!corridors || corridors.length === 0) {
            tbody.innerHTML = `<tr><td colspan="10" class="py-6 text-center text-text-muted">No inter-camera transit observations recorded yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = corridors.map(c => {
            const los = c.los_proxy || 'UNKNOWN';
            const ttiStr = c.travel_time_index !== null ? Number(c.travel_time_index).toFixed(2) : '--';
            const degStr = c.speed_degradation_pct !== null ? `${Number(c.speed_degradation_pct).toFixed(1)}%` : '--';
            const medSpeedStr = c.speed_median_kmh !== null ? `${Number(c.speed_median_kmh).toFixed(1)} km/h` : '--';
            const p05SpeedStr = c.speed_p05_kmh !== null ? `${Number(c.speed_p05_kmh).toFixed(1)} km/h` : '--';
            const p95SpeedStr = c.speed_p95_kmh !== null ? `${Number(c.speed_p95_kmh).toFixed(1)} km/h` : '--';
            const medTtStr = c.travel_time_median_s !== null ? `${Number(c.travel_time_median_s).toFixed(1)}s` : '--';
            const confStr = c.sample_confidence_score !== null ? Number(c.sample_confidence_score).toFixed(2) : '--';

            return `
                <tr class="hover:bg-surface-subtle transition-colors">
                    <td class="py-2.5 px-3 font-semibold text-text-main">${c.corridor_id || `${c.from_camera_id} -> ${c.to_camera_id}`}</td>
                    <td class="py-2.5 px-3 font-mono">${c.observation_count ?? c.sample_size ?? 0}</td>
                    <td class="py-2.5 px-3 font-mono font-medium">${medSpeedStr}</td>
                    <td class="py-2.5 px-3 font-mono text-error font-medium">${p05SpeedStr}</td>
                    <td class="py-2.5 px-3 font-mono">${p95SpeedStr}</td>
                    <td class="py-2.5 px-3 font-mono">${medTtStr}</td>
                    <td class="py-2.5 px-3 font-mono font-bold ${c.travel_time_index && c.travel_time_index >= 1.5 ? 'text-error' : 'text-text-main'}">${ttiStr}</td>
                    <td class="py-2.5 px-3 font-mono">${degStr}</td>
                    <td class="py-2.5 px-3"><span class="los-badge los-${los}">LOS ${los}</span></td>
                    <td class="py-2.5 px-3 font-mono">${confStr}</td>
                </tr>
            `;
        }).join('');

    } catch (err) {
        console.error('Error loading corridors table:', err);
        tbody.innerHTML = `<tr><td colspan="10" class="py-4 text-center text-error">Failed to load corridor statistics.</td></tr>`;
    }
}

async function loadOdMatrixTable() {
    const tbody = document.getElementById('odTableBody');
    if (!tbody) return;

    try {
        const res = await fetch('/api/v1/analytics/od-matrix');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const records = await res.json();

        if (!records || records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="py-6 text-center text-text-muted">No completed multi-camera vehicle trips recorded yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => `
            <tr class="hover:bg-surface-subtle transition-colors">
                <td class="py-2.5 px-3 font-semibold text-text-main">${r.origin}</td>
                <td class="py-2.5 px-3 font-semibold text-text-main">${r.destination}</td>
                <td class="py-2.5 px-3 font-bold text-primary font-mono">${r.trip_count}</td>
                <td class="py-2.5 px-3 font-mono">${r.median_duration_s !== null ? `${Number(r.median_duration_s).toFixed(1)}s` : '--'}</td>
                <td class="py-2.5 px-3 font-mono">${r.median_distance_km !== null ? `${Number(r.median_distance_km).toFixed(2)} km` : '--'}</td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('Error loading OD matrix:', err);
        tbody.innerHTML = `<tr><td colspan="5" class="py-4 text-center text-error">Failed to load origin-destination flows.</td></tr>`;
    }
}

async function loadTimeWindows() {
    const container = document.getElementById('timeWindowsContainer');
    if (!container) return;

    try {
        const res = await fetch('/api/v1/analytics/time-windows');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const windows = await res.json();

        if (!windows || windows.length === 0) {
            container.innerHTML = `<p class="text-xs text-text-muted col-span-3 py-2">No departure-time window activity recorded.</p>`;
            return;
        }

        container.innerHTML = windows.map(tw => `
            <div class="p-3 rounded bg-surface-subtle border border-border-light space-y-1">
                <div class="flex justify-between items-center text-xs">
                    <span class="font-bold text-text-main">${tw.time_window}</span>
                    <span class="text-[11px] text-text-muted">${tw.active_corridors} Corridors</span>
                </div>
                <div class="text-[11px] text-text-muted">
                    <span>Active road links: <b>${tw.active_corridors}</b></span>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('Error loading time windows:', err);
        container.innerHTML = `<p class="text-xs text-error col-span-3 py-2">Failed to load departure profiles.</p>`;
    }
}

// ---------------------------------------------------------------------------
// 5. LAYER 3: Vehicle Search & Trajectory Explorer
// ---------------------------------------------------------------------------
function initVehicleSearch() {
    const searchBtn = document.getElementById('lookupBtn');
    const input = document.getElementById('plateSearchInput');

    if (searchBtn && input) {
        searchBtn.addEventListener('click', () => {
            performVehicleSearch(input.value.trim());
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                performVehicleSearch(input.value.trim());
            }
        });
    }

    // Sample pills
    document.querySelectorAll('.sample-search-pill').forEach(pill => {
        pill.addEventListener('click', () => {
            const q = pill.dataset.query;
            if (input) input.value = q;
            performVehicleSearch(q);
        });
    });
}

async function performVehicleSearch(query) {
    const msgBanner = document.getElementById('searchMessageBanner');
    const detailsContainer = document.getElementById('vehicleDetailsContainer');

    if (!query) {
        showSearchMessage('Please enter a license plate number or Global ID (GV-XXXXXX).', 'warning');
        if (detailsContainer) detailsContainer.classList.add('hidden');
        return;
    }

    try {
        const res = await fetch(`/api/v1/vehicles/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();

        if (res.status === 400) {
            showSearchMessage(data.error || 'Invalid query format. Must be alphanumeric plate or GV-XXXXXX.', 'warning');
            if (detailsContainer) detailsContainer.classList.add('hidden');
            return;
        }

        if (res.status === 404) {
            showSearchMessage(data.error || `No vehicle found matching identifier '${query}'.`, 'error');
            if (detailsContainer) detailsContainer.classList.add('hidden');
            return;
        }

        if (!res.ok) {
            showSearchMessage(`Server error (HTTP ${res.status}).`, 'error');
            if (detailsContainer) detailsContainer.classList.add('hidden');
            return;
        }

        // 200 Success: render dossier and hops
        if (msgBanner) msgBanner.classList.add('hidden');
        if (detailsContainer) detailsContainer.classList.remove('hidden');

        renderVehicleDossier(data, query);

    } catch (err) {
        console.error('Error during vehicle search:', err);
        showSearchMessage('Network connection error while searching vehicle.', 'error');
        if (detailsContainer) detailsContainer.classList.add('hidden');
    }
}

function showSearchMessage(msg, type = 'error') {
    const msgBanner = document.getElementById('searchMessageBanner');
    if (!msgBanner) return;

    msgBanner.classList.remove('hidden', 'bg-error-bg', 'text-error', 'border-red-200', 'bg-amber-50', 'text-amber-800', 'border-amber-200');

    if (type === 'error') {
        msgBanner.classList.add('bg-error-bg', 'text-error', 'border', 'border-red-200');
    } else {
        msgBanner.classList.add('bg-amber-50', 'text-amber-800', 'border', 'border-amber-200');
    }

    msgBanner.textContent = msg;
}

function renderVehicleDossier(data, originalQuery) {
    document.getElementById('dossierPlateDisplay').textContent = data.canonical_plate || 'UNIDENTIFIED';
    document.getElementById('dossierGlobalId').textContent = data.global_id || '--';
    document.getElementById('dossierVehicleType').textContent = data.vehicle_type || 'car';

    const confVal = data.plate_confidence !== null ? `${(Number(data.plate_confidence) * 100).toFixed(0)}%` : '--';
    document.getElementById('dossierPlateConf').textContent = confVal;

    const summary = data.trajectory_summary || {};
    document.getElementById('dossierSightingsCount').textContent = summary.sightings_count ?? 0;
    document.getElementById('dossierNetworkDist').textContent = `${Number(summary.total_network_distance_km || 0).toFixed(2)} km`;
    document.getElementById('dossierHaversineDist').textContent = `${Number(summary.total_haversine_distance_km || 0).toFixed(2)} km`;
    document.getElementById('dossierDuration').textContent = `${Number(summary.total_duration_seconds || 0).toFixed(1)}s`;

    const avgSpd = summary.average_speed_kmh !== null && summary.average_speed_kmh !== undefined
        ? `${Number(summary.average_speed_kmh).toFixed(1)} km/h`
        : '--';
    document.getElementById('dossierAvgSpeed').textContent = avgSpd;

    // ReID Baseline Diagnostics
    const reid = data.reid_diagnostics || {};
    const reidStatusEl = document.getElementById('dossierReidStatus');
    if (reid.has_embedding) {
        reidStatusEl.className = 'font-semibold text-success';
        reidStatusEl.textContent = 'Active (Aggregated)';
    } else {
        reidStatusEl.className = 'font-semibold text-text-muted';
        reidStatusEl.textContent = 'None Recorded';
    }

    document.getElementById('dossierReidDim').textContent = `${reid.dimension || 512}-D`;
    document.getElementById('dossierReidNorm').textContent = reid.l2_norm ? `||e|| = ${Number(reid.l2_norm).toFixed(4)}` : 'N/A';

    // Anomaly Badges
    const badgeBox = document.getElementById('dossierAnomalyBadges');
    if (badgeBox) {
        const badges = [];
        if (summary.has_velocity_anomaly) {
            badges.push(`<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-red-100 text-error border border-red-300">Velocity Anomaly (&gt;140 km/h)</span>`);
        }
        if (summary.has_temporal_anomaly) {
            badges.push(`<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-red-100 text-error border border-red-300">Temporal Anomaly (Δt ≤ 0)</span>`);
        }
        if (summary.has_unreachable_segment) {
            badges.push(`<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-100 text-amber-800 border border-amber-300">Unreachable Graph Transition</span>`);
        }
        if (badges.length === 0) {
            badges.push(`<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-success-bg text-success border border-green-200">Physically Plausible Trajectory</span>`);
        }
        badgeBox.innerHTML = badges.join('');
    }

    // Chronological Hops Table
    const hops = data.sighting_hops || [];
    const hopsBody = document.getElementById('sightingHopsBody');
    const hopBadge = document.getElementById('trajectoryHopCountBadge');

    if (hopBadge) hopBadge.textContent = `${hops.length} Hops`;

    if (hopsBody) {
        if (hops.length === 0) {
            hopsBody.innerHTML = `<tr><td colspan="6" class="py-4 text-center text-text-muted">No sighting hops recorded.</td></tr>`;
        } else {
            hopsBody.innerHTML = hops.map(h => {
                const spdStr = h.transit_speed_to_next_kmh !== null ? `${Number(h.transit_speed_to_next_kmh).toFixed(1)} km/h` : 'End of route';
                return `
                    <tr class="hover:bg-surface-subtle transition-colors ${h.is_anomaly ? 'bg-red-50/50' : ''}">
                        <td class="py-2 px-2.5 font-bold font-mono text-primary">#${h.hop_index}</td>
                        <td class="py-2 px-2.5 font-semibold text-text-main">${h.camera_name || h.camera_id}</td>
                        <td class="py-2 px-2.5 font-mono text-[11px]">${h.first_seen_iso}</td>
                        <td class="py-2 px-2.5 font-mono text-[11px]">${h.last_seen_iso}</td>
                        <td class="py-2 px-2.5 font-mono">${Number(h.dwell_duration_seconds).toFixed(1)}s</td>
                        <td class="py-2 px-2.5 font-mono font-medium ${h.is_anomaly ? 'text-error font-bold' : 'text-secondary'}">${spdStr}</td>
                    </tr>
                `;
            }).join('');
        }
    }

    // Trajectory Map Overlay
    const mapIframe = document.getElementById('trajectoryMapIframe');
    if (mapIframe) {
        mapIframe.src = `/api/v1/gis/folium-map?q=${encodeURIComponent(data.global_id || originalQuery)}&t=` + Date.now();
    }
}

// ---------------------------------------------------------------------------
// 6. UTILITY: Telemetry Dashboard & Recent Activity (Legacy compatibility)
// ---------------------------------------------------------------------------
async function loadDashboardStats() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) return;
        const s = await res.json();
        document.getElementById('statTotalHits').textContent = s.total_detections ?? '--';
        document.getElementById('statUniquePlates').textContent = s.unique_plates ?? '--';
        document.getElementById('statActiveCameras').textContent = s.unique_cameras ?? '--';
        document.getElementById('statUnackAlerts').textContent = s.unacknowledged_alerts ?? '0';

        const badge = document.getElementById('headerNotificationBadge');
        if (badge) {
            badge.textContent = s.unacknowledged_alerts || 0;
            if (s.unacknowledged_alerts > 0) badge.classList.remove('hidden');
            else badge.classList.add('hidden');
        }
    } catch (err) {
        console.error('Error loading dashboard stats:', err);
    }
}

async function loadRecentActivity() {
    const tbody = document.getElementById('recentActivityBody');
    if (!tbody) return;

    try {
        const res = await fetch('/api/recent-activity');
        if (!res.ok) return;
        const rows = await res.json();

        if (!rows || rows.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" class="py-4 text-center text-text-muted">No telemetry passages recorded.</td></tr>`;
            return;
        }

        tbody.innerHTML = rows.map(r => `
            <tr class="hover:bg-surface-subtle transition-colors">
                <td class="py-2 px-3 font-bold font-mono text-primary">${r.plate_text}</td>
                <td class="py-2 px-3">${r.camera_name || r.camera_id}</td>
                <td class="py-2 px-3 font-mono">${(Number(r.confidence || 0) * 100).toFixed(0)}%</td>
                <td class="py-2 px-3 text-text-muted font-mono text-[11px]">${r.timestamp || r.created_at || '--'}</td>
            </tr>
        `).join('');
    } catch (err) {
        console.error('Error loading recent activity:', err);
    }
}

function getCameraStatusBadge(status, fps, latencyMs) {
    const s = (status || 'offline').toLowerCase();
    if (s === 'online' || s === 'running') {
        const perf = fps > 0 ? ` (${fps.toFixed(1)} FPS, ${Math.round(latencyMs)}ms)` : '';
        return `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>Online${perf}</span>`;
    } else if (s === 'reconnecting') {
        return `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-amber-800 border border-amber-300 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span>Reconnecting</span>`;
    } else {
        return `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 text-slate-600 border border-slate-300 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>Offline</span>`;
    }
}

async function loadCameras() {
    const select = document.getElementById('cameraSelect');
    const statusGrid = document.getElementById('cameraStatusGrid');

    try {
        const res = await fetch('/api/v1/system/cameras');
        if (!res.ok) return;
        const cameras = await res.json();

        if (select) {
            select.innerHTML = '<option value="">Select Camera Location...</option>' +
                cameras.map(c => `<option value="${c.camera_id}">${c.name || c.camera_id}</option>`).join('');
        }

        if (statusGrid) {
            if (cameras.length === 0) {
                statusGrid.innerHTML = `<p class="text-xs text-text-muted">No cameras loaded.</p>`;
            } else {
                statusGrid.innerHTML = cameras.map(c => {
                    const badge = getCameraStatusBadge(c.status, c.fps || 0, c.latency_ms || 0);
                    return `
                    <div class="p-2.5 rounded bg-surface-subtle border border-border-light flex justify-between items-center text-xs">
                        <div>
                            <span class="font-bold text-text-main">${c.camera_id}</span>
                            <span class="text-text-muted block text-[11px]">${c.name || c.camera_id}</span>
                        </div>
                        <div>${badge}</div>
                    </div>`;
                }).join('');
            }
        }
    } catch (err) {
        console.error('Error loading cameras:', err);
    }
}

async function updateSystemHealthStatus() {
    try {
        const res = await fetch('/api/v1/system/health');
        if (!res.ok) return;
        const health = await res.json();

        const dot = document.getElementById('sidebarStatusDot');
        const text = document.getElementById('sidebarStatusText');
        const telemText = document.getElementById('sidebarTelemetryText');

        if (dot && text) {
            if (health.status === 'healthy') {
                dot.className = 'w-2 h-2 rounded-full bg-success animate-pulse';
                text.textContent = 'System Healthy';
            } else if (health.status === 'degraded') {
                dot.className = 'w-2 h-2 rounded-full bg-amber-500 animate-pulse';
                text.textContent = 'System Degraded';
            } else {
                dot.className = 'w-2 h-2 rounded-full bg-slate-400';
                text.textContent = 'System Offline';
            }
        }

        if (telemText) {
            telemText.innerHTML = `Cameras: ${health.active_cameras || 0}/${health.total_cameras || 0} Online<br/>Throughput: ${health.total_fps || 0} FPS (${health.avg_latency_ms || 0}ms)`;
        }
    } catch (err) {
        console.debug('Error updating system health:', err);
    }
}

// ---------------------------------------------------------------------------
// 7. UTILITY: Video Processing Hub
// ---------------------------------------------------------------------------
function initSliders() {
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');
    if (confSlider && confValue) {
        confSlider.addEventListener('input', () => {
            confValue.textContent = `${confSlider.value}%`;
        });
    }

    const ocrSlider = document.getElementById('ocrSlider');
    const ocrValue = document.getElementById('ocrValue');
    if (ocrSlider && ocrValue) {
        ocrSlider.addEventListener('input', () => {
            ocrValue.textContent = `Every ${ocrSlider.value} Frames`;
        });
    }
}

function initVideoUpload() {
    const dropzone = document.getElementById('videoDropzone');
    const fileInput = document.getElementById('videoFileInput');
    const fileNameDisplay = document.getElementById('videoFileNameDisplay');
    const enqueueBtn = document.getElementById('enqueueJobBtn');

    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                const name = fileInput.files[0].name;
                fileNameDisplay.textContent = name;
                fileNameDisplay.classList.remove('hidden');
            }
        });
    }

    if (enqueueBtn) {
        enqueueBtn.addEventListener('click', async () => {
            if (!fileInput || !fileInput.files.length) {
                alert('Please choose a video file to process.');
                return;
            }
            const cameraSelect = document.getElementById('cameraSelect');
            if (!cameraSelect || !cameraSelect.value) {
                alert('Please select a camera location.');
                return;
            }

            const formData = new FormData();
            formData.append('video', fileInput.files[0]);
            formData.append('camera_id', cameraSelect.value);
            formData.append('conf', (Number(document.getElementById('confSlider').value) / 100).toFixed(2));
            formData.append('ocr_n', document.getElementById('ocrSlider').value);

            enqueueBtn.disabled = true;
            enqueueBtn.innerHTML = '<span class="material-symbols-outlined animate-spin text-sm">progress_activity</span> Uploading...';

            try {
                const res = await fetch('/api/upload-video', {
                    method: 'POST',
                    body: formData,
                });
                if (res.ok) {
                    fileInput.value = '';
                    fileNameDisplay.classList.add('hidden');
                    loadJobQueue();
                } else {
                    const err = await res.json();
                    alert(`Upload error: ${err.error || 'Unknown error'}`);
                }
            } catch (err) {
                console.error('Error uploading video:', err);
                alert('Failed to connect to server.');
            } finally {
                enqueueBtn.disabled = false;
                enqueueBtn.innerHTML = '<span class="material-symbols-outlined text-base">play_arrow</span> Start Processing';
            }
        });
    }

    const refreshJobsBtn = document.getElementById('refreshJobsBtn');
    if (refreshJobsBtn) refreshJobsBtn.addEventListener('click', loadJobQueue);

    const clearAllJobsBtn = document.getElementById('clearAllJobsBtn');
    if (clearAllJobsBtn) {
        clearAllJobsBtn.addEventListener('click', async () => {
            if (!confirm('Clear all jobs from queue?')) return;
            try {
                await fetch('/api/jobs/clear-all', { method: 'DELETE' });
                loadJobQueue();
            } catch (err) {
                console.error('Error clearing jobs:', err);
            }
        });
    }
}

async function loadJobQueue() {
    const tbody = document.getElementById('jobQueueBody');
    if (!tbody) return;

    try {
        const res = await fetch('/api/jobs');
        if (!res.ok) return;
        const jobs = await res.json();

        if (!jobs || jobs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="py-4 text-center text-text-muted">No active jobs.</td></tr>`;
            return;
        }

        tbody.innerHTML = jobs.map(j => {
            const isDone = j.status === 'completed';
            const isFailed = j.status === 'failed';
            const statusColor = isDone ? 'text-success bg-success-bg border-green-200' : isFailed ? 'text-error bg-error-bg border-red-200' : 'text-primary bg-primary/10 border-primary/20';

            const downloadBtn = isDone && j.output_video
                ? `<a href="/api/job/${j.job_id}/download" class="text-primary hover:underline font-semibold text-xs flex items-center gap-0.5"><span class="material-symbols-outlined text-sm">download</span> MP4</a>`
                : '--';

            return `
                <tr class="hover:bg-surface-subtle transition-colors">
                    <td class="py-2 px-3 font-mono font-bold">${j.job_id}</td>
                    <td class="py-2 px-3">${j.camera_id}</td>
                    <td class="py-2 px-3"><span class="px-2 py-0.5 rounded text-[10px] font-semibold border ${statusColor}">${j.status}</span></td>
                    <td class="py-2 px-3 font-mono">${j.progress ?? 0}%</td>
                    <td class="py-2 px-3">${downloadBtn}</td>
                    <td class="py-2 px-3">
                        <button onclick="deleteJob('${j.job_id}')" class="text-text-muted hover:text-error transition-colors">
                            <span class="material-symbols-outlined text-base">delete</span>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loading job queue:', err);
    }
}

async function deleteJob(jobId) {
    try {
        await fetch(`/api/job/${jobId}`, { method: 'DELETE' });
        loadJobQueue();
    } catch (err) {
        console.error('Error deleting job:', err);
    }
}

// ---------------------------------------------------------------------------
// 8. UTILITY: Threat Alerts & Blacklist Management (Phase 7E Enriched)
// ---------------------------------------------------------------------------
let currentAlertFilter = 'ALL';

function initBlacklist() {
    const addBtn = document.getElementById('addBlacklistBtn');
    const plateInput = document.getElementById('blacklistPlateInput');
    const catSelect = document.getElementById('blacklistCategorySelect');
    const sevSelect = document.getElementById('blacklistSeveritySelect');
    const reasonInput = document.getElementById('blacklistReasonInput');
    const scanBtn = document.getElementById('scanAlertsBtn');
    const refreshBtn = document.getElementById('refreshAlertsBtn');

    if (addBtn && plateInput) {
        addBtn.addEventListener('click', async () => {
            const plate = plateInput.value.trim().toUpperCase();
            const category = catSelect ? catSelect.value : 'CUSTOM';
            const severity = sevSelect ? sevSelect.value : 'HIGH';
            const reason = reasonInput && reasonInput.value.trim() ? reasonInput.value.trim() : 'Flagged vehicle';

            if (!plate) {
                alert('Please enter a license plate.');
                return;
            }

            try {
                const res = await fetch('/api/v1/blacklist', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ plate, category, severity, reason }),
                });
                if (res.ok) {
                    plateInput.value = '';
                    if (reasonInput) reasonInput.value = '';
                    loadBlacklistAndAlerts();
                }
            } catch (err) {
                console.error('Error adding to blacklist:', err);
            }
        });
    }

    if (scanBtn) {
        scanBtn.addEventListener('click', async () => {
            scanBtn.disabled = true;
            scanBtn.innerHTML = `<span class="material-symbols-outlined text-sm animate-spin">refresh</span> Scanning...`;
            try {
                const res = await fetch('/api/v1/alerts/scan', { method: 'POST' });
                if (res.ok) {
                    await loadBlacklistAndAlerts();
                }
            } catch (err) {
                console.error('Error triggering alert scan:', err);
            } finally {
                scanBtn.disabled = false;
                scanBtn.innerHTML = `<span class="material-symbols-outlined text-sm">security_update_good</span><span>Scan Trajectories</span>`;
            }
        });
    }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            loadBlacklistAndAlerts();
        });
    }

    // Filter Buttons
    document.querySelectorAll('.alert-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.alert-filter-btn').forEach(b => {
                b.classList.remove('bg-primary', 'text-white');
                b.classList.add('bg-surface-subtle', 'text-text-muted');
            });
            btn.classList.add('bg-primary', 'text-white');
            btn.classList.remove('bg-surface-subtle', 'text-text-muted');
            currentAlertFilter = btn.getAttribute('data-filter') || 'ALL';
            loadAlerts();
        });
    });
}

async function loadBlacklistAndAlerts() {
    await Promise.all([loadBlacklist(), loadAlertsSummary(), loadAlerts()]);
}

async function loadAlertsSummary() {
    try {
        const res = await fetch('/api/v1/alerts/summary');
        if (!res.ok) return;
        const s = await res.json();

        const totalEl = document.getElementById('alertCountTotal');
        const unackEl = document.getElementById('alertCountUnack');
        const critEl = document.getElementById('alertCountCritical');
        const highEl = document.getElementById('alertCountHigh');
        const medEl = document.getElementById('alertCountMedium');
        const lowEl = document.getElementById('alertCountLow');

        if (totalEl) totalEl.textContent = s.total_alerts ?? 0;
        if (unackEl) unackEl.textContent = s.unacknowledged_count ?? 0;

        const bySev = s.unack_by_severity || {};
        if (critEl) critEl.textContent = bySev.CRITICAL ?? 0;
        if (highEl) highEl.textContent = bySev.HIGH ?? 0;
        if (medEl) medEl.textContent = bySev.MEDIUM ?? 0;
        if (lowEl) lowEl.textContent = bySev.LOW ?? 0;

        // Header & Nav Badges
        const unack = s.unacknowledged_count ?? 0;
        const navBadge = document.getElementById('navAlertBadge');
        const headerBadge = document.getElementById('headerNotificationBadge');
        const statUnack = document.getElementById('statUnackAlerts');

        if (navBadge) {
            navBadge.textContent = unack;
            navBadge.classList.toggle('hidden', unack === 0);
        }
        if (headerBadge) {
            headerBadge.textContent = unack;
            headerBadge.classList.toggle('hidden', unack === 0);
        }
        if (statUnack) statUnack.textContent = unack;
    } catch (err) {
        console.error('Error loading alerts summary:', err);
    }
}

async function loadBlacklist() {
    const listEl = document.getElementById('blacklistItemsList');
    const badgeEl = document.getElementById('blacklistCountBadge');
    if (!listEl) return;

    try {
        const res = await fetch('/api/v1/blacklist');
        if (!res.ok) return;
        const items = await res.json();

        if (badgeEl) badgeEl.textContent = items ? items.length : 0;

        if (!items || items.length === 0) {
            listEl.innerHTML = `<p class="text-xs text-text-muted">No flagged plates registered.</p>`;
            return;
        }

        listEl.innerHTML = items.map(b => {
            const sevBadge = b.severity === 'CRITICAL'
                ? 'bg-error text-white'
                : (b.severity === 'HIGH' ? 'bg-orange-100 text-orange-800' : 'bg-amber-100 text-amber-800');

            return `
                <div class="p-2.5 rounded bg-surface-subtle border border-border-light flex justify-between items-start text-xs">
                    <div class="space-y-0.5">
                        <div class="flex items-center gap-1.5">
                            <span class="font-mono font-extrabold text-error">${b.plate}</span>
                            <span class="px-1.5 py-0.2 rounded text-[9px] font-bold ${sevBadge}">${b.severity}</span>
                            <span class="text-[10px] text-text-muted bg-surface px-1 rounded border border-border-light">${b.category}</span>
                        </div>
                        <span class="text-text-muted block text-[11px]">${b.reason || 'Flagged vehicle'}</span>
                    </div>
                    <button onclick="removeBlacklist('${b.plate}')" class="text-text-muted hover:text-error transition-colors p-0.5" title="Remove plate">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loading blacklist:', err);
    }
}

async function removeBlacklist(plate) {
    try {
        await fetch(`/api/v1/blacklist/${encodeURIComponent(plate)}`, { method: 'DELETE' });
        loadBlacklist();
        loadAlertsSummary();
    } catch (err) {
        console.error('Error removing from blacklist:', err);
    }
}

async function loadAlerts() {
    const tbody = document.getElementById('alertsTableBody');
    if (!tbody) return;

    try {
        let url = '/api/v1/alerts?limit=100';
        if (currentAlertFilter === 'UNACK') {
            url += '&unacknowledged=true';
        } else if (currentAlertFilter === 'CRITICAL' || currentAlertFilter === 'HIGH' || currentAlertFilter === 'MEDIUM') {
            url += `&severity=${currentAlertFilter}`;
        }

        const res = await fetch(url);
        if (!res.ok) return;
        const alerts = await res.json();

        if (!alerts || alerts.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-text-muted">No matching security alerts.</td></tr>`;
            return;
        }

        tbody.innerHTML = alerts.map(a => {
            const sevClasses = {
                'CRITICAL': 'bg-red-100 text-red-800 border-red-300 font-bold',
                'HIGH': 'bg-orange-100 text-orange-800 border-orange-300 font-bold',
                'MEDIUM': 'bg-amber-100 text-amber-800 border-amber-300 font-semibold',
                'LOW': 'bg-blue-100 text-blue-800 border-blue-300',
                'INFO': 'bg-gray-100 text-gray-800 border-gray-300',
            }[a.severity] || 'bg-gray-100 text-gray-800';

            const typeLabel = {
                'BLACKLIST_EXACT': 'Watchlist Exact',
                'BLACKLIST_FUZZY': 'Watchlist Fuzzy',
                'VELOCITY_ANOMALY': 'Kinematic Speed',
                'TEMPORAL_INVERSION': 'Temporal Anomaly',
                'TOPOLOGY_VIOLATION': 'Topology Violation',
                'IDENTITY_UNCERTAIN': 'Identity Ambiguity',
                'EXCESSIVE_DWELL': 'Loitering / Dwell',
                'RAPID_LOOPING': 'Corridor Looping',
            }[a.alert_type] || a.alert_type;

            const idDisplay = a.canonical_plate
                ? `<span class="font-mono font-bold text-text-main">${a.canonical_plate}</span>`
                : (a.global_id ? `<span class="font-mono text-primary">${a.global_id}</span>` : '--');

            const jumpId = a.canonical_plate || a.global_id;
            const jumpBtn = jumpId ? `
                <button onclick="jumpToTrajectory('${jumpId}')" class="text-primary hover:underline text-[11px] flex items-center gap-0.5 mt-0.5 font-medium" title="Inspect Vehicle Trajectory">
                    <span class="material-symbols-outlined text-xs">route</span> Inspect
                </button>
            ` : '';

            const ackBtn = a.acknowledged
                ? `<span class="text-text-muted text-[10px] font-semibold bg-surface-subtle px-1.5 py-0.5 rounded border border-border-light">Acked</span>`
                : `<button onclick="ackAlert('${a.alert_id}')" class="px-2 py-0.5 rounded bg-primary text-white text-[10px] font-semibold hover:bg-primary-hover shadow-xs">Ack</button>`;

            const timeStr = a.iso_timestamp ? a.iso_timestamp.replace('T', ' ').replace('Z', '') : '--';

            return `
                <tr class="hover:bg-surface-subtle transition-colors">
                    <td class="py-2.5 px-3">
                        <span class="px-2 py-0.5 rounded text-[10px] border ${sevClasses}">${a.severity}</span>
                    </td>
                    <td class="py-2.5 px-3 text-[11px] font-medium text-text-muted">${typeLabel}</td>
                    <td class="py-2.5 px-3">
                        ${idDisplay}
                        ${jumpBtn}
                    </td>
                    <td class="py-2.5 px-3 font-semibold">${a.camera_id}</td>
                    <td class="py-2.5 px-3">
                        <div class="font-medium text-text-main text-[11px]">${a.title}</div>
                        <div class="text-[10px] text-text-muted truncate max-w-xs" title="${a.description}">${a.description}</div>
                    </td>
                    <td class="py-2.5 px-3 font-mono text-[10px] text-text-muted">${timeStr}</td>
                    <td class="py-2.5 px-3 text-right">
                        ${ackBtn}
                    </td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loading alerts:', err);
    }
}

async function ackAlert(alertId) {
    try {
        await fetch(`/api/v1/alerts/${encodeURIComponent(alertId)}/acknowledge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ operator: 'officer_console' }),
        });
        await loadAlerts();
        await loadAlertsSummary();
    } catch (err) {
        console.error('Error acknowledging alert:', err);
    }
}

function jumpToTrajectory(identifier) {
    if (!identifier) return;
    switchTab('tabVehicleLookup', 'navVehicleLookup');
    const pInput = document.getElementById('plateSearchInput');
    if (pInput) pInput.value = identifier;
    performVehicleSearch(identifier);
}

