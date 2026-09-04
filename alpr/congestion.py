"""
Traffic Flow, Density, and Congestion Modeling Module (Phase 7C).

Provides analytical models for urban traffic network monitoring:
- Flow rate primitives:
    * camera_flow_rate_veh_hr: unique vehicle observations passing a camera node per hour.
    * corridor_transit_rate_veh_hr: inter-camera observed vehicle transitions per hour.
    * modal_flow_breakdown: throughput by vehicle class (car, motorcycle, bus, truck).
- Density and occupancy primitives:
    * estimated_density_veh_km: spatial density derived via fundamental relation k = q / v.
    * estimated_temporal_occupancy_ratio: true temporal occupancy in [0.0, 1.0] calculated
      via union of active track intervals.
- Kinematic degradation & indices:
    * free_flow_speed_kmh and free_flow_speed_source (CORRIDOR_CONFIG vs DEFAULT_ASSUMPTION).
    * travel_time_index (TTI = t_median / t_free_flow).
    * speed_performance_index (SPI = v_median / v_free_flow * 100).
    * speed_degradation_pct = (1.0 - v_median / v_free_flow) * 100.
    * travel_time_increase_pct = (TTI - 1.0) * 100.
- Project LOS Proxy & Congestion Classification:
    * los_proxy: TTI-based Level of Service proxy (A through F), explicitly distinguished
      from facility-specific HCM LOS methodology.
    * congestion_category: FREE_FLOW, LIGHT, MODERATE, HEAVY, BREAKDOWN, INSUFFICIENT_DATA.
    * sample_confidence_score: min(1.0, N / 10), sample-size sufficiency indicator.
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from alpr.trajectory import VehicleTrajectory, TrajectoryReconstructor, get_reconstructor
from alpr.analytics import CorridorAnalytics, CorridorAnalyticsEngine, get_analytics_engine

logger = logging.getLogger("alpr.congestion")


def compute_interval_union_duration(intervals: List[Tuple[float, float]], window_start: float, window_end: float) -> float:
    """
    Compute the total occupied duration from the union of overlapping track intervals,
    clipped to [window_start, window_end]. Guaranteed to be <= (window_end - window_start).
    """
    if window_end <= window_start or not intervals:
        return 0.0

    clipped = []
    for s, e in intervals:
        cs = max(window_start, min(window_end, s))
        ce = max(window_start, min(window_end, e))
        if ce > cs:
            clipped.append((cs, ce))

    if not clipped:
        return 0.0

    clipped.sort(key=lambda x: x[0])

    merged = []
    cur_start, cur_end = clipped[0]
    for s, e in clipped[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    return sum(e - s for s, e in merged)


def classify_los_proxy(tti: Optional[float]) -> Tuple[str, str]:
    """
    Map Travel Time Index (TTI) to project-specific LOS Proxy and Congestion Category.
    NOTE: This is a project TTI-based congestion proxy, NOT facility-specific HCM LOS.
    """
    if tti is None or tti <= 0.0:
        return "UNKNOWN", "INSUFFICIENT_DATA"

    if tti <= 1.10:
        return "A", "FREE_FLOW"
    elif tti <= 1.25:
        return "B", "LIGHT"
    elif tti <= 1.50:
        return "C", "MODERATE"
    elif tti <= 2.00:
        return "D", "HEAVY"
    elif tti <= 2.50:
        return "E", "SEVERE"
    else:
        return "F", "BREAKDOWN"


@dataclass
class CameraNodeFlowMetrics:
    """
    Traffic throughput and temporal occupancy measured at a specific camera observation node.
    """
    camera_id: str
    camera_name: str
    window_duration_seconds: float
    unique_vehicles_observed: int
    camera_flow_rate_veh_hr: float
    camera_flow_rate_veh_min: float
    average_dwell_time_s: float
    estimated_temporal_occupancy_ratio: float  # In [0.0, 1.0], computed via interval union
    vehicle_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "window_duration_seconds": round(self.window_duration_seconds, 2),
            "unique_vehicles_observed": self.unique_vehicles_observed,
            "camera_flow_rate_veh_hr": round(self.camera_flow_rate_veh_hr, 2),
            "camera_flow_rate_veh_min": round(self.camera_flow_rate_veh_min, 2),
            "average_dwell_time_s": round(self.average_dwell_time_s, 2),
            "estimated_temporal_occupancy_ratio": round(self.estimated_temporal_occupancy_ratio, 4),
            "vehicle_types": dict(self.vehicle_type_counts),
        }


@dataclass
class CorridorCongestionMetrics:
    """
    Traffic flow, density estimation, and congestion modeling for a directed corridor.
    """
    from_camera: str
    to_camera: str
    observation_count: int                    # Total observed transitions (N)
    valid_observation_count: int              # Non-anomalous transitions
    anomalous_observation_count: int          # Transitions with >= 1 anomaly
    corridor_transit_rate_veh_hr: float       # Observed corridor transit throughput rate
    corridor_transit_rate_veh_min: float
    network_distance_km: Optional[float]
    haversine_distance_km: Optional[float]

    # Baseline & Speeds
    free_flow_speed_kmh: float                # Corridor baseline speed
    free_flow_speed_source: str               # "CORRIDOR_CONFIG", "DEFAULT_ASSUMPTION", or "EMPIRICAL_BASELINE"
    free_flow_travel_time_s: Optional[float]  # Distance / free_flow_speed * 3600
    observed_median_speed_kmh: Optional[float]
    observed_p05_speed_kmh: Optional[float]   # Low-speed tail condition
    observed_median_travel_time_s: Optional[float]
    observed_p95_travel_time_s: Optional[float] # Tail delay condition

    # Density & Kinematic Indices
    estimated_density_veh_km: Optional[float] # Fundamental traffic relation k = q / v
    travel_time_index: Optional[float]        # TTI = t_median / t_free_flow
    speed_performance_index: Optional[float]  # SPI = v_median / v_free_flow * 100
    speed_degradation_pct: Optional[float]    # (1.0 - v_median / v_free_flow) * 100
    travel_time_increase_pct: Optional[float] # (TTI - 1.0) * 100

    # Project Classification & Confidence
    los_proxy: str                            # "A" through "F", project-level TTI proxy (NOT HCM LOS)
    congestion_category: str                  # "FREE_FLOW", "LIGHT", "MODERATE", "HEAVY", "BREAKDOWN", "INSUFFICIENT_DATA"
    sample_confidence_score: float            # min(1.0, N / 10), sample-size indicator
    vehicle_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corridor": f"{self.from_camera} -> {self.to_camera}",
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "observation_count": self.observation_count,
            "valid_observation_count": self.valid_observation_count,
            "anomalous_observation_count": self.anomalous_observation_count,
            "flow_rates": {
                "corridor_transit_rate_veh_hr": round(self.corridor_transit_rate_veh_hr, 2),
                "corridor_transit_rate_veh_min": round(self.corridor_transit_rate_veh_min, 2),
            },
            "distances_km": {
                "network_distance": round(self.network_distance_km, 3) if self.network_distance_km is not None else None,
                "haversine_distance": round(self.haversine_distance_km, 3) if self.haversine_distance_km is not None else None,
            },
            "baseline": {
                "free_flow_speed_kmh": round(self.free_flow_speed_kmh, 2),
                "free_flow_speed_source": self.free_flow_speed_source,
                "free_flow_travel_time_s": round(self.free_flow_travel_time_s, 2) if self.free_flow_travel_time_s is not None else None,
            },
            "kinematics": {
                "observed_median_speed_kmh": round(self.observed_median_speed_kmh, 2) if self.observed_median_speed_kmh is not None else None,
                "observed_p05_speed_kmh": round(self.observed_p05_speed_kmh, 2) if self.observed_p05_speed_kmh is not None else None,
                "observed_median_travel_time_s": round(self.observed_median_travel_time_s, 2) if self.observed_median_travel_time_s is not None else None,
                "observed_p95_travel_time_s": round(self.observed_p95_travel_time_s, 2) if self.observed_p95_travel_time_s is not None else None,
            },
            "congestion": {
                "estimated_density_veh_km": round(self.estimated_density_veh_km, 2) if self.estimated_density_veh_km is not None else None,
                "travel_time_index": round(self.travel_time_index, 3) if self.travel_time_index is not None else None,
                "speed_performance_index": round(self.speed_performance_index, 2) if self.speed_performance_index is not None else None,
                "speed_degradation_pct": round(self.speed_degradation_pct, 2) if self.speed_degradation_pct is not None else None,
                "travel_time_increase_pct": round(self.travel_time_increase_pct, 2) if self.travel_time_increase_pct is not None else None,
                "los_proxy": self.los_proxy,
                "congestion_category": self.congestion_category,
                "sample_confidence_score": round(self.sample_confidence_score, 2),
            },
            "vehicle_types": dict(self.vehicle_type_counts),
        }

    def summary(self) -> str:
        tti_str = f"{self.travel_time_index:.2f}" if self.travel_time_index is not None else "N/A"
        deg_str = f"{self.speed_degradation_pct:.1f}%" if self.speed_degradation_pct is not None else "N/A"
        dens_str = f"{self.estimated_density_veh_km:.1f} veh/km" if self.estimated_density_veh_km is not None else "N/A"
        return (
            f"Corridor {self.from_camera} -> {self.to_camera} | Congestion: {self.congestion_category} (LOS Proxy: {self.los_proxy})\n"
            f"  Throughput:  {self.corridor_transit_rate_veh_hr:.1f} veh/hr | Est Density: {dens_str}\n"
            f"  Baselines:   FF Speed={self.free_flow_speed_kmh:.1f} km/h ({self.free_flow_speed_source}) | Observed Med={self.observed_median_speed_kmh:.1f} km/h\n"
            f"  Indices:     TTI={tti_str} (+{self.travel_time_increase_pct:.1f}% delay) | SPI={self.speed_performance_index:.1f}% (-{deg_str} speed)\n"
            f"  Confidence:  Sample Score={self.sample_confidence_score:.2f} (N={self.observation_count})"
        )


@dataclass
class NetworkCongestionReport:
    """
    Network-wide traffic flow, density, and congestion evaluation across all corridors and camera nodes.
    """
    analysis_window_seconds: float
    total_vehicles_observed: int
    total_transit_observations: int
    corridor_metrics: Dict[Tuple[str, str], CorridorCongestionMetrics] = field(default_factory=dict)
    camera_node_metrics: Dict[str, CameraNodeFlowMetrics] = field(default_factory=dict)
    network_average_tti: Optional[float] = None
    hotspots: List[Dict[str, Any]] = field(default_factory=list)

    def get_corridor(self, from_cam: str, to_cam: str) -> Optional[CorridorCongestionMetrics]:
        return self.corridor_metrics.get((from_cam, to_cam))

    def get_camera(self, camera_id: str) -> Optional[CameraNodeFlowMetrics]:
        return self.camera_node_metrics.get(camera_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "analysis_window_seconds": round(self.analysis_window_seconds, 2),
                "total_vehicles_observed": self.total_vehicles_observed,
                "total_transit_observations": self.total_transit_observations,
                "network_average_tti": round(self.network_average_tti, 3) if self.network_average_tti is not None else None,
                "active_corridors": len(self.corridor_metrics),
                "active_cameras": len(self.camera_node_metrics),
                "hotspot_count": len(self.hotspots),
            },
            "corridors": {
                f"{k[0]}->{k[1]}": v.to_dict() for k, v in self.corridor_metrics.items()
            },
            "camera_nodes": {
                k: v.to_dict() for k, v in self.camera_node_metrics.items()
            },
            "hotspots": list(self.hotspots),
        }

    def summary(self) -> str:
        tti_str = f"{self.network_average_tti:.2f}" if self.network_average_tti is not None else "N/A"
        lines = [
            "=" * 70,
            "  NETWORK CONGESTION & TRAFFIC FLOW REPORT",
            "=" * 70,
            f"Analysis Duration:    {self.analysis_window_seconds:.1f} seconds",
            f"Vehicles Observed:    {self.total_vehicles_observed}",
            f"Transit Observations: {self.total_transit_observations}",
            f"Network Average TTI:  {tti_str}",
            f"Active Corridors:     {len(self.corridor_metrics)}",
            "",
            "--- CONGESTION HOTSPOTS (ORDERED BY SEVERITY) ---",
        ]
        if not self.hotspots:
            lines.append("  (No congested corridors detected; all corridors free-flowing or insufficient samples)")
        else:
            for i, h in enumerate(self.hotspots, 1):
                lines.append(
                    f"[{i}] {h['corridor']}: LOS Proxy {h['los_proxy']} ({h['congestion_category']}) | "
                    f"TTI={h['tti']:.2f} | Speed Degradation={h['speed_degradation_pct']:.1f}% | "
                    f"Flow={h['transit_rate_veh_hr']:.1f} veh/hr | N={h['sample_size']}"
                )

        lines.append("")
        lines.append("--- CAMERA NODE THROUGHPUT ---")
        for cid, cm in sorted(self.camera_node_metrics.items()):
            lines.append(
                f"• {cid} ({cm.camera_name}): {cm.camera_flow_rate_veh_hr:.1f} veh/hr ({cm.unique_vehicles_observed} vehs, "
                f"Occ: {cm.estimated_temporal_occupancy_ratio * 100:.1f}%)"
            )

        return "\n".join(lines)


class TrafficCongestionEngine:
    """
    Engine evaluating traffic flow, density estimation, baseline degradation,
    and project Level of Service (LOS) proxies across camera nodes and road corridors.
    """

    def __init__(
        self,
        default_free_flow_speed_kmh: float = 50.0,
        corridor_free_flow_speeds: Optional[Dict[Tuple[str, str], float]] = None,
        cameras_path: Union[str, Path] = "configs/cameras.json",
    ):
        self.default_free_flow_speed_kmh = float(default_free_flow_speed_kmh)
        self.corridor_free_flow_speeds = corridor_free_flow_speeds or {}
        self.cameras_path = Path(cameras_path)

        self.cameras_meta: Dict[str, dict] = {}
        self._load_cameras()

    def _load_cameras(self) -> None:
        if not self.cameras_path.exists():
            return
        try:
            with open(self.cameras_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cam in data:
                cid = cam.get("camera_id")
                if cid:
                    self.cameras_meta[cid] = cam
        except Exception as e:
            logger.error("Failed to load cameras in congestion engine: %s", e)

    def get_free_flow_speed(self, from_cam: str, to_cam: str) -> Tuple[float, str]:
        """
        Determine free-flow baseline speed and document its source.
        Returns: (speed_kmh, source_string)
        """
        pair = (from_cam, to_cam)
        if pair in self.corridor_free_flow_speeds:
            val = self.corridor_free_flow_speeds[pair]
            if val > 0:
                return val, "CORRIDOR_CONFIG"

        # Fallback to default configured speed
        if self.default_free_flow_speed_kmh > 0:
            return self.default_free_flow_speed_kmh, "DEFAULT_ASSUMPTION"

        return 0.0, "INVALID_BASELINE"

    def analyze_corridor(
        self,
        corridor_stats: CorridorAnalytics,
        window_duration_seconds: float,
    ) -> CorridorCongestionMetrics:
        """
        Analyze a CorridorAnalytics record to produce detailed congestion and degradation metrics.
        """
        from_cam = corridor_stats.from_camera
        to_cam = corridor_stats.to_camera
        n_obs = corridor_stats.observation_count
        dur_s = max(1.0, window_duration_seconds)
        dur_hr = dur_s / 3600.0

        # Flow rates
        flow_hr = n_obs / dur_hr
        flow_min = flow_hr / 60.0

        # Free-flow baseline
        v_ff, ff_source = self.get_free_flow_speed(from_cam, to_cam)
        net_dist = corridor_stats.network_distance_km

        t_ff = None
        if v_ff > 0 and net_dist is not None and net_dist > 0:
            t_ff = (net_dist / v_ff) * 3600.0  # seconds

        v_med = corridor_stats.speed_median_kmh
        t_med = corridor_stats.travel_time_median_s

        # Fundamental density estimation: k = q / v (veh/km)
        # q = transit rate (veh/hr), v = median speed (km/h)
        density_k = None
        if v_med is not None and v_med > 0:
            density_k = flow_hr / v_med

        # Indices and degradation
        tti = None
        spi = None
        spd_deg_pct = None
        tt_inc_pct = None

        if v_ff > 0:
            if v_med is not None and v_med > 0:
                spi = (v_med / v_ff) * 100.0
                spd_deg_pct = max(0.0, (1.0 - (v_med / v_ff)) * 100.0)

            if t_ff is not None and t_ff > 0 and t_med is not None and t_med > 0:
                tti = t_med / t_ff
                tt_inc_pct = max(0.0, (tti - 1.0) * 100.0)

        # Classification & Confidence
        sample_conf = min(1.0, n_obs / 10.0)
        los, cong_cat = classify_los_proxy(tti)

        # Small sample penalty: if N < 3, downgrade confidence to indicative
        if n_obs < 3:
            cong_cat = "INSUFFICIENT_DATA" if n_obs == 0 else f"{cong_cat}_LOW_SAMPLE"

        return CorridorCongestionMetrics(
            from_camera=from_cam,
            to_camera=to_cam,
            observation_count=n_obs,
            valid_observation_count=corridor_stats.valid_observation_count,
            anomalous_observation_count=corridor_stats.anomalous_observation_count,
            corridor_transit_rate_veh_hr=flow_hr,
            corridor_transit_rate_veh_min=flow_min,
            network_distance_km=net_dist,
            haversine_distance_km=corridor_stats.haversine_distance_km,
            free_flow_speed_kmh=v_ff,
            free_flow_speed_source=ff_source,
            free_flow_travel_time_s=t_ff,
            observed_median_speed_kmh=v_med,
            observed_p05_speed_kmh=corridor_stats.speed_p05_kmh,
            observed_median_travel_time_s=t_med,
            observed_p95_travel_time_s=corridor_stats.travel_time_p95_s,
            estimated_density_veh_km=density_k,
            travel_time_index=tti,
            speed_performance_index=spi,
            speed_degradation_pct=spd_deg_pct,
            travel_time_increase_pct=tt_inc_pct,
            los_proxy=los,
            congestion_category=cong_cat,
            sample_confidence_score=sample_conf,
            vehicle_type_counts=dict(corridor_stats.vehicle_type_counts),
        )

    def analyze(
        self,
        trajectories: List[VehicleTrajectory],
        window_duration_seconds: Optional[float] = None,
    ) -> NetworkCongestionReport:
        """
        Analyze trajectories to produce a comprehensive network congestion and flow report.
        """
        # Determine observation time window
        min_ts = float("inf")
        max_ts = float("-inf")

        # Track intervals per camera node: camera_id -> list of (first_ts, last_ts, vtype)
        camera_nodes_tracks: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)

        for traj in trajectories:
            vtype = traj.vehicle_type or "car"
            for node in traj.nodes:
                cid = node.camera_id
                camera_nodes_tracks[cid].append((node.first_timestamp, node.last_timestamp, vtype))
                if node.first_timestamp < min_ts:
                    min_ts = node.first_timestamp
                if node.last_timestamp > max_ts:
                    max_ts = node.last_timestamp

        if min_ts == float("inf") or max_ts == float("-inf"):
            window_dur = 3600.0
            min_ts = 0.0
            max_ts = window_dur
        else:
            measured_dur = max(1.0, max_ts - min_ts)
            window_dur = float(window_duration_seconds) if window_duration_seconds is not None else measured_dur

        # 1. Evaluate Camera Node Throughput & Occupancy
        camera_metrics: Dict[str, CameraNodeFlowMetrics] = {}
        dur_hr = window_dur / 3600.0

        for cid, tracks in camera_nodes_tracks.items():
            cam_name = self.cameras_meta.get(cid, {}).get("name", f"Camera {cid}")
            n_vehs = len(tracks)
            flow_hr = n_vehs / dur_hr
            flow_min = flow_hr / 60.0

            dwell_times = [max(0.0, e - s) for s, e, _ in tracks]
            avg_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0

            # Mathematical interval union for temporal occupancy ratio in [0.0, 1.0]
            raw_intervals = [(s, e) for s, e, _ in tracks]
            occupied_dur = compute_interval_union_duration(raw_intervals, min_ts, min_ts + window_dur)
            occ_ratio = min(1.0, max(0.0, occupied_dur / window_dur))

            vtype_map = defaultdict(int)
            for _, _, vt in tracks:
                vtype_map[vt] += 1

            camera_metrics[cid] = CameraNodeFlowMetrics(
                camera_id=cid,
                camera_name=cam_name,
                window_duration_seconds=window_dur,
                unique_vehicles_observed=n_vehs,
                camera_flow_rate_veh_hr=flow_hr,
                camera_flow_rate_veh_min=flow_min,
                average_dwell_time_s=avg_dwell,
                estimated_temporal_occupancy_ratio=occ_ratio,
                vehicle_type_counts=dict(vtype_map),
            )

        # 2. Evaluate Corridors via Phase 7B Corridor Analytics
        analytics_engine = get_analytics_engine()
        base_report = analytics_engine.analyze_trajectories(trajectories)

        corridor_congestion: Dict[Tuple[str, str], CorridorCongestionMetrics] = {}
        hotspots: List[Dict[str, Any]] = []
        valid_ttis = []

        for pair, c_stat in base_report.corridors.items():
            cm = self.analyze_corridor(c_stat, window_dur)
            corridor_congestion[pair] = cm

            if cm.travel_time_index is not None:
                valid_ttis.append(cm.travel_time_index)

                # Hotspot criteria: TTI >= 1.25 (LOS C or worse) and N >= 2
                if cm.travel_time_index >= 1.25 and cm.observation_count >= 2:
                    hotspots.append({
                        "corridor": f"{pair[0]} -> {pair[1]}",
                        "from_camera": pair[0],
                        "to_camera": pair[1],
                        "tti": round(cm.travel_time_index, 3),
                        "los_proxy": cm.los_proxy,
                        "congestion_category": cm.congestion_category,
                        "speed_degradation_pct": round(cm.speed_degradation_pct, 1) if cm.speed_degradation_pct is not None else 0.0,
                        "observed_median_speed_kmh": round(cm.observed_median_speed_kmh, 1) if cm.observed_median_speed_kmh is not None else 0.0,
                        "free_flow_speed_kmh": round(cm.free_flow_speed_kmh, 1),
                        "transit_rate_veh_hr": round(cm.corridor_transit_rate_veh_hr, 1),
                        "sample_size": cm.observation_count,
                        "confidence": round(cm.sample_confidence_score, 2),
                    })

        # Sort hotspots descending by TTI (worst congestion first)
        hotspots.sort(key=lambda h: h["tti"], reverse=True)

        avg_tti = float(np.mean(valid_ttis)) if valid_ttis else None

        return NetworkCongestionReport(
            analysis_window_seconds=window_dur,
            total_vehicles_observed=sum(m.unique_vehicles_observed for m in camera_metrics.values()),
            total_transit_observations=base_report.total_transit_observations,
            corridor_metrics=corridor_congestion,
            camera_node_metrics=camera_metrics,
            network_average_tti=avg_tti,
            hotspots=hotspots,
        )

    def analyze_db(
        self,
        conn: sqlite3.Connection,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        reconstructor: Optional[TrajectoryReconstructor] = None,
    ) -> NetworkCongestionReport:
        """
        Reconstruct and analyze traffic congestion directly from a SQLite database.
        """
        recon = reconstructor or get_reconstructor()
        trajectories = recon.list_all_trajectories(conn, limit=10000)

        if start_ts is not None or end_ts is not None:
            filtered = []
            for t in trajectories:
                if start_ts is not None and t.last_seen_ts < start_ts:
                    continue
                if end_ts is not None and t.first_seen_ts > end_ts:
                    continue
                filtered.append(t)
            trajectories = filtered

        win_dur = None
        if start_ts is not None and end_ts is not None and end_ts > start_ts:
            win_dur = end_ts - start_ts

        return self.analyze(trajectories, window_duration_seconds=win_dur)


# Convenience singleton function
_default_congestion_engine: Optional[TrafficCongestionEngine] = None


def get_congestion_engine() -> TrafficCongestionEngine:
    global _default_congestion_engine
    if _default_congestion_engine is None:
        _default_congestion_engine = TrafficCongestionEngine()
    return _default_congestion_engine


def analyze_traffic_congestion(
    trajectories: List[VehicleTrajectory],
    window_duration_seconds: Optional[float] = None,
) -> NetworkCongestionReport:
    """Analyze traffic congestion and flow rates with default engine."""
    return get_congestion_engine().analyze(trajectories, window_duration_seconds=window_duration_seconds)
