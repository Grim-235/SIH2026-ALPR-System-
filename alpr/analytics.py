"""
Network-Level Speed and Travel-Time Analytics Module (Phase 7B).

Aggregates inter-camera transit segments from reconstructed vehicle trajectories
into statistical corridor metrics, time-window profiles, and origin-destination (OD) flows.

Key features:
- Segment extraction: evaluates genuine inter-camera transits, excludes same-camera observations.
- Sample transparency: every corridor reports observation_count (N), valid_observation_count,
  and anomalous_observation_count.
- Robust statistics: calculates median (primary), mean, P95, min, max, std for travel times.
- Dual tail speed metrics:
    * speed_p05_kmh: low-speed congestion tail condition.
    * speed_median_kmh: robust central velocity.
    * speed_mean_kmh: arithmetic average.
    * speed_p95_kmh: descriptive high-speed tail condition.
- Anomaly rate bounded in [0.0, 1.0]: defined as anomalous_observations / N.
- Time-of-day bucketing based on departure timestamp (segment.from_timestamp).
- Trip-level Origin-Destination (OD) matrix: maps first camera -> last camera per trajectory.
- Fully compatible with Python dataclasses, dict serialization, and SQLite data sources.
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from alpr.trajectory import VehicleTrajectory, TrajectorySegment, TrajectoryReconstructor, get_reconstructor

logger = logging.getLogger("alpr.analytics")

# Standard 3-hour time-of-day buckets (start_hour inclusive, end_hour exclusive, label)
DEFAULT_TIME_WINDOWS = [
    (0, 6, "00:00-06:00"),
    (6, 9, "06:00-09:00"),
    (9, 12, "09:00-12:00"),
    (12, 15, "12:00-15:00"),
    (15, 18, "15:00-18:00"),
    (18, 21, "18:00-21:00"),
    (21, 24, "21:00-00:00"),
]


def get_time_window_label(
    timestamp: float,
    windows: Optional[List[Tuple[int, int, str]]] = None,
    tz_offset_hours: float = 0.0,
) -> str:
    """
    Map an epoch timestamp to a time-of-day window bucket based on departure hour.
    """
    if windows is None:
        windows = DEFAULT_TIME_WINDOWS

    # Extract hour of the day
    hour = int((timestamp / 3600.0 + tz_offset_hours) % 24.0)
    for start, end, label in windows:
        if start <= hour < end:
            return label
    return "00:00-06:00"


def compute_distribution_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Compute distribution statistics (mean, median, p05, p95, min, max, std)
    with safe handling of N=0, N=1, and N>=2.
    """
    if not values:
        return {
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
            "std": None,
        }

    arr = np.array(values, dtype=np.float64)
    n = len(arr)

    if n == 1:
        v = float(arr[0])
        return {
            "mean": v,
            "median": v,
            "p05": v,
            "p95": v,
            "min": v,
            "max": v,
            "std": 0.0,
        }

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


@dataclass
class CorridorAnalytics:
    """
    Statistical performance profile for a specific directed corridor (CAM_A -> CAM_B).
    """
    from_camera: str
    to_camera: str
    observation_count: int                    # Total transit observations (N)
    valid_observation_count: int              # Observations with zero anomalies
    anomalous_observation_count: int          # Observations with >= 1 anomaly
    network_distance_km: Optional[float]      # Road network shortest path distance
    haversine_distance_km: Optional[float]    # Straight-line air distance reference

    # Travel Time Metrics (seconds)
    travel_time_mean_s: Optional[float] = None
    travel_time_median_s: Optional[float] = None
    travel_time_p95_s: Optional[float] = None
    travel_time_min_s: Optional[float] = None
    travel_time_max_s: Optional[float] = None
    travel_time_std_s: Optional[float] = None

    # Speed Metrics (km/h)
    speed_mean_kmh: Optional[float] = None
    speed_median_kmh: Optional[float] = None
    speed_p05_kmh: Optional[float] = None     # Low-speed tail (congestion indicator)
    speed_p95_kmh: Optional[float] = None     # High-speed descriptive tail
    speed_min_kmh: Optional[float] = None
    speed_max_kmh: Optional[float] = None
    speed_std_kmh: Optional[float] = None

    # Vehicle-type breakdown
    vehicle_type_counts: Dict[str, int] = field(default_factory=dict)

    # Anomaly Quantifications
    velocity_anomaly_count: int = 0
    temporal_anomaly_count: int = 0
    unreachable_network_count: int = 0
    anomaly_rate: float = 0.0                 # anomalous_observation_count / observation_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corridor": f"{self.from_camera} -> {self.to_camera}",
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "observation_count": self.observation_count,
            "valid_observation_count": self.valid_observation_count,
            "anomalous_observation_count": self.anomalous_observation_count,
            "network_distance_km": round(self.network_distance_km, 3) if self.network_distance_km is not None else None,
            "haversine_distance_km": round(self.haversine_distance_km, 3) if self.haversine_distance_km is not None else None,
            "travel_time_s": {
                "mean": round(self.travel_time_mean_s, 2) if self.travel_time_mean_s is not None else None,
                "median": round(self.travel_time_median_s, 2) if self.travel_time_median_s is not None else None,
                "p95": round(self.travel_time_p95_s, 2) if self.travel_time_p95_s is not None else None,
                "min": round(self.travel_time_min_s, 2) if self.travel_time_min_s is not None else None,
                "max": round(self.travel_time_max_s, 2) if self.travel_time_max_s is not None else None,
                "std": round(self.travel_time_std_s, 2) if self.travel_time_std_s is not None else None,
            },
            "speed_kmh": {
                "mean": round(self.speed_mean_kmh, 2) if self.speed_mean_kmh is not None else None,
                "median": round(self.speed_median_kmh, 2) if self.speed_median_kmh is not None else None,
                "p05": round(self.speed_p05_kmh, 2) if self.speed_p05_kmh is not None else None,
                "p95": round(self.speed_p95_kmh, 2) if self.speed_p95_kmh is not None else None,
                "min": round(self.speed_min_kmh, 2) if self.speed_min_kmh is not None else None,
                "max": round(self.speed_max_kmh, 2) if self.speed_max_kmh is not None else None,
                "std": round(self.speed_std_kmh, 2) if self.speed_std_kmh is not None else None,
            },
            "vehicle_types": dict(self.vehicle_type_counts),
            "anomalies": {
                "velocity_anomaly_count": self.velocity_anomaly_count,
                "temporal_anomaly_count": self.temporal_anomaly_count,
                "unreachable_network_count": self.unreachable_network_count,
                "anomaly_rate": round(self.anomaly_rate, 4),
            },
        }

    def summary(self) -> str:
        tt_med = f"{self.travel_time_median_s:.1f}s" if self.travel_time_median_s is not None else "N/A"
        tt_p95 = f"{self.travel_time_p95_s:.1f}s" if self.travel_time_p95_s is not None else "N/A"
        spd_med = f"{self.speed_median_kmh:.1f} km/h" if self.speed_median_kmh is not None else "N/A"
        spd_p05 = f"{self.speed_p05_kmh:.1f} km/h" if self.speed_p05_kmh is not None else "N/A"
        dist_str = f"{self.network_distance_km:.2f} km" if self.network_distance_km is not None else "N/A"

        return (
            f"Corridor {self.from_camera} -> {self.to_camera} (Road Dist: {dist_str})\n"
            f"  Observations: N={self.observation_count} (Valid: {self.valid_observation_count}, Anomalous: {self.anomalous_observation_count})\n"
            f"  Travel Time:  Median={tt_med}, Mean={self.travel_time_mean_s:.1f}s, P95={tt_p95}\n"
            f"  Speed:        Median={spd_med}, Mean={self.speed_mean_kmh:.1f} km/h, P05={spd_p05}, P95={self.speed_p95_kmh:.1f} km/h\n"
            f"  Anomalies:    Rate={self.anomaly_rate * 100:.1f}% (Velocity: {self.velocity_anomaly_count}, Temporal: {self.temporal_anomaly_count})\n"
            f"  Fleet Mix:    {dict(self.vehicle_type_counts)}"
        )


@dataclass
class TripODRecord:
    """
    Aggregated trip flow record for a complete origin -> destination trajectory.
    """
    origin_camera: str
    destination_camera: str
    trip_count: int
    median_duration_s: Optional[float] = None
    median_distance_km: Optional[float] = None
    vehicle_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin_camera,
            "destination": self.destination_camera,
            "trip_count": self.trip_count,
            "median_duration_s": round(self.median_duration_s, 2) if self.median_duration_s is not None else None,
            "median_distance_km": round(self.median_distance_km, 3) if self.median_distance_km is not None else None,
            "vehicle_types": dict(self.vehicle_type_counts),
        }


@dataclass
class NetworkAnalyticsReport:
    """
    Comprehensive city-wide traffic speed, corridor, time-window, and OD analytics report.
    """
    total_trajectories_analyzed: int
    total_transit_observations: int
    total_valid_observations: int
    total_anomalous_observations: int
    overall_anomaly_rate: float
    corridors: Dict[Tuple[str, str], CorridorAnalytics] = field(default_factory=dict)
    time_windows: Dict[str, Dict[Tuple[str, str], CorridorAnalytics]] = field(default_factory=dict)
    od_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    od_details: Dict[Tuple[str, str], TripODRecord] = field(default_factory=dict)

    def get_corridor(self, from_cam: str, to_cam: str) -> Optional[CorridorAnalytics]:
        """Get analytics for a specific directed corridor."""
        return self.corridors.get((from_cam, to_cam))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "totals": {
                "trajectories_analyzed": self.total_trajectories_analyzed,
                "transit_observations": self.total_transit_observations,
                "valid_observations": self.total_valid_observations,
                "anomalous_observations": self.total_anomalous_observations,
                "overall_anomaly_rate": round(self.overall_anomaly_rate, 4),
                "active_corridors_count": len(self.corridors),
            },
            "corridors": {
                f"{k[0]}->{k[1]}": v.to_dict() for k, v in self.corridors.items()
            },
            "od_matrix": {k: dict(v) for k, v in self.od_matrix.items()},
            "od_details": {
                f"{k[0]}->{k[1]}": v.to_dict() for k, v in self.od_details.items()
            },
            "time_windows": {
                win: {f"{k[0]}->{k[1]}": v.to_dict() for k, v in corrs.items()}
                for win, corrs in self.time_windows.items()
            },
        }

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  NETWORK SPEED & TRAVEL-TIME ANALYTICS REPORT",
            "=" * 70,
            f"Trajectories Analyzed: {self.total_trajectories_analyzed}",
            f"Transit Observations:  N={self.total_transit_observations} (Valid: {self.total_valid_observations}, Anomalous: {self.total_anomalous_observations})",
            f"Overall Anomaly Rate:  {self.overall_anomaly_rate * 100:.2f}%",
            f"Active Road Corridors: {len(self.corridors)}",
            "",
            "--- CORRIDOR PERFORMANCE SUMMARY ---",
        ]
        if not self.corridors:
            lines.append("  (No inter-camera transit observations recorded)")
        else:
            for (c_from, c_to), stat in sorted(self.corridors.items()):
                lines.append(f"• {c_from} -> {c_to}: N={stat.observation_count}, Median TT={stat.travel_time_median_s:.1f}s, P95 TT={stat.travel_time_p95_s:.1f}s, Median Speed={stat.speed_median_kmh:.1f} km/h, P05 Speed={stat.speed_p05_kmh:.1f} km/h")

        lines.append("")
        lines.append("--- ORIGIN-DESTINATION (OD) SUMMARY ---")
        if not self.od_matrix:
            lines.append("  (No multi-camera OD trips recorded)")
        else:
            for orig in sorted(self.od_matrix.keys()):
                for dest in sorted(self.od_matrix[orig].keys()):
                    cnt = self.od_matrix[orig][dest]
                    lines.append(f"• {orig} -> {dest}: {cnt} complete trips")

        return "\n".join(lines)


class CorridorAnalyticsEngine:
    """
    Engine that aggregates trajectory transit segments into corridor statistics,
    time-window profiles, and origin-destination matrices.
    """

    def __init__(
        self,
        time_windows: Optional[List[Tuple[int, int, str]]] = None,
        tz_offset_hours: float = 0.0,
    ):
        self.time_windows = time_windows or DEFAULT_TIME_WINDOWS
        self.tz_offset_hours = tz_offset_hours

    def _aggregate_segments(
        self,
        raw_segments: List[Tuple[TrajectorySegment, str]],
    ) -> CorridorAnalytics:
        """
        Compute CorridorAnalytics from a collection of (segment, vehicle_type) tuples.
        """
        if not raw_segments:
            return CorridorAnalytics(
                from_camera="UNKNOWN",
                to_camera="UNKNOWN",
                observation_count=0,
                valid_observation_count=0,
                anomalous_observation_count=0,
                network_distance_km=None,
                haversine_distance_km=None,
            )

        from_cam = raw_segments[0][0].from_camera_id
        to_cam = raw_segments[0][0].to_camera_id
        net_dist = raw_segments[0][0].network_distance_km
        hav_dist = raw_segments[0][0].haversine_distance_km

        n_total = len(raw_segments)
        valid_count = 0
        anomalous_count = 0

        vel_anom_cnt = 0
        temp_anom_cnt = 0
        unreach_cnt = 0

        travel_times: List[float] = []
        speeds: List[float] = []
        vtype_counts: Dict[str, int] = defaultdict(int)

        for seg, vtype in raw_segments:
            vtype_counts[vtype] += 1

            has_anomaly = False
            if seg.is_velocity_anomaly:
                vel_anom_cnt += 1
                has_anomaly = True
            if seg.is_temporal_anomaly:
                temp_anom_cnt += 1
                has_anomaly = True
            if seg.is_unreachable_network:
                unreach_cnt += 1
                has_anomaly = True

            if has_anomaly:
                anomalous_count += 1
            else:
                valid_count += 1

            # Include observations in statistical distribution (transparent populating)
            # Filter out negative travel times to prevent nonsensical math in percentiles
            if seg.transit_time_seconds > 0:
                travel_times.append(seg.transit_time_seconds)

            if seg.speed_kmh is not None and seg.speed_kmh > 0:
                speeds.append(seg.speed_kmh)

        tt_stats = compute_distribution_stats(travel_times)
        spd_stats = compute_distribution_stats(speeds)

        anom_rate = (anomalous_count / n_total) if n_total > 0 else 0.0

        return CorridorAnalytics(
            from_camera=from_cam,
            to_camera=to_cam,
            observation_count=n_total,
            valid_observation_count=valid_count,
            anomalous_observation_count=anomalous_count,
            network_distance_km=net_dist,
            haversine_distance_km=hav_dist,
            travel_time_mean_s=tt_stats["mean"],
            travel_time_median_s=tt_stats["median"],
            travel_time_p95_s=tt_stats["p95"],
            travel_time_min_s=tt_stats["min"],
            travel_time_max_s=tt_stats["max"],
            travel_time_std_s=tt_stats["std"],
            speed_mean_kmh=spd_stats["mean"],
            speed_median_kmh=spd_stats["median"],
            speed_p05_kmh=spd_stats["p05"],
            speed_p95_kmh=spd_stats["p95"],
            speed_min_kmh=spd_stats["min"],
            speed_max_kmh=spd_stats["max"],
            speed_std_kmh=spd_stats["std"],
            vehicle_type_counts=dict(vtype_counts),
            velocity_anomaly_count=vel_anom_cnt,
            temporal_anomaly_count=temp_anom_cnt,
            unreachable_network_count=unreach_cnt,
            anomaly_rate=anom_rate,
        )

    def analyze_trajectories(
        self,
        trajectories: List[VehicleTrajectory],
    ) -> NetworkAnalyticsReport:
        """
        Analyze a list of VehicleTrajectory instances.
        """
        total_trajs = len(trajectories)
        corridor_buckets: Dict[Tuple[str, str], List[Tuple[TrajectorySegment, str]]] = defaultdict(list)
        time_window_buckets: Dict[str, Dict[Tuple[str, str], List[Tuple[TrajectorySegment, str]]]] = defaultdict(lambda: defaultdict(list))

        # Origin-Destination mapping (first node -> last node)
        od_trip_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        od_trip_details: Dict[Tuple[str, str], List[Tuple[float, float, str]]] = defaultdict(list)

        total_obs = 0
        total_anom = 0
        total_valid = 0

        for traj in trajectories:
            vtype = traj.vehicle_type or "car"

            # 1. Process OD trip if multi-node trajectory
            if len(traj.nodes) >= 2:
                orig_cam = traj.nodes[0].camera_id
                dest_cam = traj.nodes[-1].camera_id
                if orig_cam != dest_cam:
                    od_trip_counts[orig_cam][dest_cam] += 1
                    od_trip_details[(orig_cam, dest_cam)].append(
                        (traj.total_duration_seconds, traj.total_network_distance_km, vtype)
                    )

            # 2. Process inter-camera segments
            for seg in traj.segments:
                # Rule: exclude same-camera observations from corridor transit statistics
                if seg.is_same_camera:
                    continue

                total_obs += 1
                has_anom = (seg.is_velocity_anomaly or seg.is_temporal_anomaly or seg.is_unreachable_network)
                if has_anom:
                    total_anom += 1
                else:
                    total_valid += 1

                pair = (seg.from_camera_id, seg.to_camera_id)
                corridor_buckets[pair].append((seg, vtype))

                # Time window assignment explicitly uses departure timestamp: seg.from_timestamp
                window_label = get_time_window_label(seg.from_timestamp, self.time_windows, self.tz_offset_hours)
                time_window_buckets[window_label][pair].append((seg, vtype))

        # Compute corridor statistics
        corridors: Dict[Tuple[str, str], CorridorAnalytics] = {}
        for pair, seg_list in corridor_buckets.items():
            corridors[pair] = self._aggregate_segments(seg_list)

        # Compute time window statistics
        time_windows_stats: Dict[str, Dict[Tuple[str, str], CorridorAnalytics]] = {}
        for win_label, pair_map in time_window_buckets.items():
            time_windows_stats[win_label] = {}
            for pair, seg_list in pair_map.items():
                time_windows_stats[win_label][pair] = self._aggregate_segments(seg_list)

        # Compute OD details
        od_details: Dict[Tuple[str, str], TripODRecord] = {}
        for (orig, dest), trip_list in od_trip_details.items():
            durations = [t[0] for t in trip_list if t[0] > 0]
            distances = [t[1] for t in trip_list]
            od_vtypes: Dict[str, int] = defaultdict(int)
            for t in trip_list:
                od_vtypes[t[2]] += 1

            med_dur = float(np.median(durations)) if durations else None
            med_dist = float(np.median(distances)) if distances else None

            od_details[(orig, dest)] = TripODRecord(
                origin_camera=orig,
                destination_camera=dest,
                trip_count=len(trip_list),
                median_duration_s=med_dur,
                median_distance_km=med_dist,
                vehicle_type_counts=dict(od_vtypes),
            )

        overall_anom_rate = (total_anom / total_obs) if total_obs > 0 else 0.0

        return NetworkAnalyticsReport(
            total_trajectories_analyzed=total_trajs,
            total_transit_observations=total_obs,
            total_valid_observations=total_valid,
            total_anomalous_observations=total_anom,
            overall_anomaly_rate=overall_anom_rate,
            corridors=corridors,
            time_windows=time_windows_stats,
            od_matrix=od_trip_counts,
            od_details=od_details,
        )

    def analyze_db(
        self,
        conn: sqlite3.Connection,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        reconstructor: Optional[TrajectoryReconstructor] = None,
    ) -> NetworkAnalyticsReport:
        """
        Reconstruct and analyze trajectories directly from a SQLite database.
        """
        recon = reconstructor or get_reconstructor()
        trajectories = recon.list_all_trajectories(conn, limit=10000)

        # Optional timestamp filtering
        if start_ts is not None or end_ts is not None:
            filtered = []
            for t in trajectories:
                if start_ts is not None and t.last_seen_ts < start_ts:
                    continue
                if end_ts is not None and t.first_seen_ts > end_ts:
                    continue
                filtered.append(t)
            trajectories = filtered

        return self.analyze_trajectories(trajectories)


# Convenience singleton function
_default_engine: Optional[CorridorAnalyticsEngine] = None


def get_analytics_engine() -> CorridorAnalyticsEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = CorridorAnalyticsEngine()
    return _default_engine


def analyze_network_traffic(
    trajectories: List[VehicleTrajectory],
) -> NetworkAnalyticsReport:
    """Analyze a list of reconstructed trajectories with default engine."""
    return get_analytics_engine().analyze_trajectories(trajectories)
