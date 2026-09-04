"""
Vehicle Trajectory Reconstruction Module (Phase 7A).

Reconstructs multi-camera spatiotemporal trajectories from persisted
global vehicle identities (global_vehicles) and observations (vehicle_observations).

Key features:
- Chronological observation ordering with camera geospatial enrichment.
- Exact transit time calculation: next.first_timestamp - prev.last_timestamp.
- Dual distance preservation:
    * network_distance_km: road shortest path via Floyd-Warshall camera graph.
    * haversine_distance_km: straight-line geographic reference.
- Kinematic metrics: transit speed v = network_dist / delta_t.
- Physical plausibility checks:
    * is_velocity_anomaly: flags transit speeds exceeding plausibility bound (default 140 km/h).
    * is_temporal_anomaly: flags zero or negative transit time intervals.
    * is_unreachable_network: flags transitions between disconnected graph cameras.
    * is_same_camera: cleanly handles consecutive observations at the same camera.
- Standardized exports: dataclass, dictionary, summary text, and GIS GeoJSON FeatureCollection.
"""

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from alpr.database import (
    get_global_vehicle,
    get_global_vehicle_by_plate,
    get_vehicle_trajectory,
    get_all_global_vehicles,
)

logger = logging.getLogger("alpr.trajectory")


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two geographic points in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


@dataclass
class TrajectoryNode:
    """
    A single observation node of a vehicle at a specific camera location.
    """
    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    first_timestamp: float
    last_timestamp: float
    duration_seconds: float
    local_track_id: int
    canonical_plate: Optional[str] = None
    plate_confidence: float = 0.0
    crop_quality: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    match_status: str = "MATCH"
    match_method: str = "new_identity"
    match_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "local_track_id": self.local_track_id,
            "canonical_plate": self.canonical_plate,
            "plate_confidence": round(self.plate_confidence, 4),
            "crop_quality": round(self.crop_quality, 2),
            "bbox": list(self.bbox) if self.bbox else None,
            "match_status": self.match_status,
            "match_method": self.match_method,
            "match_confidence": round(self.match_confidence, 4),
        }

    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert node to a GeoJSON Point Feature."""
        props = self.to_dict()
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.longitude, self.latitude],
            },
            "properties": {
                "feature_type": "sighting_node",
                **props,
            },
        }


@dataclass
class TrajectorySegment:
    """
    A directed transition segment between two consecutive camera sightings.
    """
    from_camera_id: str
    to_camera_id: str
    from_timestamp: float  # Exit timestamp at origin camera (prev_node.last_timestamp)
    to_timestamp: float    # Entry timestamp at destination camera (next_node.first_timestamp)
    transit_time_seconds: float  # next_node.first_timestamp - prev_node.last_timestamp
    network_distance_km: Optional[float] = None     # Road distance via camera graph
    haversine_distance_km: Optional[float] = None   # Straight-line distance reference
    speed_kmh: Optional[float] = None               # network_dist / transit_time (km/h)
    is_same_camera: bool = False                    # True if from_cam == to_cam
    is_velocity_anomaly: bool = False               # Exceeds configured physical bound
    is_temporal_anomaly: bool = False               # transit_time <= 0
    is_unreachable_network: bool = False            # No route found in camera graph
    anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_camera_id": self.from_camera_id,
            "to_camera_id": self.to_camera_id,
            "from_timestamp": self.from_timestamp,
            "to_timestamp": self.to_timestamp,
            "transit_time_seconds": round(self.transit_time_seconds, 2),
            "network_distance_km": round(self.network_distance_km, 3) if self.network_distance_km is not None else None,
            "haversine_distance_km": round(self.haversine_distance_km, 3) if self.haversine_distance_km is not None else None,
            "speed_kmh": round(self.speed_kmh, 2) if self.speed_kmh is not None else None,
            "is_same_camera": self.is_same_camera,
            "is_velocity_anomaly": self.is_velocity_anomaly,
            "is_temporal_anomaly": self.is_temporal_anomaly,
            "is_unreachable_network": self.is_unreachable_network,
            "anomalies": list(self.anomalies),
        }

    def to_geojson_feature(
        self, from_node: TrajectoryNode, to_node: TrajectoryNode
    ) -> Dict[str, Any]:
        """Convert segment to a GeoJSON LineString Feature."""
        props = self.to_dict()
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [from_node.longitude, from_node.latitude],
                    [to_node.longitude, to_node.latitude],
                ],
            },
            "properties": {
                "feature_type": "transit_segment",
                **props,
            },
        }


@dataclass
class VehicleTrajectory:
    """
    Complete reconstructed spatiotemporal trajectory for a global vehicle identity.
    """
    global_id: str
    canonical_plate: Optional[str] = None
    vehicle_type: str = "car"
    status: str = "active"
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    total_duration_seconds: float = 0.0
    total_network_distance_km: float = 0.0
    total_haversine_distance_km: float = 0.0
    average_speed_kmh: Optional[float] = None
    nodes: List[TrajectoryNode] = field(default_factory=list)
    segments: List[TrajectorySegment] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id": self.global_id,
            "canonical_plate": self.canonical_plate,
            "vehicle_type": self.vehicle_type,
            "status": self.status,
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "total_network_distance_km": round(self.total_network_distance_km, 3),
            "total_haversine_distance_km": round(self.total_haversine_distance_km, 3),
            "average_speed_kmh": round(self.average_speed_kmh, 2) if self.average_speed_kmh is not None else None,
            "sighting_count": len(self.nodes),
            "segment_count": len(self.segments),
            "anomalies": list(self.anomalies),
            "nodes": [n.to_dict() for n in self.nodes],
            "segments": [s.to_dict() for s in self.segments],
        }

    def to_geojson(self) -> Dict[str, Any]:
        """
        Generate a GIS-compliant GeoJSON FeatureCollection representing the vehicle trajectory.
        Contains Point features for camera sightings and LineString features for inter-camera transits.
        """
        features: List[Dict[str, Any]] = []

        # 1. Add Point features for all sighting nodes
        for node in self.nodes:
            features.append(node.to_geojson_feature())

        # 2. Add LineString features for all valid inter-camera segments
        for idx, seg in enumerate(self.segments):
            if idx < len(self.nodes) - 1:
                from_n = self.nodes[idx]
                to_n = self.nodes[idx + 1]
                # Avoid drawing zero-length LineStrings on same-camera consecutive sightings
                if not seg.is_same_camera:
                    features.append(seg.to_geojson_feature(from_n, to_n))

        return {
            "type": "FeatureCollection",
            "properties": {
                "global_id": self.global_id,
                "canonical_plate": self.canonical_plate,
                "vehicle_type": self.vehicle_type,
                "status": self.status,
                "total_network_distance_km": round(self.total_network_distance_km, 3),
                "total_duration_seconds": round(self.total_duration_seconds, 2),
                "average_speed_kmh": round(self.average_speed_kmh, 2) if self.average_speed_kmh is not None else None,
                "has_anomalies": len(self.anomalies) > 0,
            },
            "features": features,
        }

    def summary(self) -> str:
        """Return a formatted human-readable ASCII summary of the trajectory."""
        plate_str = self.canonical_plate or "Unknown Plate"
        speed_str = f"{self.average_speed_kmh:.1f} km/h" if self.average_speed_kmh is not None else "N/A"
        lines = [
            f"Vehicle Trajectory: {self.global_id} (Plate: {plate_str}, Type: {self.vehicle_type}, Status: {self.status})",
            f"  Duration: {self.total_duration_seconds:.1f}s | Network Dist: {self.total_network_distance_km:.2f} km | Haversine: {self.total_haversine_distance_km:.2f} km | Avg Speed: {speed_str}",
            f"  Sightings: {len(self.nodes)} | Segments: {len(self.segments)} | Anomalies: {len(self.anomalies)}",
        ]

        if self.anomalies:
            lines.append("  Anomalies Detected:")
            for a in self.anomalies:
                lines.append(f"    * {a}")

        lines.append("  Hops:")
        for i, node in enumerate(self.nodes):
            plate_info = f"{node.canonical_plate} ({node.plate_confidence:.2f})" if node.canonical_plate else "No Plate"
            lines.append(
                f"    [{i + 1}] {node.camera_id} ({node.camera_name}) @ {node.first_timestamp:.1f}s - {node.last_timestamp:.1f}s "
                f"({node.duration_seconds:.1f}s) [Track {node.local_track_id}, Plate: {plate_info}]"
            )
            if i < len(self.segments):
                seg = self.segments[i]
                if seg.is_same_camera:
                    lines.append(f"        --> Same Camera Re-appearance (interval: {seg.transit_time_seconds:.1f}s)")
                else:
                    net_str = f"{seg.network_distance_km:.2f} km" if seg.network_distance_km is not None else "Unreachable"
                    hav_str = f"{seg.haversine_distance_km:.2f} km" if seg.haversine_distance_km is not None else "N/A"
                    spd_str = f"{seg.speed_kmh:.1f} km/h" if seg.speed_kmh is not None else "N/A"
                    anom_str = f" [ANOMALY: {', '.join(seg.anomalies)}]" if seg.anomalies else ""
                    lines.append(
                        f"        --> Transit: {seg.transit_time_seconds:.1f}s | Road: {net_str} (Air: {hav_str}) | Speed: {spd_str}{anom_str}"
                    )

        return "\n".join(lines)


class TrajectoryReconstructor:
    """
    Analytical engine for reconstructing and validating multi-camera vehicle trajectories.
    """

    def __init__(
        self,
        cameras_path: Union[str, Path] = "configs/cameras.json",
        camera_graph_path: Union[str, Path] = "configs/camera_graph.json",
        max_plausible_speed_kmh: float = 140.0,
    ):
        self.cameras_path = Path(cameras_path)
        self.camera_graph_path = Path(camera_graph_path)
        self.max_plausible_speed_kmh = float(max_plausible_speed_kmh)

        # In-memory caches
        self.cameras: Dict[str, dict] = {}
        self.distances_km: Dict[str, Dict[str, float]] = {}

        self._load_cameras()
        self._load_camera_graph()

    def _load_cameras(self) -> None:
        """Load camera geographic coordinates and metadata from config."""
        if not self.cameras_path.exists():
            logger.warning("Cameras config not found at %s", self.cameras_path)
            return

        try:
            with open(self.cameras_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cam in data:
                cid = cam.get("camera_id")
                if cid:
                    self.cameras[cid] = cam
            logger.info("Loaded %d cameras from %s", len(self.cameras), self.cameras_path)
        except Exception as e:
            logger.error("Failed to load cameras config: %s", e)

    def _load_camera_graph(self) -> None:
        """Load road topology and compute all-pairs shortest paths using Floyd-Warshall."""
        if not self.camera_graph_path.exists():
            logger.warning("Camera graph config not found at %s", self.camera_graph_path)
            return

        try:
            with open(self.camera_graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            camera_ids = sorted(graph_data.keys())
            INF = float("inf")
            dist = {a: {b: INF for b in camera_ids} for a in camera_ids}

            for cid in camera_ids:
                dist[cid][cid] = 0.0
                neighbors = graph_data[cid].get("distances_km", {})
                for n_id, d in neighbors.items():
                    if n_id in dist[cid]:
                        dist[cid][n_id] = float(d)

            for k in camera_ids:
                for i in camera_ids:
                    for j in camera_ids:
                        if dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]

            self.distances_km = {}
            for i in camera_ids:
                self.distances_km[i] = {}
                for j in camera_ids:
                    if dist[i][j] < INF:
                        self.distances_km[i][j] = dist[i][j]

            logger.info("Computed camera graph network shortest paths for %d cameras", len(camera_ids))
        except Exception as e:
            logger.error("Failed to compute camera graph shortest paths: %s", e)

    def get_network_distance_km(self, cam_a: str, cam_b: str) -> Optional[float]:
        """Retrieve shortest network path distance between two cameras."""
        if cam_a == cam_b:
            return 0.0
        return self.distances_km.get(cam_a, {}).get(cam_b)

    def reconstruct_from_observations(
        self,
        global_id: str,
        observations: List[dict],
        global_vehicle: Optional[dict] = None,
    ) -> VehicleTrajectory:
        """
        Reconstruct a VehicleTrajectory from a list of observation dictionaries.
        Observations must contain at least: camera_id, first_timestamp, last_timestamp, local_track_id.
        """
        if not observations:
            return VehicleTrajectory(global_id=global_id)

        # 1. Sort observations strictly chronologically by first_timestamp
        sorted_obs = sorted(observations, key=lambda o: (float(o.get("first_timestamp", 0.0)), float(o.get("last_timestamp", 0.0))))

        nodes: List[TrajectoryNode] = []
        for obs in sorted_obs:
            cid = obs.get("camera_id", "UNKNOWN")
            cam_meta = self.cameras.get(cid, {})
            cname = cam_meta.get("name", cid)
            lat = float(cam_meta.get("latitude", 0.0))
            lon = float(cam_meta.get("longitude", 0.0))

            t_first = float(obs.get("first_timestamp", 0.0))
            t_last = float(obs.get("last_timestamp", t_first))
            dur = max(0.0, t_last - t_first)

            bbox = None
            if obs.get("bbox_x1") is not None:
                bbox = (
                    int(obs.get("bbox_x1", 0)),
                    int(obs.get("bbox_y1", 0)),
                    int(obs.get("bbox_x2", 0)),
                    int(obs.get("bbox_y2", 0)),
                )

            node = TrajectoryNode(
                camera_id=cid,
                camera_name=cname,
                latitude=lat,
                longitude=lon,
                first_timestamp=t_first,
                last_timestamp=t_last,
                duration_seconds=dur,
                local_track_id=int(obs.get("local_track_id", 0)),
                canonical_plate=obs.get("canonical_plate"),
                plate_confidence=float(obs.get("plate_confidence", 0.0) or 0.0),
                crop_quality=float(obs.get("crop_quality", 0.0) or 0.0),
                bbox=bbox,
                match_status=str(obs.get("match_status", "MATCH")),
                match_method=str(obs.get("match_method", "new_identity")),
                match_confidence=float(obs.get("match_confidence", 1.0) or 1.0),
            )
            nodes.append(node)

        # 2. Build segments between consecutive nodes
        segments: List[TrajectorySegment] = []
        overall_anomalies: List[str] = []
        total_net_dist = 0.0
        total_hav_dist = 0.0

        for i in range(len(nodes) - 1):
            prev_n = nodes[i]
            next_n = nodes[i + 1]

            # Critical adjustment 1: transit interval = next.first_timestamp - prev.last_timestamp
            t_from = prev_n.last_timestamp
            t_to = next_n.first_timestamp
            delta_t = t_to - t_from

            seg_anomalies: List[str] = []
            is_temporal_anom = False
            is_velocity_anom = False
            is_unreach_net = False
            is_same_cam = (prev_n.camera_id == next_n.camera_id)

            # Air straight-line distance
            if prev_n.latitude != 0.0 and next_n.latitude != 0.0:
                hav_dist = haversine_distance_km(prev_n.latitude, prev_n.longitude, next_n.latitude, next_n.longitude)
            else:
                hav_dist = 0.0 if is_same_cam else None

            if is_same_cam:
                # Consecutive sighting on same camera
                net_dist = 0.0
                hav_dist = 0.0
                speed = None  # Speed between identical camera positions is not a city transit
                if delta_t < 0.0:
                    is_temporal_anom = True
                    msg = f"Negative transit time ({delta_t:.1f}s) between consecutive tracks on same camera {prev_n.camera_id}"
                    seg_anomalies.append(msg)
                    overall_anomalies.append(msg)
            else:
                # Inter-camera transit
                net_dist = self.get_network_distance_km(prev_n.camera_id, next_n.camera_id)
                if net_dist is None:
                    is_unreach_net = True
                    msg = f"No route in network graph between {prev_n.camera_id} and {next_n.camera_id}"
                    seg_anomalies.append(msg)
                    overall_anomalies.append(msg)
                    speed = None
                else:
                    total_net_dist += net_dist

                if hav_dist is not None:
                    total_hav_dist += hav_dist

                if delta_t <= 0.0:
                    is_temporal_anom = True
                    msg = f"Negative or zero transit time ({delta_t:.1f}s) from {prev_n.camera_id} to {next_n.camera_id}"
                    seg_anomalies.append(msg)
                    overall_anomalies.append(msg)
                    speed = None
                elif net_dist is not None and net_dist > 0.0:
                    hours = delta_t / 3600.0
                    speed = net_dist / hours
                    # Critical adjustment 2: Document 140 km/h plausibility bound, label as velocity anomaly
                    if speed > self.max_plausible_speed_kmh:
                        is_velocity_anom = True
                        msg = (
                            f"Calculated transit speed ({speed:.1f} km/h) from {prev_n.camera_id} to {next_n.camera_id} "
                            f"exceeds physical plausibility bound ({self.max_plausible_speed_kmh:.1f} km/h)"
                        )
                        seg_anomalies.append(msg)
                        overall_anomalies.append(msg)
                else:
                    speed = None

            segment = TrajectorySegment(
                from_camera_id=prev_n.camera_id,
                to_camera_id=next_n.camera_id,
                from_timestamp=t_from,
                to_timestamp=t_to,
                transit_time_seconds=delta_t,
                network_distance_km=net_dist,
                haversine_distance_km=hav_dist,
                speed_kmh=speed,
                is_same_camera=is_same_cam,
                is_velocity_anomaly=is_velocity_anom,
                is_temporal_anomaly=is_temporal_anom,
                is_unreachable_network=is_unreach_net,
                anomalies=seg_anomalies,
            )
            segments.append(segment)

        # 3. Overall trajectory metrics
        first_seen = nodes[0].first_timestamp
        last_seen = nodes[-1].last_timestamp
        total_duration = max(0.0, last_seen - first_seen)

        avg_speed = None
        if total_net_dist > 0.0 and total_duration > 0.0:
            avg_speed = total_net_dist / (total_duration / 3600.0)

        cplate = None
        vtype = "car"
        vstatus = "active"
        if global_vehicle:
            cplate = global_vehicle.get("canonical_plate")
            vtype = global_vehicle.get("vehicle_type", "car")
            vstatus = global_vehicle.get("status", "active")
        else:
            # Infer from nodes
            for n in nodes:
                if n.canonical_plate:
                    cplate = n.canonical_plate
                    break

        return VehicleTrajectory(
            global_id=global_id,
            canonical_plate=cplate,
            vehicle_type=vtype,
            status=vstatus,
            first_seen_ts=first_seen,
            last_seen_ts=last_seen,
            total_duration_seconds=total_duration,
            total_network_distance_km=total_net_dist,
            total_haversine_distance_km=total_hav_dist,
            average_speed_kmh=avg_speed,
            nodes=nodes,
            segments=segments,
            anomalies=overall_anomalies,
        )

    def reconstruct(self, conn: sqlite3.Connection, global_id: str) -> Optional[VehicleTrajectory]:
        """Reconstruct a trajectory for a global_id from SQLite database."""
        gv = get_global_vehicle(conn, global_id)
        obs_list = get_vehicle_trajectory(conn, global_id)
        if not obs_list:
            return None
        return self.reconstruct_from_observations(global_id, obs_list, global_vehicle=gv)

    def reconstruct_by_plate(self, conn: sqlite3.Connection, plate_text: str) -> Optional[VehicleTrajectory]:
        """Reconstruct a trajectory by vehicle canonical license plate from SQLite database."""
        gv = get_global_vehicle_by_plate(conn, plate_text)
        if not gv:
            return None
        global_id = gv.get("global_id")
        if not global_id:
            return None
        return self.reconstruct(conn, global_id)

    def list_all_trajectories(self, conn: sqlite3.Connection, limit: int = 100) -> List[VehicleTrajectory]:
        """Reconstruct all trajectories in the database, ordered by recency."""
        gvs = get_all_global_vehicles(conn, limit=limit)
        trajectories: List[VehicleTrajectory] = []
        for gv in gvs:
            gid = gv.get("global_id")
            if gid:
                t = self.reconstruct(conn, gid)
                if t:
                    trajectories.append(t)
        return trajectories


# Convenience top-level functions
_default_reconstructor: Optional[TrajectoryReconstructor] = None


def get_reconstructor() -> TrajectoryReconstructor:
    """Get or initialize singleton TrajectoryReconstructor."""
    global _default_reconstructor
    if _default_reconstructor is None:
        _default_reconstructor = TrajectoryReconstructor()
    return _default_reconstructor


def reconstruct_trajectory(conn: sqlite3.Connection, global_id: str) -> Optional[VehicleTrajectory]:
    """Reconstruct vehicle trajectory by global_id using default reconstructor."""
    return get_reconstructor().reconstruct(conn, global_id)


def reconstruct_trajectory_by_plate(conn: sqlite3.Connection, plate_text: str) -> Optional[VehicleTrajectory]:
    """Reconstruct vehicle trajectory by canonical plate using default reconstructor."""
    return get_reconstructor().reconstruct_by_plate(conn, plate_text)


def list_all_trajectories(conn: sqlite3.Connection, limit: int = 100) -> List[VehicleTrajectory]:
    """Reconstruct all vehicle trajectories using default reconstructor."""
    return get_reconstructor().list_all_trajectories(conn, limit=limit)
