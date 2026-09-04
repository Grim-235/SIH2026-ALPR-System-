"""
Dashboard Service & Orchestration Adapter Module (Phase 7D).

Encapsulates analytical engines (Phases 7A-7C) and GIS presentation (Phase 7D)
behind a clean, service-oriented interface for web route consumption:
- Eliminates mathematical/kinematic logic in Flask routes.
- Enforces HTTP search contracts (200 found, 404 absent, 400 invalid/empty).
- Sanitizes vehicle dossiers: exposes ReID diagnostics (dimension 512, L2 norm 1.0, quality)
  while withholding raw 512-D float vectors from normal responses.
- Enforces structured empty-state contracts across all endpoints.
"""

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from alpr.database import (
    init_db,
    get_global_vehicle,
    get_global_vehicle_by_plate,
    get_all_global_vehicles,
    deserialize_embedding,
    record_security_alert,
    get_security_alerts,
    get_security_alert_by_id,
    acknowledge_security_alert,
    get_security_alerts_summary,
    add_enriched_blacklist_entry,
    get_enriched_blacklist,
    remove_from_blacklist,
)
from alpr.trajectory import (
    VehicleTrajectory,
    TrajectoryReconstructor,
    get_reconstructor,
)
from alpr.analytics import (
    NetworkAnalyticsReport,
    CorridorAnalyticsEngine,
    TripODRecord,
    get_analytics_engine,
)
from alpr.congestion import (
    NetworkCongestionReport,
    TrafficCongestionEngine,
)
from alpr.gis import (
    build_network_geojson,
    generate_city_traffic_map,
    get_los_color,
    FOLIUM_AVAILABLE,
)
from alpr.alerts import (
    AlertEngine,
    AlertRecord,
    ALERT_BLACKLIST_EXACT,
    ALERT_BLACKLIST_FUZZY,
    ALERT_VELOCITY_ANOMALY,
    ALERT_TEMPORAL_INVERSION,
    ALERT_TOPOLOGY_VIOLATION,
    ALERT_IDENTITY_UNCERTAIN,
    ALERT_EXCESSIVE_DWELL,
    ALERT_RAPID_LOOPING,
)

logger = logging.getLogger("alpr.service")


class DashboardService:
    """
    Unified service adapter coordinating analytical engines and presentation layers.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/alpr.db",
        cameras_path: Union[str, Path] = "configs/cameras.json",
        camera_graph_path: Union[str, Path] = "configs/camera_graph.json",
        velocity_bound_kmh: float = 140.0,
    ):
        self.db_path = Path(db_path)
        self.cameras_path = Path(cameras_path)
        self.camera_graph_path = Path(camera_graph_path)

        # Pre-instantiate analytical engines
        self.trajectory_reconstructor = TrajectoryReconstructor(
            cameras_path=self.cameras_path,
            camera_graph_path=self.camera_graph_path,
        )
        self.analytics_engine = CorridorAnalyticsEngine()
        self.congestion_engine = TrafficCongestionEngine(
            cameras_path=self.cameras_path,
        )
        self.alert_engine = AlertEngine(
            velocity_bound_kmh=velocity_bound_kmh,
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Layer 1: GIS Map Services ──

    def get_network_geojson(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Return the complete GeoJSON FeatureCollection of camera nodes and corridors
        enriched with pre-computed congestion metrics.
        """
        c = conn or self._get_connection()
        try:
            congestion_report = self.congestion_engine.analyze_db(c)
            return build_network_geojson(
                cameras_path=self.cameras_path,
                camera_graph_path=self.camera_graph_path,
                congestion_report=congestion_report,
            )
        finally:
            if conn is None:
                c.close()

    def get_folium_map_html(
        self,
        conn: Optional[sqlite3.Connection] = None,
        active_trajectory_geojson: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate and render the interactive Folium HTML map.
        """
        if not FOLIUM_AVAILABLE:
            return "<h3>Interactive map renderer unavailable (Folium library not installed).</h3>"

        c = conn or self._get_connection()
        try:
            congestion_report = self.congestion_engine.analyze_db(c)
            folium_map = generate_city_traffic_map(
                cameras_path=self.cameras_path,
                camera_graph_path=self.camera_graph_path,
                congestion_report=congestion_report,
                active_trajectory_geojson=active_trajectory_geojson,
            )
            if folium_map:
                return folium_map.get_root().render()
            return "<h3>Unable to generate map — check camera configuration.</h3>"
        finally:
            if conn is None:
                c.close()

    # ── Layer 2: Network Analytics Services ──

    def get_analytics_summary(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Return high-level network performance KPIs and modal flow breakdown.
        Guaranteed to return explicit zero-state values if no observations exist.
        """
        c = conn or self._get_connection()
        try:
            congestion_report = self.congestion_engine.analyze_db(c)
            analytics_report = self.analytics_engine.analyze_db(c)

            modal_breakdown: Dict[str, int] = {}
            for node in congestion_report.camera_node_metrics.values():
                for v_type, cnt in getattr(node, "modal_flow_breakdown", {}).items():
                    modal_breakdown[v_type] = modal_breakdown.get(v_type, 0) + cnt

            tti = congestion_report.network_average_tti
            return {
                "total_vehicles_observed": int(congestion_report.total_vehicles_observed),
                "total_transit_observations": int(congestion_report.total_transit_observations),
                "active_corridors_count": len(congestion_report.corridor_metrics),
                "network_average_tti": round(float(tti), 2) if tti is not None else None,
                "analysis_window_seconds": round(float(congestion_report.analysis_window_seconds), 1),
                "modal_flow_breakdown": modal_breakdown,
                "hotspots_count": len(congestion_report.hotspots),
                "total_trajectories": analytics_report.total_trajectories_analyzed,
                "los_proxy_disclaimer": "Level of Service (LOS) Proxy is a TTI-based project classification and is NOT claimed as facility-specific HCM LOS methodology.",
            }
        finally:
            if conn is None:
                c.close()

    def get_corridors_metrics(self, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """
        Return structured list of corridor statistics, speeds, and LOS proxies.
        """
        c = conn or self._get_connection()
        try:
            congestion_report = self.congestion_engine.analyze_db(c)
            results = []
            for corr_id, metric in congestion_report.corridor_metrics.items():
                m_dict = metric.to_dict()
                m_dict["color"] = get_los_color(metric.los_proxy)
                results.append(m_dict)
            return sorted(results, key=lambda x: (x.get("travel_time_index") or 0.0), reverse=True)
        finally:
            if conn is None:
                c.close()

    def get_hotspots(self, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """
        Return congestion hotspots ordered by severity.
        """
        c = conn or self._get_connection()
        try:
            congestion_report = self.congestion_engine.analyze_db(c)
            return list(congestion_report.hotspots)
        finally:
            if conn is None:
                c.close()

    def get_od_matrix(self, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """
        Return trip-level origin-destination records.
        """
        c = conn or self._get_connection()
        try:
            analytics_report = self.analytics_engine.analyze_db(c)
            return [rec.to_dict() for rec in analytics_report.od_details.values()]
        finally:
            if conn is None:
                c.close()

    def get_time_windows(self, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """
        Return departure-time window traffic profiles.
        """
        c = conn or self._get_connection()
        try:
            analytics_report = self.analytics_engine.analyze_db(c)
            results = []
            for tw_label, corridors in analytics_report.time_windows.items():
                results.append({
                    "time_window": tw_label,
                    "active_corridors": len(corridors),
                    "corridors": [corr.to_dict() for corr in corridors.values()],
                })
            return sorted(results, key=lambda x: x["time_window"])
        finally:
            if conn is None:
                c.close()

    # ── Layer 3: Vehicle Search & Trajectory Services ──

    def search_vehicle(
        self,
        query: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Search for a vehicle by license plate OR Global Vehicle ID (GV-XXXXXX).

        HTTP status contract:
        - 200: Vehicle found, returns trajectory dossier and GeoJSON.
        - 400: Empty query or malformed query string.
        - 404: Valid query syntax, but vehicle is not present in database.
        """
        if not query or not str(query).strip():
            return 400, {
                "error": "Query parameter 'q' is required and cannot be empty.",
                "query": query,
            }

        clean_q = str(query).strip().upper()

        # Reject excessively long or suspicious inputs
        if len(clean_q) > 32 or not re.match(r"^[A-Z0-9\-_ ]+$", clean_q):
            return 400, {
                "error": "Malformed vehicle query identifier. Must be alphanumeric plate or GV-XXXXXX.",
                "query": query,
            }

        c = conn or self._get_connection()
        try:
            trajectory: Optional[VehicleTrajectory] = None
            gv_record: Optional[Dict[str, Any]] = None

            # 1. Search by Global Vehicle ID
            if clean_q.startswith("GV-") or clean_q.startswith("GV_"):
                gv_record = get_global_vehicle(c, clean_q)
                if gv_record:
                    trajectory = self.trajectory_reconstructor.reconstruct(c, clean_q)

            # 2. Search by License Plate
            if not trajectory:
                # Try exact plate or normalized plate
                gv_record = get_global_vehicle_by_plate(c, clean_q)
                if not gv_record:
                    # Strip spaces/dashes for fuzzy match
                    condensed = clean_q.replace(" ", "").replace("-", "")
                    gv_record = get_global_vehicle_by_plate(c, condensed)

                if gv_record:
                    trajectory = self.trajectory_reconstructor.reconstruct(
                        c, gv_record["global_id"]
                    )
                else:
                    # Fallback to direct trajectory reconstruction by plate text
                    trajectory = self.trajectory_reconstructor.reconstruct_by_plate(c, clean_q)

            # Not found
            if not trajectory:
                return 404, {
                    "error": f"No vehicle found matching identifier '{query}'.",
                    "query": query,
                }

            # 3. Assemble Sanitized Vehicle Dossier
            # Extract ReID diagnostics without raw 512 float dump
            rep_embedding_raw = gv_record.get("representative_embedding") if gv_record else None
            reid_diagnostics = {
                "has_embedding": False,
                "dimension": 512,
                "l2_norm": None,
                "crop_quality": None,
            }

            if rep_embedding_raw is not None:
                emb = deserialize_embedding(rep_embedding_raw)
                if emb is not None and isinstance(emb, np.ndarray):
                    norm = float(np.linalg.norm(emb))
                    reid_diagnostics = {
                        "has_embedding": True,
                        "dimension": int(emb.shape[0]),
                        "l2_norm": round(norm, 4),
                        "crop_quality": round(float(gv_record.get("plate_confidence") or 0.0), 2),
                    }

            # Build hops table
            hops = []
            for idx, node in enumerate(trajectory.nodes, 1):
                # Find matching forward segment if exists
                fwd_speed = None
                is_anom = False
                if idx - 1 < len(trajectory.segments):
                    seg = trajectory.segments[idx - 1]
                    fwd_speed = round(seg.speed_kmh, 1) if seg.speed_kmh is not None else None
                    is_anom = seg.is_velocity_anomaly or seg.is_temporal_anomaly or seg.is_unreachable_network

                first_iso = datetime.fromtimestamp(node.first_timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if node.first_timestamp else "N/A"
                last_iso = datetime.fromtimestamp(node.last_timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if node.last_timestamp else "N/A"

                hops.append({
                    "hop_index": idx,
                    "camera_id": node.camera_id,
                    "camera_name": node.camera_name,
                    "first_seen_iso": first_iso,
                    "last_seen_iso": last_iso,
                    "dwell_duration_seconds": round(node.duration_seconds, 1),
                    "transit_speed_to_next_kmh": fwd_speed,
                    "is_anomaly": is_anom,
                })

            plate_conf = gv_record.get("plate_confidence") if gv_record else (trajectory.nodes[0].plate_confidence if trajectory.nodes else None)
            has_vel_anom = any(s.is_velocity_anomaly for s in trajectory.segments)
            has_temp_anom = any(s.is_temporal_anomaly for s in trajectory.segments)
            has_unreach = any(s.is_unreachable_network for s in trajectory.segments)

            response_payload = {
                "status": "success",
                "global_id": trajectory.global_id,
                "canonical_plate": trajectory.canonical_plate or "UNIDENTIFIED",
                "vehicle_type": trajectory.vehicle_type,
                "plate_confidence": round(float(plate_conf), 2) if plate_conf is not None else None,
                "reid_diagnostics": reid_diagnostics,
                "trajectory_summary": {
                    "total_network_distance_km": round(trajectory.total_network_distance_km, 2),
                    "total_haversine_distance_km": round(trajectory.total_haversine_distance_km, 2),
                    "total_duration_seconds": round(trajectory.total_duration_seconds, 1),
                    "average_speed_kmh": round(trajectory.average_speed_kmh, 1) if trajectory.average_speed_kmh is not None else None,
                    "sightings_count": len(trajectory.nodes),
                    "has_velocity_anomaly": has_vel_anom,
                    "has_temporal_anomaly": has_temp_anom,
                    "has_unreachable_segment": has_unreach,
                    "anomalies": list(trajectory.anomalies),
                },
                "sighting_hops": hops,
                "geojson": trajectory.to_geojson(),
            }
            return 200, response_payload
        finally:
            if conn is None:
                c.close()

    # ── Layer 4: Security Alerts & Threat Surveillance (Phase 7E) ──

    def get_alerts_summary(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """Get high-level summary of active, unacknowledged, and categorized security alerts."""
        c = conn or self._get_connection()
        try:
            return get_security_alerts_summary(c)
        finally:
            if conn is None:
                c.close()

    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        only_unacknowledged: bool = False,
        global_id: Optional[str] = None,
        canonical_plate: Optional[str] = None,
        limit: int = 100,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve filtered security alerts."""
        c = conn or self._get_connection()
        try:
            return get_security_alerts(
                c,
                alert_type=alert_type,
                severity=severity,
                only_unacknowledged=only_unacknowledged,
                global_id=global_id,
                canonical_plate=canonical_plate,
                limit=limit,
            )
        finally:
            if conn is None:
                c.close()

    def get_alert_by_id(
        self,
        alert_id: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve single alert details by unique alert ID."""
        c = conn or self._get_connection()
        try:
            return get_security_alert_by_id(c, alert_id)
        finally:
            if conn is None:
                c.close()

    def acknowledge_alert(
        self,
        alert_id: str,
        operator: str = "operator",
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """Mark an alert as acknowledged."""
        c = conn or self._get_connection()
        try:
            return acknowledge_security_alert(c, alert_id=alert_id, acknowledged_by=operator)
        finally:
            if conn is None:
                c.close()

    def get_blacklist(
        self,
        category: Optional[str] = None,
        active_only: bool = True,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve watchlist / blacklist records."""
        c = conn or self._get_connection()
        try:
            return get_enriched_blacklist(c, category=category, active_only=active_only)
        finally:
            if conn is None:
                c.close()

    def add_to_blacklist(
        self,
        plate: str,
        category: str = "CUSTOM",
        reason: Optional[str] = None,
        severity: str = "HIGH",
        notes: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """Add or update an enriched blacklist plate."""
        c = conn or self._get_connection()
        try:
            return add_enriched_blacklist_entry(
                c,
                plate_text=plate,
                category=category,
                reason=reason,
                severity=severity,
                notes=notes,
            )
        finally:
            if conn is None:
                c.close()

    def remove_from_blacklist(
        self,
        plate: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """Remove a plate from the watchlist / blacklist."""
        c = conn or self._get_connection()
        try:
            remove_from_blacklist(c, plate_text=plate)
            return True
        finally:
            if conn is None:
                c.close()

    def scan_and_sync_alerts(
        self,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Dict[str, Any]:
        """
        Scan all stored vehicle trajectories and observations in the database,
        evaluate them against active blacklist and anomaly rules, and persist
        any discovered alerts idempotently.
        """
        c = conn or self._get_connection()
        try:
            blacklist = get_enriched_blacklist(c, active_only=True)
            trajectories = self.trajectory_reconstructor.list_all_trajectories(c)
            total_discovered = 0
            new_persisted = 0

            for traj in trajectories:
                alerts = self.alert_engine.evaluate_trajectory(traj, blacklist_records=blacklist)
                for a in alerts:
                    total_discovered += 1
                    record_security_alert(
                        c,
                        alert_id=a.alert_id,
                        alert_type=a.alert_type,
                        severity=a.severity,
                        title=a.title,
                        description=a.description,
                        camera_id=a.camera_id,
                        timestamp=a.timestamp,
                        iso_timestamp=a.iso_timestamp,
                        global_id=a.global_id,
                        canonical_plate=a.canonical_plate,
                        details=a.details,
                    )

            summary = get_security_alerts_summary(c)
            return {
                "status": "success",
                "trajectories_scanned": len(trajectories),
                "alerts_evaluated": total_discovered,
                "summary": summary,
            }
        finally:
            if conn is None:
                c.close()


# Module-level singleton
_default_service: Optional[DashboardService] = None


def get_dashboard_service(
    db_path: Union[str, Path] = "data/alpr.db",
    cameras_path: Union[str, Path] = "configs/cameras.json",
    camera_graph_path: Union[str, Path] = "configs/camera_graph.json",
    velocity_bound_kmh: float = 140.0,
) -> DashboardService:
    """Get or create singleton DashboardService instance."""
    global _default_service
    if _default_service is None:
        _default_service = DashboardService(
            db_path=db_path,
            cameras_path=cameras_path,
            camera_graph_path=camera_graph_path,
            velocity_bound_kmh=velocity_bound_kmh,
        )
    return _default_service

