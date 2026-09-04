"""
Security Alerts, Blacklist Enforcement, Suspicious Route Diagnostics & Anomaly Filtering Engine (Phase 7E).

Implements pure domain rule evaluation across multi-camera trajectories and individual vehicle sightings:
1. Blacklist / Watchlist Matching: Exact and fuzzy matching (with Indian character confusion).
2. Kinematic Plausibility Diagnostics: Speeds exceeding physical threshold (> 140 km/h bound).
3. Temporal Inversion Diagnostics: Non-positive transit intervals (Δt <= 0).
4. Network Topology Violations: Unreachable graph transitions across non-adjacent camera nodes.
5. Identity Uncertainty Diagnostics: Observations marked UNCERTAIN by the GlobalIdentityResolver.
6. Behavioral / Loitering / Surveillance Diagnostics: Excessive dwell time (> 180s) or rapid corridor looping.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from alpr.identity import compute_plate_similarity
from alpr.trajectory import TrajectoryNode, TrajectorySegment, VehicleTrajectory

log = logging.getLogger("alpr.alerts")

# Plausibility bounds & default heuristic thresholds
DEFAULT_VELOCITY_BOUND_KMH: float = 140.0
DEFAULT_EXCESSIVE_DWELL_SECONDS: float = 180.0
DEFAULT_RAPID_LOOP_WINDOW_SECONDS: float = 300.0
DEFAULT_FUZZY_BLACKLIST_THRESHOLD: float = 0.85

# Severity hierarchy
SEVERITY_CRITICAL: str = "CRITICAL"
SEVERITY_HIGH: str = "HIGH"
SEVERITY_MEDIUM: str = "MEDIUM"
SEVERITY_LOW: str = "LOW"
SEVERITY_INFO: str = "INFO"

# Alert types
ALERT_BLACKLIST_EXACT: str = "BLACKLIST_EXACT"
ALERT_BLACKLIST_FUZZY: str = "BLACKLIST_FUZZY"
ALERT_VELOCITY_ANOMALY: str = "VELOCITY_ANOMALY"
ALERT_TEMPORAL_INVERSION: str = "TEMPORAL_INVERSION"
ALERT_TOPOLOGY_VIOLATION: str = "TOPOLOGY_VIOLATION"
ALERT_IDENTITY_UNCERTAIN: str = "IDENTITY_UNCERTAIN"
ALERT_EXCESSIVE_DWELL: str = "EXCESSIVE_DWELL"
ALERT_RAPID_LOOPING: str = "RAPID_LOOPING"


@dataclass
class AlertRecord:
    """
    Standardized domain alert representation.
    """
    alert_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    camera_id: str
    timestamp: float
    iso_timestamp: str
    global_id: Optional[str] = None
    canonical_plate: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "iso_timestamp": self.iso_timestamp,
            "global_id": self.global_id,
            "canonical_plate": self.canonical_plate,
            "details": dict(self.details),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
        }


def generate_alert_id(
    alert_type: str,
    camera_id: str,
    timestamp: float,
    key_identifier: str,
) -> str:
    """
    Generate a deterministic, collision-resistant alert ID for idempotency.
    Example format: ALT-<TYPE_CODE>-<HASH8>
    """
    type_prefixes = {
        ALERT_BLACKLIST_EXACT: "BLX",
        ALERT_BLACKLIST_FUZZY: "BLF",
        ALERT_VELOCITY_ANOMALY: "VEL",
        ALERT_TEMPORAL_INVERSION: "TMP",
        ALERT_TOPOLOGY_VIOLATION: "TOP",
        ALERT_IDENTITY_UNCERTAIN: "UNC",
        ALERT_EXCESSIVE_DWELL: "DWL",
        ALERT_RAPID_LOOPING: "LOP",
    }
    prefix = type_prefixes.get(alert_type, "GEN")
    raw_payload = f"{alert_type}:{camera_id}:{timestamp:.2f}:{key_identifier}"
    digest = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()[:10].upper()
    return f"ALT-{prefix}-{digest}"


def format_iso_timestamp(ts: float) -> str:
    """Format unix timestamp into standard ISO-8601 UTC string."""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# PURE RULE EVALUATORS
# ============================================================================

def evaluate_blacklist_match(
    plate_text: Optional[str],
    blacklist_records: List[Dict[str, Any]],
    camera_id: str,
    timestamp: float,
    global_id: Optional[str] = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_BLACKLIST_THRESHOLD,
) -> List[AlertRecord]:
    """
    Evaluate plate against blacklist entries:
    - Exact match -> BLACKLIST_EXACT (CRITICAL for STOLEN/WANTED, HIGH for others).
    - Fuzzy match (similarity >= fuzzy_threshold) -> BLACKLIST_FUZZY (MEDIUM).
    """
    if not plate_text:
        return []

    plate = plate_text.strip().upper()
    alerts: List[AlertRecord] = []
    iso_ts = format_iso_timestamp(timestamp)

    for bl in blacklist_records:
        bl_plate = str(bl.get("plate_text", "")).strip().upper()
        if not bl_plate:
            continue

        category = str(bl.get("category", "CUSTOM")).strip().upper()
        reason = bl.get("reason", "Flagged vehicle")
        configured_sev = str(bl.get("severity", "HIGH")).strip().upper()

        # 1. Exact Match
        if plate == bl_plate:
            severity = SEVERITY_CRITICAL if category in ("STOLEN", "WANTED") else (configured_sev or SEVERITY_HIGH)
            aid = generate_alert_id(ALERT_BLACKLIST_EXACT, camera_id, timestamp, f"{plate}:{bl_plate}")
            alerts.append(
                AlertRecord(
                    alert_id=aid,
                    alert_type=ALERT_BLACKLIST_EXACT,
                    severity=severity,
                    title=f"Blacklist Match: {plate} ({category})",
                    description=f"Exact match against watchlist [{category}]: {reason}",
                    camera_id=camera_id,
                    timestamp=timestamp,
                    iso_timestamp=iso_ts,
                    global_id=global_id,
                    canonical_plate=plate,
                    details={
                        "matched_plate": bl_plate,
                        "category": category,
                        "reason": reason,
                        "match_type": "EXACT",
                        "similarity": 1.0,
                    },
                )
            )
            continue  # Exact match takes precedence for this blacklist entry

        # 2. Fuzzy Match (Indian character confusion credit)
        sim = compute_plate_similarity(plate, bl_plate)
        if sim >= fuzzy_threshold:
            aid = generate_alert_id(ALERT_BLACKLIST_FUZZY, camera_id, timestamp, f"{plate}:{bl_plate}")
            alerts.append(
                AlertRecord(
                    alert_id=aid,
                    alert_type=ALERT_BLACKLIST_FUZZY,
                    severity=SEVERITY_MEDIUM,
                    title=f"Suspected Watchlist Match: {plate} (~{bl_plate})",
                    description=(
                        f"Visual similarity {sim:.2f} to blacklisted {bl_plate} "
                        f"({category}: {reason}). Requires operator review."
                    ),
                    camera_id=camera_id,
                    timestamp=timestamp,
                    iso_timestamp=iso_ts,
                    global_id=global_id,
                    canonical_plate=plate,
                    details={
                        "matched_plate": bl_plate,
                        "category": category,
                        "reason": reason,
                        "match_type": "FUZZY",
                        "similarity": round(sim, 4),
                        "fuzzy_threshold": fuzzy_threshold,
                    },
                )
            )

    return alerts


def evaluate_kinematic_anomalies(
    segment: TrajectorySegment,
    global_id: Optional[str] = None,
    canonical_plate: Optional[str] = None,
    velocity_bound_kmh: float = DEFAULT_VELOCITY_BOUND_KMH,
) -> List[AlertRecord]:
    """
    Evaluate trajectory segment for physical kinematic anomalies:
    - Speed > velocity_bound_kmh -> VELOCITY_ANOMALY (DIAGNOSTIC WARNING).
    - Transit interval <= 0 -> TEMPORAL_INVERSION (DIAGNOSTIC WARNING / CRITICAL).
    """
    alerts: List[AlertRecord] = []
    dest_cam = segment.to_camera_id
    ts = segment.to_timestamp
    iso_ts = format_iso_timestamp(ts)
    id_key = global_id or canonical_plate or "UNKNOWN"

    # Temporal Inversion / Zero Duration
    if segment.is_temporal_anomaly or segment.transit_time_seconds <= 0:
        aid = generate_alert_id(
            ALERT_TEMPORAL_INVERSION,
            dest_cam,
            ts,
            f"{segment.from_camera_id}->{dest_cam}:{id_key}",
        )
        alerts.append(
            AlertRecord(
                alert_id=aid,
                alert_type=ALERT_TEMPORAL_INVERSION,
                severity=SEVERITY_HIGH,
                title=f"Diagnostic: Temporal Inversion ({segment.from_camera_id} → {dest_cam})",
                description=(
                    f"Transit interval {segment.transit_time_seconds:.1f}s <= 0 between consecutive camera observations. "
                    f"Indicates clock skew, out-of-order packet ingestion, or erroneous association."
                ),
                camera_id=dest_cam,
                timestamp=ts,
                iso_timestamp=iso_ts,
                global_id=global_id,
                canonical_plate=canonical_plate,
                details={
                    "from_camera_id": segment.from_camera_id,
                    "to_camera_id": segment.to_camera_id,
                    "transit_time_seconds": round(segment.transit_time_seconds, 2),
                    "from_timestamp": segment.from_timestamp,
                    "to_timestamp": segment.to_timestamp,
                    "is_diagnostic": True,
                },
            )
        )

    # Velocity Anomaly
    if segment.is_velocity_anomaly or (segment.speed_kmh is not None and segment.speed_kmh > velocity_bound_kmh):
        speed = segment.speed_kmh or 0.0
        aid = generate_alert_id(
            ALERT_VELOCITY_ANOMALY,
            dest_cam,
            ts,
            f"{segment.from_camera_id}->{dest_cam}:{id_key}",
        )
        alerts.append(
            AlertRecord(
                alert_id=aid,
                alert_type=ALERT_VELOCITY_ANOMALY,
                severity=SEVERITY_MEDIUM,
                title=f"Diagnostic: Kinematic Plausibility Bound Exceeded ({speed:.1f} km/h)",
                description=(
                    f"Calculated transit speed of {speed:.1f} km/h exceeds physical plausibility bound "
                    f"({velocity_bound_kmh:.1f} km/h) over corridor {segment.from_camera_id} → {dest_cam}. "
                    f"Classified as diagnostic anomaly (possible cross-camera misassociation or timing variance)."
                ),
                camera_id=dest_cam,
                timestamp=ts,
                iso_timestamp=iso_ts,
                global_id=global_id,
                canonical_plate=canonical_plate,
                details={
                    "from_camera_id": segment.from_camera_id,
                    "to_camera_id": segment.to_camera_id,
                    "speed_kmh": round(speed, 2),
                    "velocity_bound_kmh": round(velocity_bound_kmh, 2),
                    "network_distance_km": segment.network_distance_km,
                    "transit_time_seconds": round(segment.transit_time_seconds, 2),
                    "is_diagnostic": True,
                },
            )
        )

    return alerts


def evaluate_topological_anomalies(
    segment: TrajectorySegment,
    global_id: Optional[str] = None,
    canonical_plate: Optional[str] = None,
) -> List[AlertRecord]:
    """
    Evaluate trajectory segment for network graph topology violations:
    - Sighting hops between camera nodes with no connected path in camera graph.
    """
    alerts: List[AlertRecord] = []
    if segment.is_unreachable_network and not segment.is_same_camera:
        dest_cam = segment.to_camera_id
        ts = segment.to_timestamp
        iso_ts = format_iso_timestamp(ts)
        id_key = global_id or canonical_plate or "UNKNOWN"
        aid = generate_alert_id(
            ALERT_TOPOLOGY_VIOLATION,
            dest_cam,
            ts,
            f"{segment.from_camera_id}->{dest_cam}:{id_key}",
        )
        alerts.append(
            AlertRecord(
                alert_id=aid,
                alert_type=ALERT_TOPOLOGY_VIOLATION,
                severity=SEVERITY_MEDIUM,
                title=f"Topology Violation: {segment.from_camera_id} → {dest_cam}",
                description=(
                    f"Consecutive sightings at {segment.from_camera_id} and {dest_cam} "
                    f"have no configured directed path in the urban camera graph."
                ),
                camera_id=dest_cam,
                timestamp=ts,
                iso_timestamp=iso_ts,
                global_id=global_id,
                canonical_plate=canonical_plate,
                details={
                    "from_camera_id": segment.from_camera_id,
                    "to_camera_id": segment.to_camera_id,
                    "transit_time_seconds": round(segment.transit_time_seconds, 2),
                    "is_diagnostic": True,
                },
            )
        )
    return alerts


def evaluate_identity_uncertainty(
    node: TrajectoryNode,
    global_id: Optional[str] = None,
) -> List[AlertRecord]:
    """
    Evaluate observation for identity resolver UNCERTAIN status:
    - Match confidence between thresholds or candidate margin < 0.10.
    """
    alerts: List[AlertRecord] = []
    if node.match_status == "UNCERTAIN":
        aid = generate_alert_id(
            ALERT_IDENTITY_UNCERTAIN,
            node.camera_id,
            node.first_timestamp,
            f"{node.local_track_id}:{node.canonical_plate or 'NOPLATE'}",
        )
        alerts.append(
            AlertRecord(
                alert_id=aid,
                alert_type=ALERT_IDENTITY_UNCERTAIN,
                severity=SEVERITY_LOW,
                title=f"Identity Ambiguity: Local Track #{node.local_track_id} at {node.camera_id}",
                description=(
                    f"Global identity match flagged UNCERTAIN (conf: {node.match_confidence:.2f}, "
                    f"method: {node.match_method}). Multi-modal evidence was borderline or candidate margin was narrow."
                ),
                camera_id=node.camera_id,
                timestamp=node.first_timestamp,
                iso_timestamp=format_iso_timestamp(node.first_timestamp),
                global_id=global_id,
                canonical_plate=node.canonical_plate,
                details={
                    "local_track_id": node.local_track_id,
                    "match_method": node.match_method,
                    "match_confidence": round(node.match_confidence, 4),
                    "plate_confidence": round(node.plate_confidence, 4),
                    "crop_quality": round(node.crop_quality, 2),
                },
            )
        )
    return alerts


def evaluate_behavioral_anomalies(
    trajectory: VehicleTrajectory,
    max_dwell_seconds: float = DEFAULT_EXCESSIVE_DWELL_SECONDS,
    rapid_loop_window_seconds: float = DEFAULT_RAPID_LOOP_WINDOW_SECONDS,
) -> List[AlertRecord]:
    """
    Evaluate full trajectory for behavioral surveillance patterns:
    - Excessive nodal dwell time (> max_dwell_seconds).
    - Rapid corridor looping (> 2 traversals of same directed corridor within rapid_loop_window_seconds).
    """
    alerts: List[AlertRecord] = []
    gid = trajectory.global_id
    plate = trajectory.canonical_plate

    # 1. Excessive Dwell at a single camera node
    for node in trajectory.nodes:
        if node.duration_seconds > max_dwell_seconds:
            aid = generate_alert_id(
                ALERT_EXCESSIVE_DWELL,
                node.camera_id,
                node.first_timestamp,
                f"{gid}:{node.local_track_id}",
            )
            alerts.append(
                AlertRecord(
                    alert_id=aid,
                    alert_type=ALERT_EXCESSIVE_DWELL,
                    severity=SEVERITY_LOW,
                    title=f"Excessive Dwell / Loitering: {node.camera_id} ({node.duration_seconds:.0f}s)",
                    description=(
                        f"Vehicle dwell time {node.duration_seconds:.1f}s at {node.camera_id} "
                        f"exceeds typical surveillance dwell threshold ({max_dwell_seconds:.0f}s)."
                    ),
                    camera_id=node.camera_id,
                    timestamp=node.first_timestamp,
                    iso_timestamp=format_iso_timestamp(node.first_timestamp),
                    global_id=gid,
                    canonical_plate=plate or node.canonical_plate,
                    details={
                        "dwell_duration_seconds": round(node.duration_seconds, 2),
                        "dwell_threshold_seconds": round(max_dwell_seconds, 2),
                        "first_timestamp": node.first_timestamp,
                        "last_timestamp": node.last_timestamp,
                    },
                )
            )

    # 2. Rapid Corridor Looping
    corridor_hops: Dict[Tuple[str, str], List[float]] = {}
    for seg in trajectory.segments:
        if seg.is_same_camera:
            continue
        pair = (seg.from_camera_id, seg.to_camera_id)
        corridor_hops.setdefault(pair, []).append(seg.from_timestamp)

    for (from_cam, to_cam), timestamps in corridor_hops.items():
        if len(timestamps) >= 3:
            # Check window between earliest and latest of any 3 consecutive passes
            for i in range(len(timestamps) - 2):
                span = timestamps[i + 2] - timestamps[i]
                if span <= rapid_loop_window_seconds:
                    aid = generate_alert_id(
                        ALERT_RAPID_LOOPING,
                        to_cam,
                        timestamps[i + 2],
                        f"{gid}:{from_cam}->{to_cam}",
                    )
                    alerts.append(
                        AlertRecord(
                            alert_id=aid,
                            alert_type=ALERT_RAPID_LOOPING,
                            severity=SEVERITY_MEDIUM,
                            title=f"Suspicious Route Pattern: Looping {from_cam} → {to_cam}",
                            description=(
                                f"Vehicle traversed corridor {from_cam} → {to_cam} 3 times within {span:.0f}s "
                                f"(threshold: {rapid_loop_window_seconds:.0f}s). Potential surveillance or circuitous loitering."
                            ),
                            camera_id=to_cam,
                            timestamp=timestamps[i + 2],
                            iso_timestamp=format_iso_timestamp(timestamps[i + 2]),
                            global_id=gid,
                            canonical_plate=plate,
                            details={
                                "from_camera_id": from_cam,
                                "to_camera_id": to_cam,
                                "pass_count": 3,
                                "time_span_seconds": round(span, 2),
                                "threshold_seconds": round(rapid_loop_window_seconds, 2),
                            },
                        )
                    )
                    break

    return alerts


# ============================================================================
# ALERT ENGINE COORDINATOR
# ============================================================================

class AlertEngine:
    """
    Orchestrates pure rule evaluation across single sightings and reconstructed trajectories.
    """

    def __init__(
        self,
        velocity_bound_kmh: float = DEFAULT_VELOCITY_BOUND_KMH,
        excessive_dwell_seconds: float = DEFAULT_EXCESSIVE_DWELL_SECONDS,
        rapid_loop_window_seconds: float = DEFAULT_RAPID_LOOP_WINDOW_SECONDS,
        fuzzy_blacklist_threshold: float = DEFAULT_FUZZY_BLACKLIST_THRESHOLD,
    ):
        self.velocity_bound_kmh = velocity_bound_kmh
        self.excessive_dwell_seconds = excessive_dwell_seconds
        self.rapid_loop_window_seconds = rapid_loop_window_seconds
        self.fuzzy_blacklist_threshold = fuzzy_blacklist_threshold

    def evaluate_observation(
        self,
        camera_id: str,
        timestamp: float,
        plate_text: Optional[str] = None,
        global_id: Optional[str] = None,
        match_status: str = "MATCH",
        match_confidence: float = 1.0,
        match_method: str = "plate_exact",
        blacklist_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AlertRecord]:
        """
        Evaluate an individual observation (e.g. from camera worker track finalization).
        Checks blacklist (exact & fuzzy) and identity uncertainty.
        """
        alerts: List[AlertRecord] = []
        bl = blacklist_records or []

        # Blacklist check
        if plate_text:
            alerts.extend(
                evaluate_blacklist_match(
                    plate_text=plate_text,
                    blacklist_records=bl,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    global_id=global_id,
                    fuzzy_threshold=self.fuzzy_blacklist_threshold,
                )
            )

        # Uncertainty check
        if match_status == "UNCERTAIN":
            dummy_node = TrajectoryNode(
                camera_id=camera_id,
                camera_name=camera_id,
                latitude=0.0,
                longitude=0.0,
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                duration_seconds=0.0,
                local_track_id=0,
                canonical_plate=plate_text,
                match_status=match_status,
                match_method=match_method,
                match_confidence=match_confidence,
            )
            alerts.extend(evaluate_identity_uncertainty(dummy_node, global_id=global_id))

        return alerts

    def evaluate_trajectory(
        self,
        trajectory: VehicleTrajectory,
        blacklist_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AlertRecord]:
        """
        Comprehensive evaluation of a full reconstructed vehicle trajectory:
        - Blacklist checks across canonical plate and observed node plates
        - Kinematic anomalies across segments
        - Topological graph violations
        - Identity uncertainty on sighting nodes
        - Behavioral loitering / rapid corridor looping
        """
        alerts: List[AlertRecord] = []
        bl = blacklist_records or []
        gid = trajectory.global_id
        canonical_plate = trajectory.canonical_plate

        # 1. Blacklist check on trajectory canonical plate
        if canonical_plate:
            earliest_ts = trajectory.first_seen_ts
            earliest_cam = trajectory.nodes[0].camera_id if trajectory.nodes else "CAM-UNKNOWN"
            alerts.extend(
                evaluate_blacklist_match(
                    plate_text=canonical_plate,
                    blacklist_records=bl,
                    camera_id=earliest_cam,
                    timestamp=earliest_ts,
                    global_id=gid,
                    fuzzy_threshold=self.fuzzy_blacklist_threshold,
                )
            )

        # 2. Segment-level rules (Kinematics & Topology)
        for seg in trajectory.segments:
            alerts.extend(
                evaluate_kinematic_anomalies(
                    segment=seg,
                    global_id=gid,
                    canonical_plate=canonical_plate,
                    velocity_bound_kmh=self.velocity_bound_kmh,
                )
            )
            alerts.extend(
                evaluate_topological_anomalies(
                    segment=seg,
                    global_id=gid,
                    canonical_plate=canonical_plate,
                )
            )

        # 3. Node-level rules (Identity Uncertainty & Sighting Blacklist)
        for node in trajectory.nodes:
            alerts.extend(evaluate_identity_uncertainty(node, global_id=gid))
            # If node had a different plate read than canonical plate, check it too
            if node.canonical_plate and node.canonical_plate != canonical_plate:
                alerts.extend(
                    evaluate_blacklist_match(
                        plate_text=node.canonical_plate,
                        blacklist_records=bl,
                        camera_id=node.camera_id,
                        timestamp=node.first_timestamp,
                        global_id=gid,
                        fuzzy_threshold=self.fuzzy_blacklist_threshold,
                    )
                )

        # 4. Behavioral patterns
        alerts.extend(
            evaluate_behavioral_anomalies(
                trajectory=trajectory,
                max_dwell_seconds=self.excessive_dwell_seconds,
                rapid_loop_window_seconds=self.rapid_loop_window_seconds,
            )
        )

        return alerts
