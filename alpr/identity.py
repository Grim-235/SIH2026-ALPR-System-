"""
Global Vehicle Identity Resolution Engine (Phase 6A).

Resolves local single-camera vehicle tracks into city-wide multi-camera global identities
using a strict two-stage decision process:
1. Hard Feasibility Filtering (topology reachability, time direction, physical speed bound, class compatibility).
2. Multi-Modal Identity Scoring (dynamic fusion of plate, ReID, and spatiotemporal evidence).
3. Decision Logic with margin requirements: MATCH, UNCERTAIN (ambiguous/borderline), NEW.
"""

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from alpr.reid import compute_similarity

log = logging.getLogger("alpr.identity")


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the standard Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    s1, s2 = s1.upper(), s2.upper()
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[m][n]


def compute_plate_similarity(p1: Optional[str], p2: Optional[str]) -> float:
    """
    Compute normalized similarity in [0.0, 1.0] between two license plate strings.
    Gives credit for common Indian ALPR visual confusions (0/O, 8/B, 1/I, 5/S).
    """
    if not p1 or not p2:
        return 0.0

    s1 = "".join(c for c in p1.upper() if c.isalnum())
    s2 = "".join(c for c in p2.upper() if c.isalnum())

    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    raw_dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    base_sim = max(0.0, 1.0 - (raw_dist / max_len))

    # Character confusion normalization:
    # Check if distance drops when mapping easily confused characters
    confusion_map = {"0": "O", "8": "B", "1": "I", "5": "S", "2": "Z"}
    s1_norm = "".join(confusion_map.get(c, c) for c in s1)
    s2_norm = "".join(confusion_map.get(c, c) for c in s2)
    norm_dist = levenshtein_distance(s1_norm, s2_norm)

    if norm_dist < raw_dist:
        adjusted_sim = max(0.0, 1.0 - (norm_dist / max_len))
        # Weighted average favoring normalized match with slight penalty
        return min(1.0, 0.4 * base_sim + 0.6 * adjusted_sim)

    return base_sim


@dataclass
class VehicleObservation:
    """
    A single vehicle observation produced by a camera worker upon track finalization
    or representative state capture.
    """
    camera_id: str
    track_id: int
    timestamp: float
    vehicle_type: str = "car"
    canonical_plate: Optional[str] = None
    plate_confidence: float = 0.0
    best_reid_embedding: Optional[np.ndarray] = None
    crop_quality: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class GlobalVehicleIdentity:
    """
    State of a single physical vehicle tracked across multiple cameras in the city.
    """
    global_id: str
    canonical_plate: Optional[str] = None
    plate_confidence: float = 0.0
    vehicle_type: str = "car"
    representative_embedding: Optional[np.ndarray] = None
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    first_camera_id: str = ""
    last_camera_id: str = ""
    sighting_count: int = 1
    camera_trajectory: List[str] = field(default_factory=list)
    track_refs: List[Tuple[str, int]] = field(default_factory=list)  # (camera_id, local_track_id)
    status: str = "active"  # "active", "confirmed", "uncertain"

    # Internal pool of high-quality embeddings to protect representation
    _embedding_pool: List[Tuple[np.ndarray, float]] = field(default_factory=list, repr=False)

    def update_with_observation(
        self,
        obs: VehicleObservation,
        min_crop_quality: float = 0.0,
        max_pool_size: int = 5,
    ) -> None:
        """
        Incorporate a new observation into this global identity.
        Protects the representative embedding from low-quality updates.
        """
        self.last_seen_ts = obs.timestamp
        self.last_camera_id = obs.camera_id
        self.sighting_count += 1
        self.camera_trajectory.append(obs.camera_id)
        self.track_refs.append((obs.camera_id, obs.track_id))

        # Update canonical plate if observation offers higher confidence
        if obs.canonical_plate and obs.plate_confidence > self.plate_confidence:
            self.canonical_plate = obs.canonical_plate
            self.plate_confidence = obs.plate_confidence

        # Protect and update representative embedding
        if obs.best_reid_embedding is not None and obs.crop_quality >= min_crop_quality:
            emb = np.asarray(obs.best_reid_embedding, dtype=np.float32).ravel()
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                unit_emb = emb / norm
                self._embedding_pool.append((unit_emb, max(0.1, obs.crop_quality)))

                # Bound pool size by highest quality
                if len(self._embedding_pool) > max_pool_size:
                    self._embedding_pool.sort(key=lambda x: x[1], reverse=True)
                    self._embedding_pool = self._embedding_pool[:max_pool_size]

                # Weighted mean aggregation
                stacked = np.stack([x[0] for x in self._embedding_pool], axis=0)
                weights = np.array([x[1] for x in self._embedding_pool], dtype=np.float32)
                weights /= np.sum(weights)

                weighted_mean = np.sum(stacked * weights[:, np.newaxis], axis=0)
                mean_norm = np.linalg.norm(weighted_mean)
                if mean_norm > 1e-6:
                    self.representative_embedding = (weighted_mean / mean_norm).astype(np.float32)


@dataclass
class IdentityMatchResult:
    """The result and audit trail of resolving a vehicle observation."""
    status: str  # "MATCH", "NEW", "UNCERTAIN"
    global_id: Optional[str]
    matched_candidate: Optional[GlobalVehicleIdentity] = None
    confidence: float = 0.0
    match_method: str = "none"  # "plate_exact", "plate_reid_fusion", "reid_only", "plate_only", "new_identity", "uncertain_ambiguous"
    plate_similarity: Optional[float] = None
    reid_similarity: Optional[float] = None
    transit_speed_kmh: Optional[float] = None
    distance_km: Optional[float] = None
    time_delta_sec: Optional[float] = None
    candidate_scores: List[Tuple[str, float]] = field(default_factory=list)
    reason: str = ""


class GlobalIdentityResolver:
    """
    City-Wide Multi-Camera Vehicle Identity Resolution Engine.

    Enforces strict hard feasibility filtering, dynamic modality fusion,
    margin-based decision rules, and embedding quality protection.
    """

    def __init__(
        self,
        camera_graph_path: Optional[Union[str, Path]] = "configs/camera_graph.json",
        max_candidate_window_sec: float = 7200.0,    # 2 hours search window
        min_reappearance_interval_sec: float = 15.0,  # Same camera minimum separation
        max_speed_kmh: float = 140.0,                # Physical hard upper velocity limit
        min_speed_kmh: float = 5.0,                  # Slow urban traffic lower bound
        expected_speed_kmh: float = 40.0,             # Expected transit speed for plausibility
        match_threshold: float = 0.75,               # Minimum score for a MATCH
        uncertain_threshold: float = 0.55,           # Score below which candidates are rejected as NEW
        min_margin: float = 0.10,                    # Minimum gap over 2nd best candidate required for MATCH
        reid_match_threshold: float = 0.70,          # Base ReID similarity expectation (configurable)
        base_plate_weight: float = 0.55,
        base_reid_weight: float = 0.35,
        base_time_weight: float = 0.10,
    ):
        self.max_candidate_window_sec = max_candidate_window_sec
        self.min_reappearance_interval_sec = min_reappearance_interval_sec
        self.max_speed_kmh = max_speed_kmh
        self.min_speed_kmh = min_speed_kmh
        self.expected_speed_kmh = expected_speed_kmh
        self.match_threshold = match_threshold
        self.uncertain_threshold = uncertain_threshold
        self.min_margin = min_margin
        self.reid_match_threshold = reid_match_threshold

        self.base_plate_weight = base_plate_weight
        self.base_reid_weight = base_reid_weight
        self.base_time_weight = base_time_weight

        # Global identities storage (in-memory for engine)
        self.identities: Dict[str, GlobalVehicleIdentity] = {}
        self._next_id_counter = 1
        self._lock = threading.Lock()

        # Camera graph shortest paths: distances_km[cam_a][cam_b]
        self.distances_km: Dict[str, Dict[str, float]] = {}
        if camera_graph_path:
            self._load_camera_graph(camera_graph_path)

    def _load_camera_graph(self, path: Union[str, Path]) -> None:
        """Load topology graph and compute all-pairs shortest paths using Floyd-Warshall."""
        p = Path(path)
        if not p.exists():
            log.warning("Camera graph file not found at %s. Graph routing disabled.", path)
            return

        with open(p, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        cameras = list(graph_data.keys())
        # Initialize distance matrix
        dist: Dict[str, Dict[str, float]] = {c1: {c2: math.inf for c2 in cameras} for c1 in cameras}
        for c in cameras:
            dist[c][c] = 0.0

        for cam_id, info in graph_data.items():
            direct_dists = info.get("distances_km", {})
            for neighbor, d in direct_dists.items():
                if neighbor in cameras:
                    dist[cam_id][neighbor] = float(d)

        # Floyd-Warshall algorithm for all-pairs shortest paths
        for k in cameras:
            for i in cameras:
                for j in cameras:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        self.distances_km = dist
        log.info("Loaded camera graph with %d cameras and computed shortest paths.", len(cameras))

    def get_distance_km(self, cam1: str, cam2: str) -> Optional[float]:
        """Return shortest network distance between two cameras in km, or None if unreachable."""
        if cam1 == cam2:
            return 0.0
        if cam1 in self.distances_km and cam2 in self.distances_km[cam1]:
            d = self.distances_km[cam1][cam2]
            return d if not math.isinf(d) else None
        return None

    def check_feasibility(
        self,
        candidate: GlobalVehicleIdentity,
        obs: VehicleObservation,
    ) -> Tuple[bool, str, Optional[float], Optional[float]]:
        """
        Stage 1: Hard Feasibility Filtering.
        MUST pass all hard physical constraints before any soft scoring is evaluated.

        Returns:
            (is_feasible, reason, distance_km, transit_speed_kmh)
        """
        delta_t = obs.timestamp - candidate.last_seen_ts

        # 1. Temporal direction: observation must be strictly after candidate
        if delta_t <= 0:
            return False, f"Negative or zero time delta (delta_t={delta_t:.2f}s)", None, None

        # 2. Candidate window expiration
        if delta_t > self.max_candidate_window_sec:
            return False, f"Candidate window expired (delta_t={delta_t:.1f}s > {self.max_candidate_window_sec:.1f}s)", None, None

        # 3. Vehicle class compatibility
        # Strict class incompatibility rules (e.g. motorcycle != truck/bus)
        c_class = candidate.vehicle_type.lower()
        o_class = obs.vehicle_type.lower()
        if c_class != o_class:
            incompatible_pairs = [
                {"motorcycle", "car"},
                {"motorcycle", "truck"},
                {"motorcycle", "bus"},
                {"bus", "motorcycle"},
                {"truck", "motorcycle"},
            ]
            if {c_class, o_class} in incompatible_pairs:
                return False, f"Incompatible vehicle class: {c_class} vs {o_class}", None, None

        # 4. Topology and Transit Speed Constraint
        if candidate.last_camera_id == obs.camera_id:
            # Same-camera re-appearance check
            if delta_t < self.min_reappearance_interval_sec:
                return False, (
                    f"Same-camera rapid re-appearance below minimum interval "
                    f"({delta_t:.1f}s < {self.min_reappearance_interval_sec:.1f}s)"
                ), 0.0, 0.0
            return True, "Feasible (same-camera re-appearance)", 0.0, 0.0
        else:
            # Cross-camera transition
            dist = self.get_distance_km(candidate.last_camera_id, obs.camera_id)
            if dist is None:
                return False, f"Unreachable camera transition {candidate.last_camera_id} -> {obs.camera_id}", None, None

            # Calculate transit speed in km/h
            speed_kmh = (dist / delta_t) * 3600.0
            if speed_kmh > self.max_speed_kmh:
                return False, (
                    f"Physically impossible transit speed: {speed_kmh:.1f} km/h > {self.max_speed_kmh:.1f} km/h "
                    f"({dist:.1f}km in {delta_t:.1f}s)"
                ), dist, speed_kmh

            return True, "Feasible cross-camera transition", dist, speed_kmh

    def score_candidate(
        self,
        candidate: GlobalVehicleIdentity,
        obs: VehicleObservation,
        distance_km: Optional[float],
        transit_speed_kmh: Optional[float],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Stage 2: Multi-Modal Soft Scoring with Dynamic Weight Re-normalization.

        Returns:
            (fused_score, metrics_dict)
        """
        plate_sim: Optional[float] = None
        reid_sim: Optional[float] = None
        time_score: Optional[float] = None

        active_weights = {}

        # 1. Plate Evidence
        if candidate.canonical_plate and obs.canonical_plate:
            plate_sim = compute_plate_similarity(candidate.canonical_plate, obs.canonical_plate)
            # Modulate by OCR confidences
            conf_factor = min(1.0, max(0.5, (candidate.plate_confidence + obs.plate_confidence) / 2.0))
            plate_score = plate_sim * conf_factor
            active_weights["plate"] = (self.base_plate_weight, plate_score)

        # 2. ReID Appearance Evidence
        if candidate.representative_embedding is not None and obs.best_reid_embedding is not None:
            raw_cosine = compute_similarity(candidate.representative_embedding, obs.best_reid_embedding)
            # Map cosine [-1, 1] -> [0, 1]
            reid_sim = max(0.0, min(1.0, float(raw_cosine)))
            active_weights["reid"] = (self.base_reid_weight, reid_sim)

        # 3. Spatiotemporal Transit Plausibility
        if distance_km is not None and distance_km > 0 and transit_speed_kmh is not None:
            # Transit plausibility curve: Gaussian decay centered at expected_speed_kmh
            sigma = 25.0  # Speed variance in km/h
            diff = abs(transit_speed_kmh - self.expected_speed_kmh)
            time_score = math.exp(-0.5 * (diff / sigma) ** 2)
            active_weights["time"] = (self.base_time_weight, time_score)
        elif distance_km == 0.0:
            # Same-camera reappearance: reasonable plausibility if well beyond min interval
            delta_t = obs.timestamp - candidate.last_seen_ts
            time_score = min(1.0, delta_t / 120.0)
            active_weights["time"] = (self.base_time_weight, time_score)

        if not active_weights:
            return 0.0, {
                "plate_sim": None,
                "reid_sim": None,
                "time_score": None,
                "fused_score": 0.0,
            }

        # Dynamic weight re-normalization
        total_base_weight = sum(w for w, _ in active_weights.values())
        fused_score = 0.0
        for w, val in active_weights.values():
            norm_weight = w / total_base_weight
            fused_score += norm_weight * val

        fused_score = max(0.0, min(1.0, float(fused_score)))

        metrics = {
            "plate_sim": plate_sim,
            "reid_sim": reid_sim,
            "time_score": time_score,
            "fused_score": fused_score,
            "active_modalities": list(active_weights.keys()),
        }
        return fused_score, metrics

    def resolve_observation(self, obs: VehicleObservation) -> Tuple[GlobalVehicleIdentity, IdentityMatchResult]:
        """
        Process a single vehicle observation and resolve its global identity.
        Thread-safe across multiple concurrent camera workers.

        Returns:
            (global_identity, match_result)
        """
        with self._lock:
            return self._resolve_observation_locked(obs)

    def _resolve_observation_locked(self, obs: VehicleObservation) -> Tuple[GlobalVehicleIdentity, IdentityMatchResult]:
        feasible_candidates: List[Tuple[GlobalVehicleIdentity, float, Dict[str, Any], float, float]] = []

        for candidate in self.identities.values():
            is_feasible, reason, dist, speed = self.check_feasibility(candidate, obs)
            if is_feasible:
                score, metrics = self.score_candidate(candidate, obs, dist, speed)
                feasible_candidates.append((candidate, score, metrics, dist or 0.0, speed or 0.0))

        # Deterministic sort: descending by score, then ascending by global_id for reproducibility
        feasible_candidates.sort(key=lambda x: (x[1], -int(x[0].global_id.split("-")[-1]) if "-" in x[0].global_id else 0), reverse=True)

        candidate_scores = [(c[0].global_id, c[1]) for c in feasible_candidates]

        # Case 1: No candidates pass hard feasibility
        if not feasible_candidates:
            new_id = self._create_new_identity(obs)
            result = IdentityMatchResult(
                status="NEW",
                global_id=new_id.global_id,
                matched_candidate=new_id,
                confidence=1.0,
                match_method="new_identity",
                candidate_scores=[],
                reason="NEW GLOBAL ID because no active candidate is eligible",
            )
            return new_id, result

        best_cand, best_score, best_metrics, best_dist, best_speed = feasible_candidates[0]
        second_score = feasible_candidates[1][1] if len(feasible_candidates) > 1 else 0.0
        margin = best_score - second_score

        # Case 2: All feasible candidates score below uncertain threshold
        if best_score < self.uncertain_threshold:
            new_id = self._create_new_identity(obs)
            result = IdentityMatchResult(
                status="NEW",
                global_id=new_id.global_id,
                matched_candidate=new_id,
                confidence=1.0 - best_score,
                match_method="new_identity",
                plate_similarity=best_metrics.get("plate_sim"),
                reid_similarity=best_metrics.get("reid_sim"),
                transit_speed_kmh=best_speed,
                distance_km=best_dist,
                candidate_scores=candidate_scores,
                reason=(
                    f"NEW GLOBAL ID because top candidate score {best_score:.3f} "
                    f"< threshold {self.uncertain_threshold:.2f}"
                ),
            )
            return new_id, result

        # Case 3: High score candidate
        if best_score >= self.match_threshold:
            # Check margin to prevent false merges when candidates are ambiguous
            if len(feasible_candidates) > 1 and margin < self.min_margin:
                # Ambiguous top candidates -> UNCERTAIN
                result = IdentityMatchResult(
                    status="UNCERTAIN",
                    global_id=best_cand.global_id,
                    matched_candidate=best_cand,
                    confidence=best_score,
                    match_method="uncertain_ambiguous",
                    plate_similarity=best_metrics.get("plate_sim"),
                    reid_similarity=best_metrics.get("reid_sim"),
                    transit_speed_kmh=best_speed,
                    distance_km=best_dist,
                    time_delta_sec=obs.timestamp - best_cand.last_seen_ts,
                    candidate_scores=candidate_scores,
                    reason=(
                        f"Ambiguous top candidates: {best_cand.global_id} ({best_score:.3f}) vs "
                        f"{feasible_candidates[1][0].global_id} ({second_score:.3f}) "
                        f"with margin {margin:.3f} < {self.min_margin:.2f}"
                    ),
                )
                return best_cand, result

            # Clear winner -> MATCH
            method = "plate_reid_fusion"
            if "plate" in best_metrics.get("active_modalities", []) and "reid" not in best_metrics.get("active_modalities", []):
                method = "plate_only"
            elif "reid" in best_metrics.get("active_modalities", []) and "plate" not in best_metrics.get("active_modalities", []):
                method = "reid_only"

            best_cand.update_with_observation(obs)
            result = IdentityMatchResult(
                status="MATCH",
                global_id=best_cand.global_id,
                matched_candidate=best_cand,
                confidence=best_score,
                match_method=method,
                plate_similarity=best_metrics.get("plate_sim"),
                reid_similarity=best_metrics.get("reid_sim"),
                transit_speed_kmh=best_speed,
                distance_km=best_dist,
                time_delta_sec=obs.timestamp - best_cand.last_seen_ts,
                candidate_scores=candidate_scores,
                reason=f"Matched {best_cand.global_id} with score {best_score:.3f} (margin={margin:.3f})",
            )
            return best_cand, result

        # Case 4: Borderline score in [uncertain_threshold, match_threshold) -> UNCERTAIN
        result = IdentityMatchResult(
            status="UNCERTAIN",
            global_id=best_cand.global_id,
            matched_candidate=best_cand,
            confidence=best_score,
            match_method="uncertain_borderline",
            plate_similarity=best_metrics.get("plate_sim"),
            reid_similarity=best_metrics.get("reid_sim"),
            transit_speed_kmh=best_speed,
            distance_km=best_dist,
            time_delta_sec=obs.timestamp - best_cand.last_seen_ts,
            candidate_scores=candidate_scores,
            reason=f"Borderline score {best_score:.3f} in [{self.uncertain_threshold:.2f}, {self.match_threshold:.2f})",
        )
        return best_cand, result

    def _create_new_identity(self, obs: VehicleObservation) -> GlobalVehicleIdentity:
        """Create and register a new global vehicle identity."""
        global_id = f"GV-{self._next_id_counter:06d}"
        self._next_id_counter += 1

        new_identity = GlobalVehicleIdentity(
            global_id=global_id,
            canonical_plate=obs.canonical_plate,
            plate_confidence=obs.plate_confidence,
            vehicle_type=obs.vehicle_type,
            representative_embedding=obs.best_reid_embedding.copy() if obs.best_reid_embedding is not None else None,
            first_seen_ts=obs.timestamp,
            last_seen_ts=obs.timestamp,
            first_camera_id=obs.camera_id,
            last_camera_id=obs.camera_id,
            sighting_count=1,
            camera_trajectory=[obs.camera_id],
            track_refs=[(obs.camera_id, obs.track_id)],
            status="active",
        )
        if obs.best_reid_embedding is not None:
            new_identity._embedding_pool.append(
                (obs.best_reid_embedding.copy(), max(0.1, obs.crop_quality))
            )

        self.identities[global_id] = new_identity
        log.info("Created new global identity: %s (plate=%s, cam=%s)", global_id, obs.canonical_plate, obs.camera_id)
        return new_identity
