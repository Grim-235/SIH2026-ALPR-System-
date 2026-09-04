"""
Phase 6A -- Acceptance tests for Global Vehicle Identity Resolution Engine.

Tests:
1. Camera graph network shortest path computation.
2. Hard feasibility: impossible speed, negative time, unreachable camera, vehicle class mismatch, same-camera rapid reappearance.
3. Multi-modal identity scoring: exact plate, OCR confusion recovery via ReID, plate-less vehicle ReID, missing modality re-normalization.
4. Decision logic: margin requirement protection, ambiguous candidates -> UNCERTAIN, no feasible candidates -> NEW.
5. Candidate window expiration.
6. Representative embedding protection against degraded crops.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.identity import (
    GlobalVehicleIdentity,
    VehicleObservation,
    IdentityMatchResult,
    GlobalIdentityResolver,
    compute_plate_similarity,
    levenshtein_distance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("test_phase6a")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "[PASS]" if condition else "[FAIL]"
    msg = f"  {status}: {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if condition:
        PASS += 1
    else:
        FAIL += 1


def make_unit_embedding(seed: int = 42) -> np.ndarray:
    """Generate a reproducible, unit-normalized 512-D embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def make_similar_embedding(base_emb: np.ndarray, similarity: float = 0.90) -> np.ndarray:
    """Generate an embedding with a controlled cosine similarity to base_emb."""
    rng = np.random.RandomState(int(similarity * 1000))
    noise = rng.randn(len(base_emb)).astype(np.float32)
    # Orthogonalize noise relative to base_emb
    noise -= float(np.dot(noise, base_emb)) * base_emb
    noise /= np.linalg.norm(noise)

    # Blend: cos(theta) = similarity
    theta = np.arccos(min(1.0, max(-1.0, similarity)))
    blended = np.cos(theta) * base_emb + np.sin(theta) * noise
    return (blended / np.linalg.norm(blended)).astype(np.float32)


def main():
    print("\n" + "=" * 65)
    print("  Phase 6A -- Global Identity Resolution Acceptance Tests")
    print("=" * 65)

    # ── Suite 1: Camera Graph & Shortest Paths ──
    print("\n[1] Camera Graph & Shortest Path Computation")
    resolver = GlobalIdentityResolver(camera_graph_path="configs/camera_graph.json")
    check("Resolver initializes", resolver is not None)
    check("Camera graph distances loaded", len(resolver.distances_km) >= 4)

    # Direct neighbors: CAM-001 -> CAM-002 is 6.5 km
    d_12 = resolver.get_distance_km("CAM-001", "CAM-002")
    check("Direct edge distance CAM-001 -> CAM-002 == 6.5 km", d_12 == 6.5, f"dist={d_12}")

    # Same camera distance == 0.0
    d_11 = resolver.get_distance_km("CAM-001", "CAM-001")
    check("Same-camera distance == 0.0 km", d_11 == 0.0)

    # Multi-hop shortest path: CAM-001 to CAM-004
    # Route via CAM-002: 6.5 + 14.2 = 20.7 km; Route via CAM-003: 10.5 + 10.8 = 21.3 km -> shortest is 20.7 km
    d_14 = resolver.get_distance_km("CAM-001", "CAM-004")
    check("Multi-hop shortest path CAM-001 -> CAM-004 == 20.7 km", d_14 is not None and abs(d_14 - 20.7) < 1e-3, f"dist={d_14}")

    # Unknown camera returns None
    d_unk = resolver.get_distance_km("CAM-001", "CAM-999")
    check("Unknown camera transition returns None", d_unk is None)

    # ── Suite 2: Hard Feasibility Filtering ──
    print("\n[2] Stage 1 -- Hard Feasibility Filtering")
    t0 = 10000.0
    emb_car1 = make_unit_embedding(101)

    # Create candidate at CAM-001 at t=10000s
    cand = GlobalVehicleIdentity(
        global_id="GV-000001",
        canonical_plate="MH12AB1234",
        plate_confidence=0.92,
        vehicle_type="car",
        representative_embedding=emb_car1,
        first_seen_ts=t0,
        last_seen_ts=t0,
        first_camera_id="CAM-001",
        last_camera_id="CAM-001",
        camera_trajectory=["CAM-001"],
    )

    # 2.1 Impossible speed: 6.5 km in 10 seconds -> 2340 km/h > 140 km/h
    obs_speeding = VehicleObservation(
        camera_id="CAM-002",
        track_id=1,
        timestamp=t0 + 10.0,
        vehicle_type="car",
        canonical_plate="MH12AB1234",
        plate_confidence=0.95,
        best_reid_embedding=emb_car1,
    )
    feasible, reason, dist, speed = resolver.check_feasibility(cand, obs_speeding)
    check("Hard filter: Impossible speed (2340 km/h) is REJECTED", not feasible)
    check("Reason mentions impossible speed", "impossible transit speed" in reason.lower())

    # 2.2 Negative time delta (observation in the past)
    obs_past = VehicleObservation(
        camera_id="CAM-002",
        track_id=2,
        timestamp=t0 - 100.0,
        vehicle_type="car",
        canonical_plate="MH12AB1234",
    )
    feasible, reason, _, _ = resolver.check_feasibility(cand, obs_past)
    check("Hard filter: Negative time delta is REJECTED", not feasible)
    check("Reason mentions negative time", "negative" in reason.lower())

    # 2.3 Unknown / unreachable camera transition
    obs_unreach = VehicleObservation(
        camera_id="CAM-999",
        track_id=3,
        timestamp=t0 + 600.0,
        vehicle_type="car",
    )
    feasible, reason, _, _ = resolver.check_feasibility(cand, obs_unreach)
    check("Hard filter: Unreachable camera transition is REJECTED", not feasible)

    # 2.4 Vehicle class mismatch: motorcycle vs car/truck
    cand_moto = GlobalVehicleIdentity(
        global_id="GV-000002",
        canonical_plate="MH12AB1234",
        vehicle_type="motorcycle",
        last_seen_ts=t0,
        last_camera_id="CAM-001",
    )
    obs_truck = VehicleObservation(
        camera_id="CAM-002",
        track_id=4,
        timestamp=t0 + 600.0,  # 10 min -> 39 km/h (feasible time)
        vehicle_type="truck",
        canonical_plate="MH12AB1234",  # identical plate spoof
    )
    feasible, reason, _, _ = resolver.check_feasibility(cand_moto, obs_truck)
    check("Hard filter: Class mismatch (motorcycle vs truck) is REJECTED", not feasible)
    check("Reason mentions incompatible class", "incompatible" in reason.lower())

    # 2.5 Same-camera rapid reappearance below 15s interval
    obs_rapid = VehicleObservation(
        camera_id="CAM-001",
        track_id=5,
        timestamp=t0 + 5.0,  # 5s < 15s
        vehicle_type="car",
        canonical_plate="MH12AB1234",
    )
    feasible, reason, _, _ = resolver.check_feasibility(cand, obs_rapid)
    check("Hard filter: Same-camera rapid re-appearance (<15s) is REJECTED", not feasible)

    # 2.6 Same-camera valid reappearance after 45s
    obs_valid_same = VehicleObservation(
        camera_id="CAM-001",
        track_id=6,
        timestamp=t0 + 45.0,  # 45s >= 15s
        vehicle_type="car",
        canonical_plate="MH12AB1234",
    )
    feasible, reason, _, _ = resolver.check_feasibility(cand, obs_valid_same)
    check("Hard filter: Same-camera reappearance (>=15s) is ACCEPTED", feasible)

    # 2.7 Plausible transition: CAM-001 -> CAM-002 (6.5 km in 600s = 39.0 km/h)
    obs_plausible = VehicleObservation(
        camera_id="CAM-002",
        track_id=7,
        timestamp=t0 + 600.0,  # 10 mins
        vehicle_type="car",
        canonical_plate="MH12AB1234",
        plate_confidence=0.90,
        best_reid_embedding=emb_car1,
        crop_quality=150.0,
    )
    feasible, reason, dist, speed = resolver.check_feasibility(cand, obs_plausible)
    check("Hard filter: Plausible transition (39 km/h) is ACCEPTED", feasible)
    check("Plausible transit speed calculated correctly", abs(speed - 39.0) < 0.1, f"speed={speed:.1f} km/h")

    # ── Suite 3: Multi-Modal Soft Scoring & Modality Re-normalization ──
    print("\n[3] Stage 2 -- Multi-Modal Scoring & Dynamic Fusion")
    # 3.1 Exact plate match scoring
    sim_exact = compute_plate_similarity("MH12AB1234", "MH12AB1234")
    check("Exact plate similarity == 1.0000", sim_exact == 1.0)

    # 3.2 Indian ALPR OCR confusion tolerance (0 vs O)
    sim_confuse = compute_plate_similarity("MH12AB1234", "MH12AB123O")
    check("Plate similarity with 0/O confusion > 0.85", sim_confuse >= 0.85, f"sim={sim_confuse:.3f}")

    # 3.3 Dynamic fusion: all modalities present (Plate + ReID + Time)
    score_full, metrics_full = resolver.score_candidate(cand, obs_plausible, dist, speed)
    check("Full modality fused score in [0.0, 1.0]", 0.0 <= score_full <= 1.0)
    check("Full modality score is high for matching vehicle", score_full >= 0.85, f"score={score_full:.3f}")
    check("Active modalities include plate, reid, and time", set(metrics_full["active_modalities"]) == {"plate", "reid", "time"})

    # 3.4 Missing Plate: ReID + Time re-normalization
    obs_no_plate = VehicleObservation(
        camera_id="CAM-002",
        track_id=8,
        timestamp=t0 + 600.0,
        vehicle_type="car",
        canonical_plate=None,
        best_reid_embedding=emb_car1,
        crop_quality=120.0,
    )
    score_no_plate, metrics_no_plate = resolver.score_candidate(cand, obs_no_plate, dist, speed)
    check("Missing plate re-normalizes weights across ReID and Time", set(metrics_no_plate["active_modalities"]) == {"reid", "time"})
    check("ReID-only evidence produces strong fused score", score_no_plate >= 0.80, f"score={score_no_plate:.3f}")

    # 3.5 Missing ReID: Plate + Time re-normalization
    obs_no_reid = VehicleObservation(
        camera_id="CAM-002",
        track_id=9,
        timestamp=t0 + 600.0,
        vehicle_type="car",
        canonical_plate="MH12AB1234",
        plate_confidence=0.92,
        best_reid_embedding=None,
    )
    score_no_reid, metrics_no_reid = resolver.score_candidate(cand, obs_no_reid, dist, speed)
    check("Missing ReID re-normalizes weights across Plate and Time", set(metrics_no_reid["active_modalities"]) == {"plate", "time"})
    check("Plate-only evidence produces strong fused score", score_no_reid >= 0.85, f"score={score_no_reid:.3f}")

    # ── Suite 4: Decision Engine & Margin Requirement ──
    print("\n[4] Decision Engine: MATCH, UNCERTAIN, and NEW Identities")
    res_engine = GlobalIdentityResolver(camera_graph_path="configs/camera_graph.json")

    # 4.1 First observation -> creates initial global identity GV-000001
    obs1 = VehicleObservation(
        camera_id="CAM-001",
        track_id=10,
        timestamp=t0,
        vehicle_type="car",
        canonical_plate="DL01AB9999",
        plate_confidence=0.91,
        best_reid_embedding=emb_car1,
        crop_quality=200.0,
    )
    id1, res1 = res_engine.resolve_observation(obs1)
    check("First observation produces status NEW", res1.status == "NEW")
    check("Assigned global ID is GV-000001", id1.global_id == "GV-000001")
    check("Total registered identities == 1", len(res_engine.identities) == 1)

    # 4.2 Clear matching observation at CAM-002 (feasible transition)
    obs2_match = VehicleObservation(
        camera_id="CAM-002",
        track_id=11,
        timestamp=t0 + 650.0,  # 10.8 mins
        vehicle_type="car",
        canonical_plate="DL01AB9999",
        plate_confidence=0.90,
        best_reid_embedding=emb_car1,
        crop_quality=220.0,
    )
    id2, res2 = res_engine.resolve_observation(obs2_match)
    check("Clear observation produces status MATCH", res2.status == "MATCH")
    check("Matched same global ID GV-000001", id2.global_id == "GV-000001")
    check("Trajectory recorded CAM-001 -> CAM-002", id2.camera_trajectory == ["CAM-001", "CAM-002"])
    check("Sighting count incremented to 2", id2.sighting_count == 2)
    check("Registered identities count remains 1", len(res_engine.identities) == 1)

    # 4.3 Distinct vehicle observation -> creates GV-000002
    emb_car2 = make_unit_embedding(202)
    obs3_distinct = VehicleObservation(
        camera_id="CAM-003",
        track_id=12,
        timestamp=t0 + 700.0,
        vehicle_type="car",
        canonical_plate="KA03MG4321",
        plate_confidence=0.88,
        best_reid_embedding=emb_car2,
    )
    id3, res3 = res_engine.resolve_observation(obs3_distinct)
    check("Distinct vehicle produces status NEW", res3.status == "NEW")
    check("Assigned new global ID GV-000002", id3.global_id == "GV-000002")
    check("Total registered identities == 2", len(res_engine.identities) == 2)

    # 4.4 Margin Requirement Test: Ambiguous Candidates -> UNCERTAIN
    # Setup two candidate vehicles with similar appearance and no legible plates at CAM-003 & CAM-002
    ambig_engine = GlobalIdentityResolver(camera_graph_path="configs/camera_graph.json", min_margin=0.10)
    emb_base = make_unit_embedding(500)
    emb_cand1 = make_similar_embedding(emb_base, similarity=0.90)
    emb_cand2 = make_similar_embedding(emb_base, similarity=0.89)

    # Register Candidate A at CAM-002
    ambig_engine.resolve_observation(VehicleObservation(
        camera_id="CAM-002",
        track_id=20,
        timestamp=t0,
        vehicle_type="car",
        canonical_plate=None,
        best_reid_embedding=emb_cand1,
    ))
    # Register Candidate B at CAM-003
    ambig_engine.resolve_observation(VehicleObservation(
        camera_id="CAM-003",
        track_id=21,
        timestamp=t0,
        vehicle_type="car",
        canonical_plate=None,
        best_reid_embedding=emb_cand2,
    ))

    # Sighting at CAM-004 where both transitions from CAM-002 and CAM-003 are equally feasible
    # CAM-002 -> CAM-004 is 14.2 km; CAM-003 -> CAM-004 is 10.8 km
    # At t = t0 + 1200s (20 mins), speed from CAM-002 is 42.6 km/h, from CAM-003 is 32.4 km/h (both plausible!)
    obs_ambig = VehicleObservation(
        camera_id="CAM-004",
        track_id=22,
        timestamp=t0 + 1200.0,
        vehicle_type="car",
        canonical_plate=None,
        best_reid_embedding=emb_base,
    )
    id_ambig, res_ambig = ambig_engine.resolve_observation(obs_ambig)
    check("Ambiguous close candidates trigger UNCERTAIN status", res_ambig.status == "UNCERTAIN")
    check("Reason mentions ambiguous top candidates", "ambiguous" in res_ambig.reason.lower())
    check("Scores for both candidates recorded in result", len(res_ambig.candidate_scores) >= 2)

    # 4.5 Candidate Window Expiration -> NEW
    exp_engine = GlobalIdentityResolver(camera_graph_path="configs/camera_graph.json", max_candidate_window_sec=3600.0)
    exp_engine.resolve_observation(VehicleObservation(
        camera_id="CAM-001",
        track_id=30,
        timestamp=t0,
        vehicle_type="car",
        canonical_plate="UP32XX0001",
    ))
    # Sighting 4 hours later (> 1 hour window)
    obs_expired = VehicleObservation(
        camera_id="CAM-002",
        track_id=31,
        timestamp=t0 + 14400.0,
        vehicle_type="car",
        canonical_plate="UP32XX0001",
    )
    id_exp, res_exp = exp_engine.resolve_observation(obs_expired)
    check("Expired candidate window produces status NEW", res_exp.status == "NEW")
    check("Reason states no active candidate eligible", "no active candidate" in res_exp.reason.lower())

    # ── Suite 5: Representative Embedding Protection ──
    print("\n[5] Representative Embedding Quality Protection")
    prot_identity = GlobalVehicleIdentity(
        global_id="GV-000099",
        canonical_plate="MH01AA1111",
        vehicle_type="car",
        representative_embedding=emb_car1.copy(),
        first_seen_ts=t0,
        last_seen_ts=t0,
        last_camera_id="CAM-001",
    )
    prot_identity._embedding_pool.append((emb_car1.copy(), 100.0))

    # Degraded crop with quality = 2.0 (below min_crop_quality = 10.0)
    garbage_emb = make_unit_embedding(999)
    obs_degraded = VehicleObservation(
        camera_id="CAM-002",
        track_id=99,
        timestamp=t0 + 600.0,
        vehicle_type="car",
        best_reid_embedding=garbage_emb,
        crop_quality=2.0,  # very low quality
    )
    prot_identity.update_with_observation(obs_degraded, min_crop_quality=10.0)
    check("Degraded embedding rejected from global representative pool", len(prot_identity._embedding_pool) == 1)
    # Cosine similarity with original embedding remains 1.0000
    sim_prot = float(np.dot(prot_identity.representative_embedding, emb_car1))
    check("Global representative embedding remains uncontaminated", abs(sim_prot - 1.0) < 1e-4, f"sim={sim_prot:.5f}")

    # ── Summary ──
    print("\n" + "=" * 65)
    print(f"  Results: {PASS}/{PASS + FAIL} passed, {FAIL} failed")
    print("=" * 65 + "\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
