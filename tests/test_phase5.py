"""
Phase 5 -- Acceptance tests for Vehicle Re-Identification (ReID).

Tests Phase 5A baseline visual feature extraction, L2 normalization, cosine similarity,
batch inference, multi-crop track embedding aggregation, and CameraWorker integration.
"""

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.reid import (
    VehicleReID,
    extract_embedding,
    compute_similarity,
    aggregate_embeddings,
)
from alpr.tracker import (
    VehicleTracker,
    VehicleTrackState,
    ActiveVehicleTrack,
)
from workers.camera_worker import CameraWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("test_phase5")

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


def main():
    print("\n" + "=" * 60)
    print("  Phase 5 -- Vehicle Re-Identification (ReID) Acceptance Tests")
    print("  (Phase 5A: Baseline Feature Extractor -- ImageNet ResNet-18)")
    print("=" * 60)

    # -- Suite 1: ReID Initialization & Model Properties --
    print("\n[1] VehicleReID Initialization & Model Properties")
    reid = VehicleReID(device="cpu")
    check("VehicleReID instantiates on CPU", reid is not None)
    check("Embedding dimension is 512", reid.embedding_dim == 512, f"dim={reid.embedding_dim}")
    check("Backbone is resnet18", reid.backbone_name == "resnet18")
    check("Input size is configured", len(reid.img_size) == 2)

    # -- Suite 2: Crop Preprocessing & Edge Cases --
    print("\n[2] Crop Preprocessing & Edge Cases")
    check("None crop returns None tensor", reid.preprocess_crop(None) is None)
    check("Empty numpy array returns None tensor", reid.preprocess_crop(np.empty((0, 0, 3), dtype=np.uint8)) is None)
    check("Tiny crop (<10px) returns None tensor", reid.preprocess_crop(np.zeros((5, 5, 3), dtype=np.uint8)) is None)
    check("None crop returns None embedding", reid.extract_embedding(None) is None)
    check("Empty crop returns None embedding", reid.extract_embedding(np.empty((0, 0, 3), dtype=np.uint8)) is None)
    check("Tiny crop returns None embedding", reid.extract_embedding(np.zeros((8, 8, 3), dtype=np.uint8)) is None)

    # Valid synthetic crop
    sample_crop = np.full((120, 80, 3), 128, dtype=np.uint8)
    t = reid.preprocess_crop(sample_crop)
    check("Valid crop preprocesses to 4D tensor", t is not None and t.dim() == 4)
    check("Preprocessed tensor shape matches BCHW", t.shape == (1, 3, reid.img_size[0], reid.img_size[1]))

    # -- Suite 3: L2 Normalization & Extraction Properties --
    print("\n[3] L2 Normalization & Extraction Properties")
    # Load real image from inputs/
    real_img = cv2.imread("inputs/1.jpg")
    check("Real sample image inputs/1.jpg exists", real_img is not None)

    # Take a subcrop representing a vehicle area
    h, w = real_img.shape[:2]
    crop1 = real_img[int(h * 0.3):int(h * 0.7), int(w * 0.2):int(w * 0.6)]
    crop2 = real_img[int(h * 0.4):int(h * 0.8), int(w * 0.5):int(w * 0.9)]

    emb1 = reid.extract_embedding(crop1)
    check("extract_embedding returns array", emb1 is not None and isinstance(emb1, np.ndarray))
    check("Embedding shape is (512,)", emb1.shape == (512,), f"shape={emb1.shape}")
    check("Embedding dtype is float32", emb1.dtype == np.float32)

    norm1 = float(np.linalg.norm(emb1))
    check("Embedding L2 norm == 1.0000", abs(norm1 - 1.0) < 1e-4, f"norm={norm1:.6f}")

    emb2 = reid.extract_embedding(crop2)
    norm2 = float(np.linalg.norm(emb2))
    check("Second crop L2 norm == 1.0000", abs(norm2 - 1.0) < 1e-4, f"norm={norm2:.6f}")

    # Convenience function check
    emb_conv = extract_embedding(crop1)
    check("Convenience function extract_embedding works", emb_conv is not None and emb_conv.shape == (512,))

    # -- Suite 4: Cosine Similarity Properties --
    print("\n[4] Cosine Similarity Mathematical Properties")
    sim_self = reid.compute_similarity(emb1, emb1)
    check("Self-similarity equals exactly 1.0000", abs(sim_self - 1.0) < 1e-4, f"sim={sim_self:.5f}")

    sim_12 = reid.compute_similarity(emb1, emb2)
    check("Pairwise similarity is in [-1.0, 1.0]", -1.0 <= sim_12 <= 1.0, f"sim={sim_12:.4f}")

    # Symmetry: sim(A, B) == sim(B, A)
    sim_21 = reid.compute_similarity(emb2, emb1)
    check("Cosine similarity is symmetric", abs(sim_12 - sim_21) < 1e-5)

    # Edge cases
    check("Similarity with None is 0.0", reid.compute_similarity(emb1, None) == 0.0)
    check("Similarity with zero vector is 0.0", reid.compute_similarity(emb1, np.zeros(512)) == 0.0)

    # Identical crop produces ~1.0
    crop1_copy = crop1.copy()
    emb1_copy = reid.extract_embedding(crop1_copy)
    sim_ident = reid.compute_similarity(emb1, emb1_copy)
    check("Identical crop similarity ~= 1.0000", abs(sim_ident - 1.0) < 1e-4, f"sim={sim_ident:.5f}")

    # Convenience function check
    sim_conv = compute_similarity(emb1, emb2)
    check("Convenience compute_similarity matches", abs(sim_conv - sim_12) < 1e-5)

    # -- Suite 5: Batch Inference Consistency --
    print("\n[5] Batch Inference Consistency")
    crops_batch = [crop1, crop2, None, crop1_copy]
    batch_embs = reid.extract_batch(crops_batch)
    check("Batch returns same length as input", len(batch_embs) == len(crops_batch))
    check("Batch handles None crop at correct position", batch_embs[2] is None)
    check("Batch embedding 0 matches single extraction", np.allclose(batch_embs[0], emb1, atol=1e-4))
    check("Batch embedding 1 matches single extraction", np.allclose(batch_embs[1], emb2, atol=1e-4))
    check("Empty batch returns empty list", reid.extract_batch([]) == [])

    # -- Suite 6: Multi-Crop Track Aggregation --
    print("\n[6] Multi-Crop Track Aggregation")
    agg = reid.aggregate_embeddings([emb1, emb2, emb1_copy])
    check("aggregate_embeddings returns array", agg is not None)
    check("Aggregated embedding shape is (512,)", agg.shape == (512,))
    agg_norm = float(np.linalg.norm(agg))
    check("Aggregated embedding L2 norm == 1.0000", abs(agg_norm - 1.0) < 1e-4, f"norm={agg_norm:.6f}")
    check("Empty list aggregate returns None", reid.aggregate_embeddings([]) is None)
    check("All-None list aggregate returns None", reid.aggregate_embeddings([None, None]) is None)

    # Convenience function check
    agg_conv = aggregate_embeddings([emb1, emb2])
    check("Convenience aggregate_embeddings works", agg_conv is not None)

    # -- Suite 7: VehicleTrackState Integration --
    print("\n[7] VehicleTrackState ReID Integration")
    track = VehicleTrackState(
        track_id=1,
        camera_id="CAM-001",
        vehicle_type="car",
    )
    check("reid_embeddings initially empty", len(track.reid_embeddings) == 0)
    check("crop_history initially empty", len(track.crop_history) == 0)
    check("best_reid_embedding initially None", track.best_reid_embedding is None)

    # Add first ReID observation
    track.update_reid(emb1, crop1, quality=100.0)
    check("reid_embeddings records observation", len(track.reid_embeddings) == 1)
    check("crop_history stores crop", len(track.crop_history) == 1)
    check("best_reid_embedding populated", track.best_reid_embedding is not None)
    check("best_reid_embedding norm == 1.0", abs(float(np.linalg.norm(track.best_reid_embedding)) - 1.0) < 1e-4)

    # Add additional observations to test bounding cap (max 5)
    for i in range(7):
        dummy_crop = np.full((50, 50, 3), i * 30, dtype=np.uint8)
        dummy_emb = reid.extract_embedding(dummy_crop)
        if dummy_emb is not None:
            track.update_reid(dummy_emb, dummy_crop, quality=float(50 + i * 10), max_crops=5)

    check("crop_history capped at max_crops (5)", len(track.crop_history) <= 5, f"len={len(track.crop_history)}")
    check("reid_embeddings bounded to match crop_history", len(track.reid_embeddings) == len(track.crop_history))
    check("best_reid_embedding remains normalized", abs(float(np.linalg.norm(track.best_reid_embedding)) - 1.0) < 1e-4)

    # -- Suite 8: CameraWorker Orchestration with ReID --
    print("\n[8] CameraWorker Orchestration with ReID")
    tracker = VehicleTracker(camera_id="TEST-REID-CAM")
    worker = CameraWorker(
        camera_id="TEST-REID-WORKER",
        source="inputs/1.jpg",
        tracker=tracker,
        reid=reid,
        reid_every_n=1,
    )
    check("Worker initializes with tracker and reid", worker.tracker is not None and worker.reid is not None)

    # Run worker on test image
    worker.start()

    check("Worker processed frame", worker.frames_processed >= 1)
    check("ReID extractions recorded", worker.reid_extractions >= 1, f"extractions={worker.reid_extractions}")

    # Check tracks created and embedded
    embedded_tracks = [
        s for s in worker.tracker.finalized_tracks
        if s.best_reid_embedding is not None
    ]
    check("Finalized tracks have best_reid_embedding", len(embedded_tracks) >= 1, f"embedded_count={len(embedded_tracks)}")

    if embedded_tracks:
        sample_state = embedded_tracks[0]
        check("Sample track has 512-D embedding", sample_state.best_reid_embedding.shape == (512,))
        check("Sample track embedding is L2-normalized", abs(float(np.linalg.norm(sample_state.best_reid_embedding)) - 1.0) < 1e-4)
        check("Sample track has crop_history", len(sample_state.crop_history) >= 1)

    # -- Summary --
    print("\n" + "=" * 60)
    print(f"  Results: {PASS}/{PASS + FAIL} passed, {FAIL} failed")
    print("=" * 60 + "\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
