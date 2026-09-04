"""
Phase 5 Benchmark -- Vehicle ReID Feature Extraction Latency & Similarity Metrics.

Measures:
1. Single-crop inference latency on CPU (ms / crop).
2. Batch inference throughput on CPU (crops / second).
3. Representative multi-crop aggregation latency (ms / track).
4. Pairwise similarity metrics between tracked vehicles across test videos.
"""

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.reid import VehicleReID, compute_similarity, aggregate_embeddings
from alpr.tracker import VehicleTracker

logging.basicConfig(
    level=logging.WARNING,  # Quiet down logs for clean benchmark output
)


def run_reid_benchmarks():
    print("=" * 70)
    print("  Phase 5 -- Vehicle ReID Benchmark Report")
    print("  (Phase 5A: Baseline Feature Extractor -- ImageNet ResNet-18 on CPU)")
    print("=" * 70)

    reid = VehicleReID(device="cpu")

    # ── 1. Latency Benchmark (Single Crop) ──
    print("\n[1] Single-Crop Feature Extraction Latency")
    dummy_crops = [
        np.random.randint(0, 256, (180, 220, 3), dtype=np.uint8)
        for _ in range(30)
    ]

    # Warmup
    for c in dummy_crops[:5]:
        _ = reid.extract_embedding(c)

    latencies = []
    for c in dummy_crops:
        t0 = time.perf_counter()
        emb = reid.extract_embedding(c)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    latencies = np.array(latencies)
    avg_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print(f"  Samples:    {len(latencies)} crops")
    print(f"  Avg Latency: {avg_lat:.2f} ms")
    print(f"  P50 Latency: {p50_lat:.2f} ms")
    print(f"  P95 Latency: {p95_lat:.2f} ms")
    print(f"  Min / Max:   {min_lat:.2f} ms / {max_lat:.2f} ms")

    # ── 2. Batch Inference Benchmark ──
    print("\n[2] Batch Inference Throughput (Batch Sizes: 1, 4, 8, 16)")
    for b_size in [1, 4, 8, 16]:
        batch = [np.random.randint(0, 256, (180, 220, 3), dtype=np.uint8) for _ in range(b_size)]
        # Warmup
        _ = reid.extract_batch(batch)

        durations = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = reid.extract_batch(batch)
            durations.append(time.perf_counter() - t0)

        avg_dur = np.mean(durations)
        crops_per_sec = b_size / avg_dur
        ms_per_crop = (avg_dur * 1000.0) / b_size
        print(f"  Batch size {b_size:>2}: {ms_per_crop:5.2f} ms/crop | {crops_per_sec:5.1f} crops/sec (total {avg_dur * 1000:5.1f} ms/batch)")

    # ── 3. Multi-Crop Track Aggregation ──
    print("\n[3] Multi-Crop Track Aggregation Overhead")
    embs_5 = [reid.extract_embedding(c) for c in dummy_crops[:5]]
    agg_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = reid.aggregate_embeddings(embs_5)
        agg_times.append((time.perf_counter() - t0) * 1000.0)

    print(f"  Mean aggregation time (5 embeddings): {np.mean(agg_times):.4f} ms")

    # ── 4. Real Video Vehicles Feature Extraction & Similarity ──
    print("\n[4] Real Video Tracks -- Crop Quality & Pairwise Similarity")
    video_path = "inputs/cam01.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        tracker = VehicleTracker(camera_id="BENCH-CAM01")

        frame_idx = 0
        while frame_idx < 60 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            active = tracker.update(frame, frame_number=frame_idx, timestamp=time.time())
            for trk in active:
                state = tracker.active_tracks.get(trk.track_id)
                if state and state.best_vehicle_crop is not None:
                    if len(state.reid_embeddings) < 5:
                        emb = reid.extract_embedding(state.best_vehicle_crop)
                        if emb is not None:
                            state.update_reid(emb, state.best_vehicle_crop, state.best_crop_quality)
            frame_idx += 1

        cap.release()
        tracker.finalize_all()

        tracks_with_reid = [s for s in tracker.finalized_tracks if s.best_reid_embedding is not None]
        print(f"  Processed {frame_idx} frames of {video_path}")
        print(f"  Vehicles tracked with ReID: {len(tracks_with_reid)}")

        for t in tracks_with_reid[:5]:
            plate_info = t.canonical_plate or "NO_PLATE"
            print(f"    Track #{t.track_id:>2} ({t.vehicle_type:<5}): {len(t.crop_history)} crops, {len(t.reid_embeddings)} embeddings, plate={plate_info}")

        if len(tracks_with_reid) >= 2:
            print("\n  Pairwise Cosine Similarity Matrix (Top Tracks):")
            sample_tracks = tracks_with_reid[:min(4, len(tracks_with_reid))]
            header = "         " + "  ".join([f"Trk #{t.track_id:>2}" for t in sample_tracks])
            print(header)
            for t1 in sample_tracks:
                row = [f"Trk #{t1.track_id:>2}: "]
                for t2 in sample_tracks:
                    sim = compute_similarity(t1.best_reid_embedding, t2.best_reid_embedding)
                    row.append(f" {sim:6.3f} ")
                print("".join(row))

    print("\n" + "=" * 70)
    print("  Benchmark Completed Successfully")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_reid_benchmarks()
