from alpr.detector import (
    load_detector,
    detect_plates,
    resolve_device,
    ensure_model,
    VehicleDetector,
    VehicleDetection,
    VEHICLE_CLASS_MAP,
)
from alpr.ocr import (
    load_ocr,
    recognize_plate,
    is_probable_indian_plate,
    PlateQualityGate,
    assess_plate_quality,
)
from alpr.tracker import (
    VehicleTracker,
    VehicleTrackState,
    ActiveVehicleTrack,
    PlateRead,
)
from alpr.anpr import VehicleANPR
from alpr.reid import (
    VehicleReID,
    extract_embedding,
    compute_similarity,
    aggregate_embeddings,
)
from alpr.identity import (
    GlobalVehicleIdentity,
    VehicleObservation,
    IdentityMatchResult,
    GlobalIdentityResolver,
    compute_plate_similarity,
)
from alpr.database import (
    init_db,
    save_global_identity,
    record_vehicle_observation,
    get_global_vehicle,
    get_global_vehicle_by_plate,
    get_vehicle_trajectory,
    get_all_global_vehicles,
)
from alpr.trajectory import (
    haversine_distance_km,
    TrajectoryNode,
    TrajectorySegment,
    VehicleTrajectory,
    TrajectoryReconstructor,
    reconstruct_trajectory,
    reconstruct_trajectory_by_plate,
    list_all_trajectories,
)
from alpr.analytics import (
    CorridorAnalytics,
    TripODRecord,
    NetworkAnalyticsReport,
    CorridorAnalyticsEngine,
    analyze_network_traffic,
)
from alpr.congestion import (
    CameraNodeFlowMetrics,
    CorridorCongestionMetrics,
    NetworkCongestionReport,
    TrafficCongestionEngine,
    analyze_traffic_congestion,
    classify_los_proxy,
    compute_interval_union_duration,
)
