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
from alpr.gis import (
    get_los_color,
    build_network_geojson,
    generate_city_traffic_map,
)
from alpr.service import (
    DashboardService,
    get_dashboard_service,
)
from alpr.alerts import (
    AlertRecord,
    AlertEngine,
    evaluate_blacklist_match,
    evaluate_kinematic_anomalies,
    evaluate_topological_anomalies,
    evaluate_identity_uncertainty,
    evaluate_behavioral_anomalies,
    ALERT_BLACKLIST_EXACT,
    ALERT_BLACKLIST_FUZZY,
    ALERT_VELOCITY_ANOMALY,
    ALERT_TEMPORAL_INVERSION,
    ALERT_TOPOLOGY_VIOLATION,
    ALERT_IDENTITY_UNCERTAIN,
    ALERT_EXCESSIVE_DWELL,
    ALERT_RAPID_LOOPING,
)
from alpr.database import (
    record_security_alert,
    get_security_alerts,
    get_security_alert_by_id,
    acknowledge_security_alert,
    get_security_alerts_summary,
    add_enriched_blacklist_entry,
    get_enriched_blacklist,
    get_thread_connection,
    update_camera_status,
    get_camera_statuses,
    record_security_alert_obj,
    execute_with_retry,
)
from workers.orchestrator import (
    PipelineOrchestrator,
    CameraTelemetry,
)

