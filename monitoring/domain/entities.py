from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class SystemHealthReport:
    """Consolidated telemetry representing hardware utilization and OS metrics."""
    overall_status: str  # "HEALTHY", "WARNING", "CRITICAL"
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_used_gb: float
    ram_usage_percent: float
    disk_total_gb: float
    disk_free_gb: float
    disk_usage_percent: float
    cuda_available: bool
    gpu_device_name: Optional[str]
    gpu_vram_total_mb: Optional[float]
    gpu_vram_used_mb: Optional[float]
    gpu_vram_free_mb: Optional[float]
    system_uptime_sec: float
    efficientnet_health: str  # "HEALTHY", "WARNING", "CRITICAL"
    unext_health: str
    gradcam_health: str
    calibration_health: str
    stats_engine_health: str
    longitudinal_health: str
    report_gen_health: str
    pdf_gen_health: str
    sqlite_health: str
    api_health: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationValidationReport:
    """Anatomical segmentation checks (dimensions, regions, pixel values)."""
    is_valid: bool
    warnings: List[str]
    pixel_count: int
    num_components: int
    min_val: int
    max_val: int
    unique_vals: List[int]
    height: int
    width: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConsistencyReport:
    """Clinical correspondence checks between model classification and segmentation."""
    is_consistent: bool
    warnings: List[str]
    primary_class: str
    confidence_score: float
    calibrated: bool
    tumor_area_mm2: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
