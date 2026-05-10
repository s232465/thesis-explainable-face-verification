from __future__ import annotations

from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


HeatmapNormalize = Literal["minmax", "none"]
Decision = Literal["match", "non-match"]
ExplainMethod = Literal["integrated_gradients", "occlusion", "gradcam", "none"]
OcclusionMethod = Literal["gray", "blur"]


@dataclass
class FacePair:
    imgA: str
    imgB: str
    pair_id: str = ""
    label: Optional[int] = None  # 1=genuine, 0=impostor


@dataclass
class AlignedFace:
    path: str
    size: List[int]  # [H, W]
    detector: str = "mtcnn"
    landmarks: Optional[List[List[float]]] = None


@dataclass
class AlignedPair:
    A: AlignedFace
    B: AlignedFace


@dataclass
class ParseResult:
    label_map_path: str
    parts: List[str]
    masks_path: str
    parsing_model: str
    parsing_confidence: Optional[float] = None


@dataclass
class FVResult:
    model: str
    embeddingA_path: str
    embeddingB_path: str
    similarity_cosine: float

    threshold: float
    decision: Decision

    device: str
    backend: str = "pytorch"

    # AdaFace norm (higher = better quality)
    qualityA: Optional[float] = None
    qualityB: Optional[float] = None

    # Threshold calibration info
    threshold_calibrated: bool = False
    threshold_far_target: Optional[float] = None
    threshold_dataset: Optional[str] = None
    threshold_source: Optional[str] = None
    threshold_notes: Optional[str] = None

    tar_at_far: Optional[float] = None
    actual_far: Optional[float] = None

    decision_margin: Optional[float] = None
    confidence_label: Optional[str] = None


@dataclass
class XMapResult:
    method: ExplainMethod

    heatmapA_path: Optional[str] = None
    heatmapB_path: Optional[str] = None

    signed: bool = False
    normalize: HeatmapNormalize = "minmax"

    layer: Optional[str] = None
    notes: Optional[str] = None

    region_overlays: Dict[str, str] = field(default_factory=dict)


@dataclass
class PartSignal:
    """Per-part occlusion signal for one method (gray or blur).

    A = reference image, B = probe image.
    delta = sim_occluded - sim0.

    For a match decision, support = max(0, -delta) and conflict = max(0, +delta).
    For non-match, the signs are flipped.

    supportB/conflictB are the primary signals used for the operator overlay
    since the probe is the image being explained. The aggregated support/conflict
    fields are the mean of valid side-specific values.
    """
    validA: bool
    validB: bool

    deltaA: Optional[float] = None
    deltaB: Optional[float] = None
    deltaMean: Optional[float] = None

    supportA:  Optional[float] = None
    conflictA: Optional[float] = None
    supportB:  Optional[float] = None
    conflictB: Optional[float] = None

    support:  Optional[float] = None
    conflict: Optional[float] = None

    areaA_pixels: int = 0
    areaB_pixels: int = 0
    areaA_frac: float = 0.0
    areaB_frac: float = 0.0

    flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PartSignalsMethod:
    method: OcclusionMethod
    parts: List[str]
    per_part: Dict[str, PartSignal]


@dataclass
class PartSignals:
    """Container for both occlusion methods, keyed by name (e.g. methods["gray"])."""
    methods: Dict[str, PartSignalsMethod]


@dataclass
class GlobalMetricsMethod:
    sim0: float
    decision: Decision
    support_total: float
    conflict_total: float
    evidence_balance: float
    top_3_support_parts: List[Tuple[str, float]] = field(default_factory=list)
    top_3_conflict_parts: List[Tuple[str, float]] = field(default_factory=list)
    n_parts_used: int = 0
    flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalMetrics:
    """Per-method metrics plus cross-method agreement (Spearman rho between
    gray and blur deltaMean values). High rho means both methods agree on
    which regions matter most."""
    by_method: Dict[str, GlobalMetricsMethod] = field(default_factory=dict)

    agreement_gray_vs_blur_spearman_rho: Optional[float] = None
    agreement_n_parts: int = 0

    flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorExplanation:
    bullets: List[str] = field(default_factory=list)
    caution: Optional[str] = None
    overlays: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    pair: FacePair
    aligned: Optional[AlignedPair] = None
    parsing: Optional[Dict[str, ParseResult]] = None
    fv: Optional[FVResult] = None
    xmap: Optional[XMapResult] = None
    part_signals: Optional[PartSignals] = None
    metrics: Optional[GlobalMetrics] = None
    operator: Optional[OperatorExplanation] = None

    def to_dict(self) -> Dict[str, Any]:
        def _convert(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj
        return _convert(self)
