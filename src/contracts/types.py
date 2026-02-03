from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Literal, Any

Decision = Literal["match", "non-match"]
ExplainMethod = Literal["integrated_gradients", "occlusion", "gradcam", "none"]

@dataclass
class FacePair:
    imgA: str
    imgB: str
    pair_id: str = ""
    label: Optional[int] = None  # 1 genuine, 0 impostor, None unknown

@dataclass
class AlignedFace:
    path: str                    # saved aligned crop path
    size: List[int]              # [H, W]
    detector: str = "mtcnn"
    landmarks: Optional[List[List[float]]] = None  # if available

@dataclass
class AlignedPair:
    A: AlignedFace
    B: AlignedFace

@dataclass
class ParseResult:
    label_map_path: str          # saved as .png or .npy
    parts: List[str]             # e.g. ["left_eye","right_eye","nose",...]
    masks_path: str              # saved masks as .npz
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

@dataclass
class XMapResult:
    method: ExplainMethod
    heatmapA_path: str           # .npy or image
    heatmapB_path: str
    signed: bool = True
    normalize: str = "minmax"

@dataclass
class PartSignal:
    support: float               # supports predicted decision
    conflict: float              # contradicts predicted decision
    area_frac: float             # part area / face area
    flags: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PartSignals:
    parts: List[str]
    A: Dict[str, PartSignal]     # key: part name
    B: Dict[str, PartSignal]

@dataclass
class GlobalMetrics:
    confidence: Optional[float] = None
    conflict_ratio: Optional[float] = None
    evidence_concentration: Optional[float] = None
    reliability: Optional[float] = None
    flags: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OperatorExplanation:
    bullets: List[str] = field(default_factory=list)
    caution: Optional[str] = None
    overlays: Dict[str, str] = field(default_factory=dict)  # name -> image path

@dataclass
class PipelineResult:
    pair: FacePair
    aligned: Optional[AlignedPair] = None
    parsing: Optional[Dict[str, ParseResult]] = None  # {"A":..., "B":...}
    fv: Optional[FVResult] = None
    xmap: Optional[XMapResult] = None
    part_signals: Optional[PartSignals] = None
    metrics: Optional[GlobalMetrics] = None
    operator: Optional[OperatorExplanation] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
