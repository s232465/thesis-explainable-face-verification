from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoConfig,
    SegformerForSemanticSegmentation,
)

try:
    from transformers import SegformerImageProcessor
except Exception:
    SegformerImageProcessor = None


@dataclass
class HFParsingOutput:
    label_map: np.ndarray          # (H, W) int64
    id2label: Dict[int, str]
    labels: List[str]              # index = class id


class HFFaceParser:
    """Face parsing using a SegFormer head fine-tuned on CelebAMask-HQ
    (jonathandinu/face-parsing).

    Uses SegformerImageProcessor when available, falls back to
    AutoFeatureExtractor for older transformers versions.
    """

    def __init__(self, model_name: str = "jonathandinu/face-parsing", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        cfg = AutoConfig.from_pretrained(model_name)
        self.id2label = (
            {int(k): v for k, v in cfg.id2label.items()}
            if getattr(cfg, "id2label", None) else {}
        )

        if self.id2label:
            max_id = max(self.id2label.keys())
            self.labels = [self.id2label.get(i, f"class_{i}") for i in range(max_id + 1)]
        else:
            self.labels = []

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

        # Try the proper processor first, fall back to legacy extractor
        self.processor = None
        if SegformerImageProcessor is not None:
            try:
                self.processor = SegformerImageProcessor.from_pretrained(
                    model_name, use_fast=False
                )
            except Exception:
                self.processor = None

        if self.processor is None:
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)

    @torch.inference_mode()
    def parse_pil(self, img: Image.Image) -> HFParsingOutput:
        """Parse a PIL image, returning a label map at original resolution."""
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits  # (1, num_classes, h, w)

        W, H = img.size
        up = torch.nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        label_map = up.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)

        return HFParsingOutput(label_map=label_map, id2label=self.id2label, labels=self.labels)

    def masks_from_label_map(
        self,
        label_map: np.ndarray,
        parts: List[str],
        label_name_to_id: Dict[str, int],
    ) -> Dict[str, np.ndarray]:
        """Extract binary masks for the requested parts."""
        masks: Dict[str, np.ndarray] = {}
        for p in parts:
            if p not in label_name_to_id:
                continue
            pid = label_name_to_id[p]
            masks[p] = (label_map == pid).astype(np.uint8)
        return masks
