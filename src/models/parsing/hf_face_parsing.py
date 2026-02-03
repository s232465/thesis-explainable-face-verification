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

# SegformerImageProcessor exists in modern Transformers. We'll import it safely.
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
    """
    Face parsing with a SegFormer head fine-tuned on CelebAMask-HQ.
    Model: jonathandinu/face-parsing. :contentReference[oaicite:2]{index=2}

    We avoid AutoImageProcessor here because some older checkpoints don't declare
    image_processor_type metadata, which causes AutoImageProcessor to fail.
    Instead we load SegformerImageProcessor directly (or fallback to AutoFeatureExtractor). :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, model_name: str = "jonathandinu/face-parsing", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load config first (gives id2label)
        cfg = AutoConfig.from_pretrained(model_name)
        self.id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()} if getattr(cfg, "id2label", None) else {}

        if self.id2label:
            max_id = max(self.id2label.keys())
            self.labels = [self.id2label.get(i, f"class_{i}") for i in range(max_id + 1)]
        else:
            self.labels = []

        # Load model
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device).eval()

        # Load processor with robust fallbacks
        self.processor = None

        # Preferred: SegformerImageProcessor
        if SegformerImageProcessor is not None:
            try:
                # use_fast=False avoids the "fast processor" warning paths
                self.processor = SegformerImageProcessor.from_pretrained(model_name, use_fast=False)
            except Exception:
                self.processor = None

        # Fallback: AutoFeatureExtractor (legacy but works for older repos)
        if self.processor is None:
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)

    @torch.inference_mode()
    def parse_pil(self, img: Image.Image) -> HFParsingOutput:
        """
        Returns label_map with the same HxW as the input image.
        """
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # (B, num_classes, h, w)

        # Upsample logits to the original image size
        W, H = img.size
        up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        label_map = up.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)

        return HFParsingOutput(label_map=label_map, id2label=self.id2label, labels=self.labels)

    def masks_from_label_map(
        self,
        label_map: np.ndarray,
        parts: List[str],
        label_name_to_id: Dict[str, int],
    ) -> Dict[str, np.ndarray]:
        """
        Returns dict(part_name -> binary mask HxW uint8)
        """
        masks: Dict[str, np.ndarray] = {}
        for p in parts:
            if p not in label_name_to_id:
                continue
            pid = label_name_to_id[p]
            masks[p] = (label_map == pid).astype(np.uint8)
        return masks
