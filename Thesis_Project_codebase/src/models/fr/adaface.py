from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# AdaFace third-party repo must be cloned into third_party/adaface
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAFACE_ROOT = _REPO_ROOT / "third_party" / "adaface"
if not _ADAFACE_ROOT.exists():
    raise FileNotFoundError(f"Expected AdaFace repo at: {_ADAFACE_ROOT}")

if str(_ADAFACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADAFACE_ROOT))

import net  # type: ignore  # noqa: E402


@dataclass
class AdaFaceConfig:
    architecture: str = "ir_50"
    ckpt_path: str = "pretrained/adaface_ir50_ms1mv2.ckpt"
    device: str = "cuda"


class AdaFaceEmbedder:
    def __init__(self, cfg: AdaFaceConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        ckpt = (_REPO_ROOT / cfg.ckpt_path).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(
                f"AdaFace checkpoint not found: {ckpt}\n"
                f"Place your checkpoint at: {_REPO_ROOT / 'pretrained'}"
            )

        self.model = net.build_model(cfg.architecture).to(self.device).eval()

        try:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt), map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise ValueError(f"Unexpected checkpoint format type: {type(state)}")

        # Strip "model." or "module." prefixes that appear in some checkpoint formats
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v
                          for k, v in state_dict.items() if k.startswith("model.")}
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v
                          for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[AdaFace] missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"[AdaFace] unexpected keys ({len(unexpected)}): {unexpected[:10]}")

        self.last_quality_score: Optional[float] = None

    def preprocess_rgb_uint8(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert 112x112 RGB uint8 array to model input tensor."""
        if rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB, got dtype={rgb.dtype}")
        if rgb.shape != (112, 112, 3):
            raise ValueError(f"Expected (112,112,3), got {rgb.shape}")

        bgr = rgb[..., ::-1].copy()
        x = torch.from_numpy(bgr).float() / 255.0
        x = (x - 0.5) / 0.5
        x = x.permute(2, 0, 1).unsqueeze(0)
        return x.to(self.device)

    @torch.inference_mode()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return L2-normalised 512-d embedding."""
        out = self.model(x)
        self.last_quality_score = None
        feat_raw = None

        # AdaFace returns (features, norm) — norm encodes quality
        if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[0]):
            feat_raw = out[0]
            norm = out[1]
            if torch.is_tensor(norm):
                try:
                    self.last_quality_score = float(norm.detach().reshape(-1)[0].cpu().item())
                except Exception:
                    self.last_quality_score = None

        if feat_raw is None:
            while isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                raise TypeError(f"Unexpected model output after unwrapping: {type(out)}")
            feat_raw = out

        feat = F.normalize(feat_raw, p=2, dim=1)
        return feat.squeeze(0).detach().cpu()
