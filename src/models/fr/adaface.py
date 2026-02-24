from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --- Make the vendored AdaFace repo importable ---
# Your tree shows: third_party/adaface/
# We append that folder so `import net` works (AdaFace's own inference does `import net`).
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../<repo>/
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
    """
    AdaFace inference wrapper.

    Expectations (AdaFace):
      - aligned crop size: 112x112
      - channel order: BGR
      - normalization: (x/255 - 0.5) / 0.5  -> [-1, 1]
    """

    def __init__(self, cfg: AdaFaceConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        ckpt = (_REPO_ROOT / cfg.ckpt_path).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(
                f"AdaFace checkpoint not found: {ckpt}\n"
                f"Place your checkpoint at: {_REPO_ROOT / 'pretrained'}"
            )

        # Build model from AdaFace code
        self.model = net.build_model(cfg.architecture).to(self.device).eval()

        # Load checkpoint (PyTorch Lightning .ckpt often needs weights_only=False)
        try:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        except TypeError:
            # For older torch versions without weights_only
            state = torch.load(str(ckpt), map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise ValueError(f"Unexpected checkpoint format type: {type(state)}")

        # Strip 'model.' prefix if present (common in lightning checkpoints)
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}

        # Strip 'module.' prefix if present (DataParallel)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[AdaFace] Warning: missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[AdaFace] Warning: unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    def preprocess_rgb_uint8(self, rgb: np.ndarray) -> torch.Tensor:
        """
        rgb: (112,112,3) uint8 in RGB order
        returns: (1,3,112,112) float tensor, BGR order, normalized to [-1,1]
        """
        if rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB input, got dtype={rgb.dtype}")
        if rgb.shape != (112, 112, 3):
            raise ValueError(f"Expected shape (112,112,3), got {rgb.shape}")

        # RGB -> BGR
        bgr = rgb[..., ::-1].copy()

        x = torch.from_numpy(bgr).float() / 255.0  # [0,1]
        x = (x - 0.5) / 0.5                        # [-1,1]
        x = x.permute(2, 0, 1).unsqueeze(0)        # (1,3,112,112)
        return x.to(self.device)

    @torch.inference_mode()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1,3,112,112) float, BGR, normalized
        returns: (512,) L2-normalized embedding on CPU
        """
        out = self.model(x)

        # Robustly unwrap nested tuple/list outputs until we get a tensor
        while isinstance(out, (tuple, list)):
            out = out[0]

        if not torch.is_tensor(out):
            raise TypeError(f"Unexpected model output type after unwrapping: {type(out)}")

        feat = F.normalize(out, p=2, dim=1)
        return feat.squeeze(0).detach().cpu()