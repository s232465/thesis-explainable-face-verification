from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _unwrap(out):
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


def adaface_embedding(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    feat = _unwrap(out)
    feat = F.normalize(feat, p=2, dim=1)
    return feat


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last_conv


@dataclass
class GradCAMResult:
    cam: np.ndarray
    layer_name: str


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or find_last_conv_layer(model)

        self._acts = None
        self._grads = None

        def fwd_hook(_, __, output):
            self._acts = output

        def bwd_hook(_, grad_input, grad_output):
            self._grads = grad_output[0]

        self._h1 = self.target_layer.register_forward_hook(fwd_hook)
        self._h2 = self.target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        self._h1.remove()
        self._h2.remove()

    def cam_from_score(self, score: torch.Tensor, input_hw: Tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self._acts is None or self._grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/grads.")

        acts = self._acts
        grads = self._grads

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_hw, mode="bilinear", align_corners=False)
        cam = cam[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy().astype(np.float32)


def pair_gradcam(
    model: torch.nn.Module,
    xA: torch.Tensor,
    xB: torch.Tensor,
    target_layer: Optional[torch.nn.Module] = None,
) -> Tuple[GradCAMResult, GradCAMResult, float]:
    input_hw = (xA.shape[2], xA.shape[3])
    gc = GradCAM(model, target_layer=target_layer)

    # A
    xA_ = xA.clone().requires_grad_(True)
    eA = adaface_embedding(model, xA_)
    eB = adaface_embedding(model, xB).detach()
    simA = F.cosine_similarity(eA, eB, dim=1)[0]
    camA = gc.cam_from_score(simA, input_hw)

    # B
    xB_ = xB.clone().requires_grad_(True)
    eB2 = adaface_embedding(model, xB_)
    eA2 = adaface_embedding(model, xA).detach()
    simB = F.cosine_similarity(eA2, eB2, dim=1)[0]
    camB = gc.cam_from_score(simB, input_hw)

    with torch.inference_mode():
        eA0 = adaface_embedding(model, xA)
        eB0 = adaface_embedding(model, xB)
        sim0 = float(F.cosine_similarity(eA0, eB0, dim=1)[0].item())

    layer_name = gc.target_layer.__class__.__name__
    gc.close()
    return GradCAMResult(cam=camA, layer_name=layer_name), GradCAMResult(cam=camB, layer_name=layer_name), sim0