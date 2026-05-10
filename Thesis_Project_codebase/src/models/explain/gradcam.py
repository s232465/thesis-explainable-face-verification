from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Callable

import numpy as np
import torch
import torch.nn.functional as F


def _unwrap_feat_norm(out):
    """Extract (feat_tensor, norm_or_None) from model output, handling
    single tensors, (feat, norm) tuples, and nested structures."""
    norm = None
    if (isinstance(out, (tuple, list)) and len(out) >= 2
            and torch.is_tensor(out[0]) and torch.is_tensor(out[1])):
        return out[0], out[1]

    while isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"Cannot extract tensor from model output: {type(out)}")
    return out, norm


def adaface_embedding(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Default embedding function for Grad-CAM. Returns (B,512) L2-normalised."""
    out = model(x)
    feat_raw, _ = _unwrap_feat_norm(out)
    return F.normalize(feat_raw, p=2, dim=1)


def find_deepest_spatial_conv_layer(
    model: torch.nn.Module,
    sample_x: torch.Tensor,
    *,
    debug: bool = False,
) -> tuple[str, torch.nn.Module]:
    """Find the last Conv2d that still produces spatial feature maps (H>1, W>1)."""
    conv_modules: List[tuple[str, torch.nn.Module]] = [
        (name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)
    ]
    if not conv_modules:
        raise RuntimeError("No Conv2d layer found.")

    shapes: Dict[str, tuple[int, int]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_m, _inp, out):
            if torch.is_tensor(out) and out.ndim == 4:
                shapes[name] = (int(out.shape[-2]), int(out.shape[-1]))
        return hook

    for name, m in conv_modules:
        hooks.append(m.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            model(sample_x)
    finally:
        for h in hooks:
            h.remove()

    # Pick the deepest conv with spatial dims > 1x1
    best_name, best_layer = conv_modules[-1]
    for name, layer in conv_modules:
        hw = shapes.get(name)
        if hw is not None and hw[0] > 1 and hw[1] > 1:
            best_name, best_layer = name, layer

    if debug:
        print(f"[GradCAM] selected: {best_name}, spatial: {shapes.get(best_name)}")

    return best_name, best_layer


@dataclass
class GradCAMResult:
    cam: np.ndarray
    layer_name: str


class GradCAM:
    def __init__(self, model, target_layer, target_layer_name, *, debug=False):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.target_layer_name = target_layer_name
        self.debug = debug

        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        def fwd_hook(_, __, output):
            self._acts = output
            if self.debug and torch.is_tensor(output):
                print("[GradCAM] acts:", tuple(output.shape))

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
            raise RuntimeError("Hooks did not capture activations/gradients.")

        acts = self._acts
        grads = self._grads

        if acts.ndim != 4 or acts.shape[-1] <= 1 or acts.shape[-2] <= 1:
            raise RuntimeError(
                f"No spatial map at selected layer: shape={tuple(acts.shape)}"
            )

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
    target_layer_name: Optional[str] = None,
    *,
    debug: bool = False,
    embed_fn: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[GradCAMResult, float]:
    """Compute Grad-CAM for the probe image (B) w.r.t. cosine similarity.

    Only the probe map is produced because the reference (A) is shown clean
    to the operator. The forward pass order matters: reference embedding is
    computed first (detached) so the hook state holds xB activations when
    cam_from_score runs.

    Returns (cam_result, baseline_similarity).
    """
    input_hw = (xB.shape[2], xB.shape[3])

    if embed_fn is None:
        embed_fn = adaface_embedding

    if target_layer is None:
        layer_name, layer = find_deepest_spatial_conv_layer(model, xB, debug=debug)
    else:
        layer = target_layer
        layer_name = target_layer_name or target_layer.__class__.__name__

    gc = GradCAM(model, target_layer=layer, target_layer_name=layer_name, debug=debug)

    try:
        # Reference first (detached), then probe with gradients
        eA_ref = embed_fn(model, xA).detach()
        xB_ = xB.clone().requires_grad_(True)
        eB_probe = embed_fn(model, xB_)
        sim_score = F.cosine_similarity(eA_ref, eB_probe, dim=1)[0]
        camB = gc.cam_from_score(sim_score, input_hw)

        with torch.no_grad():
            eA0 = embed_fn(model, xA)
            eB0 = embed_fn(model, xB)
            sim0 = float(F.cosine_similarity(eA0, eB0, dim=1)[0].item())

        return GradCAMResult(cam=camB, layer_name=layer_name), sim0

    finally:
        gc.close()
