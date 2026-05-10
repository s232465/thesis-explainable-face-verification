"""
fidelity_eval.py
----------------
Probe-side fidelity evaluation for the semantic occlusion explanations.

Evaluates whether the pipeline's region importance rankings generalise
across perturbation methods (gray vs blur), using cross-perturbation
deletion metrics and GradCAM triangulation.

Usage:
    python scripts/fidelity_eval.py \
        --manifest  assets/gorilla/manifest.csv \
        --results   results/gorilla \
        --outdir    results/fidelity \
        --device    cuda \
        --seed      42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Constants

PARTS = ["skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "hair"]

GRAY_FILL      = 128
TOP_K          = 3
N_RANDOM       = 30        # used only when exhaustive enumeration is too large
MIN_PART_PIXELS = 30
BOOTSTRAP_N    = 5000

# With 8 parts and TOP_K=3 the exhaustive combination count is C(8,3)=56.
# We enumerate exhaustively whenever the pool has <= EXHAUST_THRESHOLD combos.
EXHAUST_THRESHOLD = 100

BLUE  = "#185FA5"
CORAL = "#D85A30"
GREEN = "#1D9E75"
AMBER = "#BA7517"
MGRAY = "#888780"
BORDER = "#B4B2A9"
TEXT  = "#2C2C2A"
# I/O

def load_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(f)]
def load_result(pair_dir: Path) -> Dict[str, Any]:
    p = pair_dir / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"result.json not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)
def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
def load_masks(path: Path) -> Dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k].astype(np.uint8) for k in npz.files}
def load_cam(path: Path) -> np.ndarray:
    cam = np.load(path).astype(np.float32)
    cam -= cam.min()
    d = cam.max()
    if d > 1e-8:
        cam /= d
    return cam
# Occlusion

def occlude_gray(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask.astype(bool)] = GRAY_FILL
    return out
def occlude_blur(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
        blurred = cv2.GaussianBlur(rgb, (11, 11), 0)
    except ImportError:
        from PIL import ImageFilter
        blurred = np.array(
            Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=5)),
            dtype=np.uint8,
        )
    out = rgb.copy()
    out[mask.astype(bool)] = blurred[mask.astype(bool)]
    return out
def apply_occlusion(rgb: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    if method == "gray":
        return occlude_gray(rgb, mask)
    if method == "blur":
        return occlude_blur(rgb, mask)
    raise ValueError(f"Unknown occlusion method: {method}")
# AdaFace

_ADA = None
def get_ada(ada_ckpt: str, device: str):
    global _ADA
    if _ADA is None:
        root = Path(__file__).resolve().parents[1]
        ada_dir = root / "src" / "models" / "fr"
        if str(ada_dir) not in sys.path:
            sys.path.insert(0, str(ada_dir))
        from adaface import AdaFaceEmbedder, AdaFaceConfig
        _ADA = AdaFaceEmbedder(
            AdaFaceConfig(architecture="ir_50", ckpt_path=ada_ckpt, device=device)
        )
    return _ADA
def embed(rgb: np.ndarray, ada) -> torch.Tensor:
    return ada.embed(ada.preprocess_rgb_uint8(rgb)).view(-1).detach()
def cosine(e1: torch.Tensor, e2: torch.Tensor) -> float:
    return float(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))[0].item())
# Helpers

def normalize_label(x: Optional[str]) -> str:
    s = (x or "").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "match": "match", "mated": "match", "same": "match",
        "same-person": "match", "genuine": "match",
        "nonmatch": "non-match", "non-match": "non-match",
        "nonmated": "non-match", "non-mated": "non-match",
        "different": "non-match", "different-person": "non-match",
        "impostor": "non-match",
    }
    return aliases.get(s, s)
def decision_sign(decision: str) -> int:
    """
    Sign for decision-support score.
    match:     support = -raw_delta  (removal of helpful region drops similarity)
    non-match: support = +raw_delta  (removal of helpful region raises similarity)
    """
    return -1 if normalize_label(decision) == "match" else +1
def bootstrap_mean_ci(
    vals: List[float], rng: random.Random, n_boot: int = BOOTSTRAP_N
) -> Optional[Tuple[float, float]]:
    if not vals:
        return None
    arr = np.array(vals, dtype=np.float32)
    means = [float(arr[[rng.randrange(len(arr)) for _ in range(len(arr))]].mean())
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
# Spearman rho

def spearman_rho(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None

    def rankdata(a: List[float]) -> List[float]:
        idx = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[idx[j + 1]] == a[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    den = (
        sum((rx[i] - mx) ** 2 for i in range(len(rx))) *
        sum((ry[i] - my) ** 2 for i in range(len(ry)))
    ) ** 0.5
    return float(num / den) if den != 0 else None
# GradCAM part scores

def gradcam_part_scores(
    cam: np.ndarray, masks: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for p in PARTS:
        mask = masks.get(p)
        if mask is None:
            continue
        mb = mask.astype(bool)
        area = int(mb.sum())
        if area < MIN_PART_PIXELS:
            continue
        vals = cam[mb]
        out[p] = {
            "mean": float(vals.mean()),
            "sum": float(vals.sum()),
            "area_pixels": area,
        }
    return out
# Probe-side occlusion importance

def probe_occ_importance(
    rgbB: np.ndarray,
    masksB: Dict[str, np.ndarray],
    method: str,
    eA: torch.Tensor,
    s0: float,
    decision: str,
    ada,
) -> Dict[str, Dict[str, float]]:
    """Per-part probe-side occlusion importance. Reference A stays fixed,
    each part of probe B is occluded and the similarity shift is measured."""
    out: Dict[str, Dict[str, float]] = {}
    sign = decision_sign(decision)

    with torch.no_grad():
        for p in PARTS:
            mask = masksB.get(p)
            if mask is None:
                continue
            area = int((mask > 0).sum())
            if area < MIN_PART_PIXELS:
                continue

            oB = apply_occlusion(rgbB, (mask > 0).astype(np.uint8), method)
            eB_occ = embed(oB, ada)
            s_occ = cosine(eA, eB_occ)

            raw_delta     = float(s_occ - s0)
            abs_delta     = abs(raw_delta)
            support_delta = float(sign * raw_delta)
            norm_abs      = abs_delta / area if area > 0 else 0.0

            out[p] = {
                "s_occ":        s_occ,
                "raw_delta":    raw_delta,
                "abs_delta":    abs_delta,
                "support_delta": support_delta,
                "area_pixels":  area,
                "norm_abs_delta": norm_abs,
            }
    return out
# Sampling helper — exhaustive when pool is small

def sample_groups(
    pool: List[str],
    k: int,
    n: int,
    rng: random.Random,
) -> List[Tuple[str, ...]]:
    """
    Returns up to n groups of size k drawn from pool.
    Uses exhaustive enumeration when C(pool, k) <= EXHAUST_THRESHOLD,
    otherwise random sampling without replacement.
    """
    if len(pool) < k:
        return []
    all_combos = list(combinations(pool, k))
    if len(all_combos) <= EXHAUST_THRESHOLD:
        return all_combos
    seen = set()
    result = []
    attempts = 0
    while len(result) < n and attempts < n * 10:
        attempts += 1
        c = tuple(sorted(rng.sample(pool, k)))
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result
# Cross-perturbation deletion (absolute / support)

def cross_deletion(
    rank_scores: Dict[str, float],
    obs_scores: Dict[str, float],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """Cross-perturbation deletion test: rank parts by one method's scores,
    observe the other method's scores. Compares top-k vs bottom-k vs random baseline."""
    valid = sorted(
        [p for p in PARTS if p in rank_scores and p in obs_scores],
        key=lambda p: rank_scores[p],
        reverse=True,
    )

    if len(valid) < TOP_K * 2:
        return None

    top_k    = valid[:TOP_K]
    bottom_k = valid[-TOP_K:]

    mean_top    = float(np.mean([obs_scores[p] for p in top_k]))
    mean_bottom = float(np.mean([obs_scores[p] for p in bottom_k]))
    ratio       = mean_top / (mean_bottom + 1e-9)

    # Random baseline from the full valid part set
    # (with 8 parts and TOP_K=3, the middle pool is too small, so we use all parts)
    random_groups = sample_groups(valid, TOP_K, N_RANDOM, rng)
    rand_means = [
        float(np.mean([obs_scores[p] for p in grp]))
        for grp in random_groups
    ]
    mean_random  = float(np.mean(rand_means)) if rand_means else mean_bottom
    random_ratio = mean_random / (mean_bottom + 1e-9)

    return {
        "top_k_parts":     top_k,
        "bottom_k_parts":  bottom_k,
        "mean_top_k":      mean_top,
        "mean_bottom_k":   mean_bottom,
        "mean_random_k":   mean_random,
        "top_vs_bottom":   ratio,
        "top_minus_bottom": mean_top - mean_bottom,
        "random_vs_bottom": random_ratio,
        "top_beats_random": mean_top > mean_random,
        "n_random_groups":  len(random_groups),
        "random_pool": "full_valid_set",  # documents the fallback
    }
# Signed directional deletion

def signed_directional_deletion(
    gray_raw: Dict[str, float],
    blur_raw: Dict[str, float],
    decision: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """Signed directional deletion — tests whether the green/red overlay assignments
    generalise across perturbation methods.

    For mated pairs: selects parts where gray_delta < 0 (similarity-contributing),
    ranks by magnitude, and checks if blur agrees directionally.

    For non-mated pairs: selects parts where gray_delta > 0 (dissimilarity-contributing),
    same cross-check logic.

    More specific than the absolute deletion because it conditions on the expected
    sign direction and penalises cross-method disagreement.
    """
    d = normalize_label(decision)
    is_match = (d == "match")

    directional: Dict[str, float] = {}
    for p in PARTS:
        if p not in gray_raw or p not in blur_raw:
            continue
        g = gray_raw[p]
        if is_match and g < 0:
            directional[p] = abs(g)      # rank by similarity contribution
        elif not is_match and g > 0:
            directional[p] = g           # rank by dissimilarity contribution

    n_dir = len(directional)

    # Observation function — rewards correct direction, penalises wrong direction
    def obs(p: str) -> float:
        b = blur_raw[p]
        return max(0.0, -b) if is_match else max(0.0, b)

    if n_dir < TOP_K * 2:
        # Not enough directional parts to run the deletion test
        # Check all-same-sign case separately
        all_raw = list(gray_raw.values())
        all_positive = all(v >= 0 for v in all_raw)
        all_negative = all(v <= 0 for v in all_raw)
        flag = None
        if is_match and all_negative:
            flag = "all_negative_match"     # all green: clean match, no red warning needed
        elif not is_match and all_positive:
            flag = "all_positive_nonmatch"  # unexpected: would mean all parts support match
        elif not is_match and all_negative:
            flag = "all_negative_nonmatch"  # diffuse evidence: no specific red parts
        return {
            "n_directional": n_dir,
            "insufficient": True,
            "flag": flag,
            "note": (
                f"Only {n_dir} directionally appropriate parts found "
                f"(need {TOP_K * 2} for deletion test). "
                "This may indicate diffuse evidence or skin-dominant signal."
            ),
        }

    ranked   = sorted(directional.items(), key=lambda x: x[1], reverse=True)
    top_k    = [p for p, _ in ranked[:TOP_K]]
    bottom_k = [p for p, _ in ranked[-TOP_K:]]

    obs_top    = [obs(p) for p in top_k]
    obs_bottom = [obs(p) for p in bottom_k]
    mean_top    = float(np.mean(obs_top))
    mean_bottom = float(np.mean(obs_bottom))

    # Random baseline from directional pool
    rand_pool = [p for p in directional if p not in set(top_k + bottom_k)]
    if len(rand_pool) < TOP_K:
        rand_pool = list(directional.keys())

    rand_groups = sample_groups(rand_pool, TOP_K, N_RANDOM, rng)
    rand_means  = [float(np.mean([obs(p) for p in grp])) for grp in rand_groups]
    mean_random = float(np.mean(rand_means)) if rand_means else mean_bottom

    return {
        "n_directional":    n_dir,
        "insufficient":     False,
        "flag":             None,
        "top_k_parts":      top_k,
        "bottom_k_parts":   bottom_k,
        "mean_top_k":       mean_top,
        "mean_bottom_k":    mean_bottom,
        "mean_random_k":    mean_random,
        "top_minus_bottom": float(mean_top - mean_bottom),
        "top_vs_bottom":    float(mean_top / (mean_bottom + 1e-9)),
        "top_beats_random": mean_top > mean_random,
        "n_random_groups":  len(rand_groups),
    }
# Per-pair evaluation

def evaluate_pair(
    pair_dir: Path,
    result: Dict[str, Any],
    ada,
    rng: random.Random,
) -> Dict[str, Any]:
    fv       = result.get("fv") or {}
    decision = normalize_label(fv.get("decision", "non-match"))
    margin   = fv.get("decision_margin")
    conf     = fv.get("confidence_label")
    s0_stored = fv.get("similarity_cosine")

    aligned = result.get("aligned") or {}

    def resolve(side: str, fname: str) -> Path:
        p = Path((aligned.get(side) or {}).get("path", ""))
        return p if p.exists() else pair_dir / fname

    pathA = resolve("A", "aligned_A.png")
    pathB = resolve("B", "aligned_B.png")
    if not pathA.exists() or not pathB.exists():
        return {"error": "aligned images not found"}

    parsing = result.get("parsing") or {}

    def mpath(side: str, fname: str) -> Path:
        p = Path((parsing.get(side) or {}).get("masks_path", ""))
        return p if p.exists() else pair_dir / fname

    mpB = mpath("B", "parse_masks_B.npz")
    if not mpB.exists():
        return {"error": "probe masks not found"}

    cam_path = pair_dir / "gradcam_B.npy"

    rgbA   = load_rgb(pathA)
    rgbB   = load_rgb(pathB)
    masksB = load_masks(mpB)

    with torch.no_grad():
        eA = embed(rgbA, ada)
        eB = embed(rgbB, ada)
        s0 = cosine(eA, eB)

    # s0 consistency check
    s0_warning = None
    if s0_stored is not None and abs(s0 - float(s0_stored)) > 1e-4:
        s0_warning = f"s0 mismatch: recomputed={s0:.6f} stored={float(s0_stored):.6f}"

    gray = probe_occ_importance(rgbB, masksB, "gray", eA, s0, decision, ada)
    blur = probe_occ_importance(rgbB, masksB, "blur", eA, s0, decision, ada)

    abs_gray  = {p: v["abs_delta"]     for p, v in gray.items()}
    abs_blur  = {p: v["abs_delta"]     for p, v in blur.items()}
    sup_gray  = {p: v["support_delta"] for p, v in gray.items()}
    sup_blur  = {p: v["support_delta"] for p, v in blur.items()}
    raw_gray  = {p: v["raw_delta"]     for p, v in gray.items()}
    raw_blur  = {p: v["raw_delta"]     for p, v in blur.items()}

    # Area-normalised abs_delta for fair GradCAM mean comparison
    norm_gray = {p: v["norm_abs_delta"] for p, v in gray.items()}

    # Primary: absolute influence cross-perturbation deletion
    abs_del_A = cross_deletion(abs_gray, abs_blur, rng)   # gray rank -> blur observe
    abs_del_B = cross_deletion(abs_blur, abs_gray, rng)   # blur rank -> gray observe

    # Secondary: decision-support cross-perturbation deletion
    # Flag pairs where all support_deltas share the same sign
    sup_vals_gray = list(sup_gray.values())
    sup_all_same_sign = (
        all(v >= 0 for v in sup_vals_gray) or
        all(v <= 0 for v in sup_vals_gray)
    ) if sup_vals_gray else False

    sup_del_A = cross_deletion(sup_gray, sup_blur, rng)
    sup_del_B = cross_deletion(sup_blur, sup_gray, rng)

    # Signed directional deletion 
    dir_del = signed_directional_deletion(raw_gray, raw_blur, decision, rng)

    # Correct top-sup reporting: only name a part when its support_delta is positive
    top_abs_gray = max(abs_gray, key=abs_gray.get) if abs_gray else None
    top_abs_blur = max(abs_blur, key=abs_blur.get) if abs_blur else None

    sup_gray_pos = {p: v for p, v in sup_gray.items() if v > 0}
    sup_blur_pos = {p: v for p, v in sup_blur.items() if v > 0}
    top_sup_gray = max(sup_gray_pos, key=sup_gray_pos.get) if sup_gray_pos else None
    top_sup_blur = max(sup_blur_pos, key=sup_blur_pos.get) if sup_blur_pos else None

    # GradCAM triangulation (non-blocking diagnostic)
    rho_gc_mean_gray  = None
    rho_gc_sum_gray   = None
    rho_gc_mean_blur  = None
    rho_gc_norm_gray  = None   # area-normalised comparison
    top_gc_part       = None
    gradcam_error     = None
    gc_scores: Dict[str, Dict[str, float]] = {}

    if cam_path.exists():
        try:
            cam = load_cam(cam_path)
            gc_scores = gradcam_part_scores(cam, masksB)
            if gc_scores:
                top_gc_part = max(gc_scores, key=lambda p: gc_scores[p]["mean"])

            common_gray = [p for p in PARTS if p in gc_scores and p in abs_gray]
            common_blur = [p for p in PARTS if p in gc_scores and p in abs_blur]

            if len(common_gray) >= 3:
                rho_gc_mean_gray = spearman_rho(
                    [gc_scores[p]["mean"] for p in common_gray],
                    [abs_gray[p] for p in common_gray],
                )
                rho_gc_sum_gray = spearman_rho(
                    [gc_scores[p]["sum"] for p in common_gray],
                    [abs_gray[p] for p in common_gray],
                )
                # Area-normalised comparison: fairer for GradCAM mean vs occlusion
                rho_gc_norm_gray = spearman_rho(
                    [gc_scores[p]["mean"] for p in common_gray],
                    [norm_gray[p] for p in common_gray],
                )
            else:
                gradcam_error = f"too few common parts ({len(common_gray)})"

            if len(common_blur) >= 3:
                rho_gc_mean_blur = spearman_rho(
                    [gc_scores[p]["mean"] for p in common_blur],
                    [abs_blur[p] for p in common_blur],
                )
        except Exception as e:
            gradcam_error = f"GradCAM failed: {e}"
    else:
        gradcam_error = "gradcam_B.npy not found"

    return {
        "s0":              s0,
        "s0_warning":      s0_warning,
        "decision":        decision,
        "decision_margin": margin,
        "confidence_label": conf,

        "abs_gray":  abs_gray,
        "abs_blur":  abs_blur,
        "sup_gray":  sup_gray,
        "sup_blur":  sup_blur,
        "raw_gray":  raw_gray,
        "raw_blur":  raw_blur,
        "norm_gray": norm_gray,

        "abs_del_A": abs_del_A,
        "abs_del_B": abs_del_B,
        "sup_del_A": sup_del_A,
        "sup_del_B": sup_del_B,
        "sup_all_same_sign": sup_all_same_sign,

        "dir_del": dir_del,          # signed directional deletion

        "rho_gc_mean_gray":  rho_gc_mean_gray,
        "rho_gc_sum_gray":   rho_gc_sum_gray,
        "rho_gc_mean_blur":  rho_gc_mean_blur,
        "rho_gc_norm_gray":  rho_gc_norm_gray,
        "gc_scores":         gc_scores,
        "top_gc_part":       top_gc_part,
        "top_abs_gray":      top_abs_gray,
        "top_abs_blur":      top_abs_blur,
        "top_sup_gray":      top_sup_gray,
        "top_sup_blur":      top_sup_blur,
        "n_parts_abs":       len(abs_gray),

        "gradcam_error": gradcam_error,
        "error": None,
    }
# Summary helpers

def stat_block(vals: List[float], label: str) -> List[str]:
    """Report with rho < 0 distinguished from 0 <= rho < 0.5."""
    if not vals:
        return [f"  {label}: N/A"]
    n_neg  = sum(1 for v in vals if v < 0)
    n_low  = sum(1 for v in vals if 0 <= v < 0.5)
    n_mod  = sum(1 for v in vals if 0.5 <= v < 0.7)
    n_good = sum(1 for v in vals if v >= 0.7)
    n = len(vals)
    return [
        f"  {label}",
        f"    n                   : {n}",
        f"    mean                : {np.mean(vals):.4f}",
        f"    median              : {np.median(vals):.4f}",
        f"    std                 : {np.std(vals):.4f}",
        f"    >= 0.7  (good)      : {n_good}/{n} ({n_good/n*100:.1f}%)",
        f"    0.5–0.7 (moderate)  : {n_mod}/{n} ({n_mod/n*100:.1f}%)",
        f"    0–0.5   (weak)      : {n_low}/{n} ({n_low/n*100:.1f}%)",
        f"    < 0     (opposite)  : {n_neg}/{n} ({n_neg/n*100:.1f}%)",
        f"    [rho < 0 = methods rank regions in OPPOSITE order]",
    ]
def deletion_block(
    dels: List[Dict[str, Any]], label: str, rng: random.Random
) -> List[str]:
    if not dels:
        return [f"  {label}: N/A"]
    tmb  = [d["top_minus_bottom"] for d in dels]
    tvb  = [d["top_vs_bottom"]    for d in dels]
    rvb  = [d["random_vs_bottom"] for d in dels]
    beat = sum(1 for t, r in zip(tvb, rvb) if t > r)
    ci   = bootstrap_mean_ci(tmb, rng)
    lines = [
        f"  {label}",
        f"    mean top-minus-bottom    : {np.mean(tmb):.4f}",
        f"    median top-minus-bottom  : {np.median(tmb):.4f}",
        f"    mean top/bottom ratio    : {np.mean(tvb):.3f}",
        f"    mean random/bottom ratio : {np.mean(rvb):.3f}",
        f"    top beats random         : {beat}/{len(tvb)} ({beat/len(tvb)*100:.1f}%)",
        f"    pairs with ratio > 1.0   : {sum(1 for v in tvb if v > 1)}/{len(tvb)}",
    ]
    if ci is not None:
        lines.append(
            f"    bootstrap 95% CI (mean top-minus-bottom): [{ci[0]:.4f}, {ci[1]:.4f}]"
        )
    return lines
def directional_block(
    dir_dels: List[Dict[str, Any]], label: str, rng: random.Random
) -> List[str]:
    if not dir_dels:
        return [f"  {label}: N/A"]

    sufficient = [d for d in dir_dels if not d.get("insufficient")]
    insufficient = [d for d in dir_dels if d.get("insufficient")]
    flags = {}
    for d in insufficient:
        f = d.get("flag") or "other"
        flags[f] = flags.get(f, 0) + 1

    lines = [
        f"  {label}",
        f"    n total                  : {len(dir_dels)}",
        f"    n sufficient (ran test)  : {len(sufficient)}",
        f"    n insufficient           : {len(insufficient)}",
    ]
    for f, cnt in flags.items():
        lines.append(f"      {f}: {cnt}")

    if sufficient:
        tmb  = [d["top_minus_bottom"] for d in sufficient]
        beat = sum(1 for d in sufficient if d.get("top_beats_random"))
        ci   = bootstrap_mean_ci(tmb, rng)
        lines += [
            f"    mean top-minus-bottom    : {np.mean(tmb):.4f}",
            f"    median top-minus-bottom  : {np.median(tmb):.4f}",
            f"    top beats random         : {beat}/{len(sufficient)} ({beat/len(sufficient)*100:.1f}%)",
        ]
        if ci:
            lines.append(
                f"    bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
            )
    return lines
def write_summary(
    all_results: List[Dict[str, Any]],
    outdir: Path,
    rng: random.Random,
) -> None:
    ok   = [r for r in all_results if r.get("error") is None]
    errs = [r for r in all_results if r.get("error")]

    abs_A = [r["abs_del_A"] for r in ok if r.get("abs_del_A")]
    abs_B = [r["abs_del_B"] for r in ok if r.get("abs_del_B")]
    sup_A = [r["sup_del_A"] for r in ok if r.get("sup_del_A")]
    sup_B = [r["sup_del_B"] for r in ok if r.get("sup_del_B")]
    dir_all = [r["dir_del"] for r in ok if r.get("dir_del") is not None]

    # Correctness-conditioned breakdown
    match_correct    = [r for r in ok if r.get("ground_truth") == "match"
                        and r.get("decision") == "match"]
    match_incorrect  = [r for r in ok if r.get("ground_truth") == "match"
                        and r.get("decision") != "match"]
    nm_correct       = [r for r in ok if r.get("ground_truth") == "non-match"
                        and r.get("decision") == "non-match"]
    nm_incorrect     = [r for r in ok if r.get("ground_truth") == "non-match"
                        and r.get("decision") == "match"]

    rhos_gray = [r["rho_gc_mean_gray"]  for r in ok if r.get("rho_gc_mean_gray")  is not None]
    rhos_sum  = [r["rho_gc_sum_gray"]   for r in ok if r.get("rho_gc_sum_gray")   is not None]
    rhos_blur = [r["rho_gc_mean_blur"]  for r in ok if r.get("rho_gc_mean_blur")  is not None]
    rhos_norm = [r["rho_gc_norm_gray"]  for r in ok if r.get("rho_gc_norm_gray")  is not None]

    same_sign_count = sum(1 for r in ok if r.get("sup_all_same_sign"))
    s0_warnings = [r for r in ok if r.get("s0_warning")]

    lines = [
        "=" * 72,
        "FIDELITY EVALUATION V3 — PROBE-SIDE SEMANTIC OCCLUSION",
        "=" * 72,
        "",
        "PRIMARY METRIC — Cross-perturbation deletion on absolute influence",
        f"  TOP_K={TOP_K}, N_RANDOM={N_RANDOM}",
        "  Config A: rank by gray |delta|, observe blur |delta|",
        "  Config B: rank by blur |delta|, observe gray |delta|",
        "  Random baseline: drawn from FULL valid part set (with 8 parts and",
        "  TOP_K=3 the middle pool has only 2 parts, always below TOP_K).",
        "  Exhaustive enumeration used when pool has <= 100 combinations.",
        "",
        *deletion_block(abs_A, "Config A (gray rank -> blur observe)", rng),
        "",
        *deletion_block(abs_B, "Config B (blur rank -> gray observe)", rng),
        "",
        "SECONDARY METRIC A — Signed directional deletion",
        "  Conditions on sign direction before cross-perturbation test.",
        "  Mated:     ranks green parts (gray_delta < 0) by |gray_delta|,",
        "             observes max(0, -blur_delta) — rewards directional agreement.",
        "  Non-mated: ranks red parts (gray_delta > 0) by gray_delta,",
        "             observes max(0, blur_delta) — rewards directional agreement.",
        "  Directly validates the green/red overlay logic.",
        "  Insufficient = fewer than 2*TOP_K directionally correct parts found",
        "  (common for diffuse-evidence pairs).",
        "",
        *directional_block(dir_all, "All pairs", rng),
        "",
        "SECONDARY METRIC B — Decision-support cross-perturbation deletion",
        f"  Pairs with all-same-sign support_delta (ambiguous): {same_sign_count}/{len(ok)}",
        "  [These pairs produce a positive top-minus-bottom by definition —",
        "   interpret their contribution to the aggregate with caution.]",
        "",
        *deletion_block(sup_A, "Support Config A (gray rank -> blur observe)", rng),
        "",
        *deletion_block(sup_B, "Support Config B (blur rank -> gray observe)", rng),
        "",
        "CORRECTNESS-CONDITIONED BREAKDOWN (primary metric — abs Config A)",
        f"  mated correct    : n={len(match_correct)}",
    ]

    def cond_tmb(subset: List[Dict]) -> str:
        vals = [r["abs_del_A"]["top_minus_bottom"]
                for r in subset if r.get("abs_del_A")]
        return f"{np.mean(vals):.4f}" if vals else "N/A"

    lines += [
        f"    mean top-minus-bottom: {cond_tmb(match_correct)}",
        f"  mated incorrect  : n={len(match_incorrect)}",
        f"    mean top-minus-bottom: {cond_tmb(match_incorrect)}",
        f"  non-match correct: n={len(nm_correct)}",
        f"    mean top-minus-bottom: {cond_tmb(nm_correct)}",
        f"  non-match incorrect: n={len(nm_incorrect)}",
        f"    mean top-minus-bottom: {cond_tmb(nm_incorrect)}",
        "",
        "TRIANGULATION ANALYSIS — GradCAM rank agreement",
        "  Non-blocking diagnostic.  Low rho is expected and informative:",
        "  GradCAM mean is size-normalised; occlusion abs_delta is size-sensitive.",
        "  The mean/sum rho divergence diagnoses area-inflation effects.",
        "  The area-normalised comparison is the methodologically fair version.",
        "",
        *stat_block(rhos_gray, "GradCAM mean vs gray abs_delta [size-mismatch]"),
        "",
        *stat_block(rhos_norm, "GradCAM mean vs gray norm_abs_delta [fair comparison]"),
        "",
        *stat_block(rhos_sum,  "GradCAM sum vs gray abs_delta [size-sensitive, diagnostic]"),
        "",
        *stat_block(rhos_blur, "GradCAM mean vs blur abs_delta"),
        "",
    ]

    if rhos_gray and rhos_norm:
        diff = float(np.mean(rhos_norm)) - float(np.mean(rhos_gray))
        lines += [
            f"  mean/norm rho difference (norm - unnorm): {diff:+.4f}",
            f"  [positive = normalisation improves GradCAM agreement]",
            "",
        ]

    if s0_warnings:
        lines += [
            f"s0 consistency warnings: {len(s0_warnings)} pair(s)",
            *[f"  {r.get('pair_id','?')}: {r['s0_warning']}" for r in s0_warnings[:10]],
            "",
        ]

    gc_fail = [r for r in ok if r.get("gradcam_error")]
    if gc_fail:
        lines += [
            f"GradCAM non-fatal issues: {len(gc_fail)} pair(s)",
            *[f"  {r.get('pair_id','?')}: {r.get('gradcam_error')}" for r in gc_fail[:25]],
            "",
        ]

    if errs:
        lines += [
            f"Primary evaluation errors: {len(errs)} pair(s)",
            *[f"  {r.get('pair_id','?')}: {r.get('error')}" for r in errs[:25]],
            "",
        ]

    lines += ["=" * 72, ""]
    txt = "\n".join(lines)
    (outdir / "fidelity_summary.txt").write_text(txt, encoding="utf-8")
    print(f"[Fidelity] Saved: {outdir / 'fidelity_summary.txt'}")
    print("\n" + txt)
# CSV

def save_csv(all_results: List[Dict[str, Any]], outdir: Path) -> None:
    fields = [
        "pair_id", "category", "ground_truth",
        "s0", "decision", "decision_margin", "confidence_label",
        # Absolute deletion
        "abs_A_top_minus_bottom", "abs_A_top_vs_bottom",
        "abs_A_random_vs_bottom", "abs_A_top_beats_random",
        "abs_B_top_minus_bottom", "abs_B_top_vs_bottom",
        "abs_B_random_vs_bottom", "abs_B_top_beats_random",
        # Decision-support deletion
        "sup_A_top_minus_bottom", "sup_A_top_vs_bottom",
        "sup_B_top_minus_bottom", "sup_B_top_vs_bottom",
        "sup_all_same_sign",
        # Signed directional deletion
        "dir_n_directional", "dir_insufficient", "dir_flag",
        "dir_top_minus_bottom", "dir_top_vs_bottom", "dir_top_beats_random",
        # GradCAM triangulation
        "rho_gc_mean_gray", "rho_gc_norm_gray",
        "rho_gc_sum_gray",  "rho_gc_mean_blur",
        # Parts
        "n_parts_abs", "top_gc_part",
        "top_abs_gray", "top_abs_blur",
        "top_sup_gray", "top_sup_blur",
        # Diagnostics
        "s0_warning", "gradcam_error", "error",
    ]

    def f4(v: Any) -> str:
        return f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "")

    rows = []
    for r in all_results:
        absA = r.get("abs_del_A") or {}
        absB = r.get("abs_del_B") or {}
        supA = r.get("sup_del_A") or {}
        supB = r.get("sup_del_B") or {}
        dd   = r.get("dir_del") or {}

        rows.append({
            "pair_id":        r.get("pair_id", ""),
            "category":       r.get("category", ""),
            "ground_truth":   r.get("ground_truth", ""),
            "s0":             f4(r.get("s0")),
            "decision":       r.get("decision", ""),
            "decision_margin": f4(r.get("decision_margin")),
            "confidence_label": r.get("confidence_label", ""),

            "abs_A_top_minus_bottom": f4(absA.get("top_minus_bottom")),
            "abs_A_top_vs_bottom":   f4(absA.get("top_vs_bottom")),
            "abs_A_random_vs_bottom": f4(absA.get("random_vs_bottom")),
            "abs_A_top_beats_random": str(absA.get("top_beats_random", "")),

            "abs_B_top_minus_bottom": f4(absB.get("top_minus_bottom")),
            "abs_B_top_vs_bottom":   f4(absB.get("top_vs_bottom")),
            "abs_B_random_vs_bottom": f4(absB.get("random_vs_bottom")),
            "abs_B_top_beats_random": str(absB.get("top_beats_random", "")),

            "sup_A_top_minus_bottom": f4(supA.get("top_minus_bottom")),
            "sup_A_top_vs_bottom":    f4(supA.get("top_vs_bottom")),
            "sup_B_top_minus_bottom": f4(supB.get("top_minus_bottom")),
            "sup_B_top_vs_bottom":    f4(supB.get("top_vs_bottom")),
            "sup_all_same_sign":      str(r.get("sup_all_same_sign", "")),

            "dir_n_directional":   str(dd.get("n_directional", "")),
            "dir_insufficient":    str(dd.get("insufficient", "")),
            "dir_flag":            str(dd.get("flag") or ""),
            "dir_top_minus_bottom": f4(dd.get("top_minus_bottom")),
            "dir_top_vs_bottom":    f4(dd.get("top_vs_bottom")),
            "dir_top_beats_random": str(dd.get("top_beats_random", "")),

            "rho_gc_mean_gray":  f4(r.get("rho_gc_mean_gray")),
            "rho_gc_norm_gray":  f4(r.get("rho_gc_norm_gray")),
            "rho_gc_sum_gray":   f4(r.get("rho_gc_sum_gray")),
            "rho_gc_mean_blur":  f4(r.get("rho_gc_mean_blur")),

            "n_parts_abs":  str(r.get("n_parts_abs", "")),
            "top_gc_part":  str(r.get("top_gc_part") or ""),
            "top_abs_gray": str(r.get("top_abs_gray") or ""),
            "top_abs_blur": str(r.get("top_abs_blur") or ""),
            "top_sup_gray": str(r.get("top_sup_gray") or ""),
            "top_sup_blur": str(r.get("top_sup_blur") or ""),

            "s0_warning":    str(r.get("s0_warning") or ""),
            "gradcam_error": str(r.get("gradcam_error") or ""),
            "error":         str(r.get("error") or ""),
        })

    out = outdir / "fidelity_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[Fidelity] Saved: {out}")
# Plots

def make_plots(all_results: List[Dict[str, Any]], outdir: Path) -> None:
    ok = [r for r in all_results if r.get("error") is None]

    abs_A = [r["abs_del_A"]["top_minus_bottom"] for r in ok if r.get("abs_del_A")]
    abs_B = [r["abs_del_B"]["top_minus_bottom"] for r in ok if r.get("abs_del_B")]
    dir_suf = [r["dir_del"]["top_minus_bottom"]
               for r in ok
               if r.get("dir_del") and not r["dir_del"].get("insufficient")]
    rhos_gray = [r["rho_gc_mean_gray"] for r in ok if r.get("rho_gc_mean_gray") is not None]
    rhos_norm = [r["rho_gc_norm_gray"] for r in ok if r.get("rho_gc_norm_gray") is not None]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
    ax1, ax2, ax3, ax4 = axes

    # Panel 1: Absolute cross-perturbation deletion
    all_vals = abs_A + abs_B + [0.0]
    bins1 = np.linspace(min(all_vals), max(all_vals + [0.05]), 14)
    ax1.hist(abs_A, bins=bins1, color=BLUE,  alpha=0.85,
             label=f"gray→blur (n={len(abs_A)})", edgecolor="white", linewidth=0.5)
    ax1.hist(abs_B, bins=bins1, color=CORAL, alpha=0.75,
             label=f"blur→gray (n={len(abs_B)})", edgecolor="white", linewidth=0.5)
    ax1.axvline(0.0, color="#444", lw=1.2, ls=":")
    ax1.set_title("Absolute cross-perturbation deletion", color=TEXT, fontsize=10)
    ax1.set_xlabel("Top-minus-bottom", color=MGRAY, fontsize=9)
    ax1.set_ylabel("Pairs", color=MGRAY, fontsize=9)
    ax1.legend(fontsize=7, framealpha=0.85)

    # Panel 2: Signed directional deletion
    bins2 = np.linspace(min(dir_suf + [0.0]), max(dir_suf + [0.05]), 12) if dir_suf else [0, 0.05]
    ax2.hist(dir_suf, bins=bins2, color=GREEN, alpha=0.85,
             label=f"sufficient pairs (n={len(dir_suf)})", edgecolor="white", linewidth=0.5)
    ax2.axvline(0.0, color="#444", lw=1.2, ls=":")
    ax2.set_title("Signed directional deletion", color=TEXT, fontsize=10)
    ax2.set_xlabel("Top-minus-bottom", color=MGRAY, fontsize=9)
    ax2.set_ylabel("Pairs", color=MGRAY, fontsize=9)
    ax2.legend(fontsize=7, framealpha=0.85)

    # Panel 3: GradCAM unnormalised vs normalised
    bins3 = np.linspace(-1.0, 1.0, 14)
    ax3.hist(rhos_gray, bins=bins3, color=BLUE, alpha=0.7,
             label=f"unnorm (n={len(rhos_gray)})", edgecolor="white", linewidth=0.5)
    ax3.hist(rhos_norm, bins=bins3, color=CORAL, alpha=0.7,
             label=f"area-norm (n={len(rhos_norm)})", edgecolor="white", linewidth=0.5)
    ax3.axvline(0.0, color="#444", lw=1.2, ls=":")
    ax3.axvline(0.5, color=AMBER, lw=1, ls=":")
    ax3.axvline(0.7, color=GREEN, lw=1, ls=":")
    ax3.set_title("GradCAM mean vs gray |delta|", color=TEXT, fontsize=10)
    ax3.set_xlabel("Spearman rho", color=MGRAY, fontsize=9)
    ax3.set_ylabel("Pairs", color=MGRAY, fontsize=9)
    ax3.legend(fontsize=7, framealpha=0.85)

    # Panel 4: Correctness-conditioned top-minus-bottom
    correct_vals   = [r["abs_del_A"]["top_minus_bottom"] for r in ok
                      if r.get("abs_del_A")
                      and r.get("ground_truth") == r.get("decision")]
    incorrect_vals = [r["abs_del_A"]["top_minus_bottom"] for r in ok
                      if r.get("abs_del_A")
                      and r.get("ground_truth") != r.get("decision")]

    bins4 = np.linspace(
        min(correct_vals + incorrect_vals + [0.0]),
        max(correct_vals + incorrect_vals + [0.05]), 12
    )
    if correct_vals:
        ax4.hist(correct_vals, bins=bins4, color=GREEN, alpha=0.75,
                 label=f"correct (n={len(correct_vals)})", edgecolor="white", linewidth=0.5)
    if incorrect_vals:
        ax4.hist(incorrect_vals, bins=bins4, color=CORAL, alpha=0.75,
                 label=f"incorrect (n={len(incorrect_vals)})", edgecolor="white", linewidth=0.5)
    ax4.axvline(0.0, color="#444", lw=1.2, ls=":")
    ax4.set_title("Correctness-conditioned (abs Config A)", color=TEXT, fontsize=10)
    ax4.set_xlabel("Top-minus-bottom", color=MGRAY, fontsize=9)
    ax4.set_ylabel("Pairs", color=MGRAY, fontsize=9)
    ax4.legend(fontsize=7, framealpha=0.85)

    for ax in axes:
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(BORDER)
        ax.tick_params(colors=MGRAY, labelsize=8)
        ax.set_facecolor("white")

    fig.suptitle(
        "Fidelity evaluation v3 — probe-side semantic occlusion",
        fontsize=11, color=TEXT
    )
    fig.tight_layout()
    out = outdir / "fidelity_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[Fidelity] Saved: {out}")
# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe-side fidelity evaluation v3 with signed directional deletion."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results",  required=True)
    parser.add_argument("--outdir",   default="results/fidelity_v3")
    parser.add_argument("--ada_ckpt", default="pretrained/adaface_ir50_ms1mv2.ckpt")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--seed",     default=42, type=int)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ada = get_ada(args.ada_ckpt, args.device)
    manifest = load_manifest(args.manifest)

    print(f"\n{'=' * 72}")
    print("  Fidelity evaluation v3")
    print("  Probe side only (B perturbed, A fixed)")
    print("  Primary  : absolute cross-perturbation deletion")
    print("  Secondary: signed directional deletion + support deletion")
    print("  Triangulation: GradCAM (non-blocking diagnostic)")
    print(f"  Device   : {args.device}")
    print(f"{'=' * 72}\n")

    all_results: List[Dict[str, Any]] = []

    for i, row in enumerate(manifest, 1):
        pair_id  = row["pair_id"]
        category = row.get("category", "")
        gt       = normalize_label(row.get("ground_truth", ""))
        pair_dir = Path(args.results) / pair_id

        print(f"[{i:03d}/{len(manifest)}] {pair_id}  ({category}, gt={gt})")

        try:
            result = load_result(pair_dir)
        except FileNotFoundError as e:
            print(f"       SKIP: {e}")
            all_results.append({
                "pair_id": pair_id, "category": category,
                "ground_truth": gt, "error": str(e),
            })
            continue

        r = evaluate_pair(pair_dir, result, ada, rng)
        r["pair_id"]      = pair_id
        r["category"]     = category
        r["ground_truth"] = gt
        all_results.append(r)

        if r.get("error"):
            print(f"       ERROR: {r['error']}")
            continue

        if r.get("s0_warning"):
            print(f"       WARN: {r['s0_warning']}")

        absA = r.get("abs_del_A") or {}
        dd   = r.get("dir_del")   or {}
        rho  = r.get("rho_gc_mean_gray")

        dir_str = (
            f"dir={dd['top_minus_bottom']:.4f}"
            if not dd.get("insufficient") and dd.get("top_minus_bottom") is not None
            else f"dir=INSUF({dd.get('flag','?')})"
        )
        absA_val = absA.get("top_minus_bottom")
        absA_str = f"{absA_val:.4f}" if absA_val is not None else "N/A"
        rho_str = f"{rho:+.3f}" if rho is not None else "N/A"

        print(
            f"       absA={absA_str}  "
            f"{dir_str}  "
            f"rho={rho_str}  "
            f"top_abs={r.get('top_abs_gray', '?')}"
        )

    save_csv(all_results, outdir)
    write_summary(all_results, outdir, rng)
    make_plots(all_results, outdir)
    print("\nDone.")
if __name__ == "__main__":
    main()