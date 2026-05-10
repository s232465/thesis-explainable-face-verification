"""
run_pair.py - Main pipeline for a single face pair.

Steps:
  1. MTCNN alignment (112x112)
  2. AdaFace embedding + cosine comparison
  3. (optional) Grad-CAM heatmap on probe
  4. (optional) SegFormer face parsing
  5. (optional) Per-region occlusion scoring (gray + blur)
  6. (optional) Operator figure generation
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN

from src.contracts.io import save_json
from src.contracts.types import (
    AlignedFace,
    AlignedPair,
    FacePair,
    FVResult,
    GlobalMetrics,
    GlobalMetricsMethod,
    ParseResult,
    PartSignal,
    PartSignals,
    PartSignalsMethod,
    PipelineResult,
    XMapResult,
)
from src.models.parsing.hf_face_parsing import HFFaceParser
from src.models.fr.adaface import AdaFaceEmbedder, AdaFaceConfig
from src.models.explain.gradcam import pair_gradcam
# Image helpers

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
def _normalize_face_tensor(face: torch.Tensor) -> np.ndarray:
    """Bring MTCNN output tensor to [0,1] HWC float32."""
    x = face.detach().cpu().numpy()
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0).transpose(1, 2, 0).astype(np.float32)
def save_tensor_as_image(face: torch.Tensor, out_path: Path) -> None:
    x = (_normalize_face_tensor(face) * 255.0).astype(np.uint8)
    Image.fromarray(x, mode="RGB").save(out_path)
def face_tensor_to_rgb_uint8(face: torch.Tensor) -> np.ndarray:
    return (_normalize_face_tensor(face) * 255.0).astype(np.uint8)
def face_tensor_to_pil(face: torch.Tensor) -> Image.Image:
    x = (_normalize_face_tensor(face) * 255.0).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")
def overlay_label_map(img_rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Quick pseudo-colour overlay for debugging parse results."""
    H, W = label_map.shape
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    colors[..., 0] = (label_map * 37) % 255
    colors[..., 1] = (label_map * 91) % 255
    colors[..., 2] = (label_map * 151) % 255
    out = img_rgb.astype(np.float32) * (1 - alpha) + colors.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)
def save_cam_overlay(rgb_uint8: np.ndarray, cam01: np.ndarray, out_path: Path, alpha: float = 0.45) -> None:
    cam_u8 = (cam01 * 255.0).clip(0, 255).astype(np.uint8)
    heat_rgb = np.stack([cam_u8, np.zeros_like(cam_u8), 255 - cam_u8], axis=-1)
    out = rgb_uint8.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_path)
# Alignment

@torch.no_grad()
def align_face(mtcnn: MTCNN, img: Image.Image) -> torch.Tensor:
    face = mtcnn(img)
    if face is None:
        raise ValueError("No face detected.")
    return face
# Similarity

def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    e1 = e1.view(-1)
    e2 = e2.view(-1)
    return float(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0), dim=1)[0].item())
# Mask helpers

def merge_masks(masks_dict: Dict[str, np.ndarray], keys: List[str], out_key: str) -> None:
    """OR multiple binary masks into one composite."""
    present = [masks_dict[k] for k in keys if k in masks_dict]
    if not present:
        return
    merged = np.zeros_like(present[0], dtype=bool)
    for m in present:
        merged |= m.astype(bool)
    masks_dict[out_key] = merged.astype(np.uint8)
def prune_keys(masks_dict: Dict[str, np.ndarray], keys_to_remove: List[str]) -> None:
    for k in keys_to_remove:
        masks_dict.pop(k, None)
def mask_stats(masks_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {k: int(np.count_nonzero(v)) for k, v in masks_dict.items()}
def validate_and_resolve_parts(parts, name_to_id):
    """Check which requested parts exist in the parser output. When 'mouth' is
    requested, also pull in u_lip/l_lip for compositing later."""
    missing = [p for p in parts if p not in name_to_id]
    available = set(name_to_id.keys())
    suggestions = {}

    for p in missing:
        if p == "mouth":
            cands = [c for c in ["mouth", "u_lip", "l_lip", "lip"] if c in available]
        elif p == "hair":
            cands = [c for c in ["hair", "hat", "headwear"] if c in available]
        else:
            cands = [c for c in available if p in c or c in p][:5]
        if cands:
            suggestions[p] = cands

    resolved = [p for p in parts if p in name_to_id]

    if "mouth" in parts:
        for sub in ("u_lip", "l_lip"):
            if sub in available and sub not in resolved:
                resolved.append(sub)

    report = {
        "requested_parts": parts,
        "resolved_parts": resolved,
        "missing_parts": missing,
        "suggestions": suggestions,
        "available_label_count": len(available),
    }
    return resolved, report
# Occlusion

def occlude_rgb_gray(rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """Gray-fill occlusion. 128 maps to 0.0 in AdaFace normalised space."""
    out = rgb.copy()
    out[mask01.astype(bool)] = 128
    return out
def occlude_rgb_blur(rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """Gaussian blur occlusion (k=11)."""
    try:
        import cv2
        blurred = cv2.GaussianBlur(rgb, (11, 11), 0)
    except Exception:
        from PIL import ImageFilter
        blurred = np.array(
            Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=5)),
            dtype=np.uint8,
        )
    out = rgb.copy()
    out[mask01.astype(bool)] = blurred[mask01.astype(bool)]
    return out
# Per-part scoring

def support_conflict_from_delta(delta: float, decision: str) -> tuple:
    """Convert raw occlusion delta to (support, conflict) magnitudes."""
    if decision == "match":
        return max(0.0, -delta), max(0.0, delta)
    else:
        return max(0.0, delta), max(0.0, -delta)
# Decision confidence

MARGIN_LOW = 0.05
MARGIN_HIGH = 0.15
def compute_decision_margin(sim: float, threshold: float) -> float:
    return float(sim - threshold)
def margin_confidence_label(margin: float) -> str:
    am = abs(margin)
    if am < MARGIN_LOW:
        return "low"
    if am < MARGIN_HIGH:
        return "medium"
    return "high"
def spearman_rho(x: List[float], y: List[float]) -> Optional[float]:
    """Spearman rank correlation (no scipy dependency)."""
    if len(x) != len(y) or len(x) < 2:
        return None

    def rankdata(a):
        idx = sorted(range(len(a)), key=lambda i: a[i])
        ranks = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[idx[j + 1]] == a[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    den = (
        sum((rx[i] - mx) ** 2 for i in range(len(rx)))
        * sum((ry[i] - my) ** 2 for i in range(len(ry)))
    ) ** 0.5
    return float(num / den) if den != 0.0 else None
def compute_part_signals_for_method(
    *, method, rgbA, rgbB, masksA, masksB, parts, ada,
    embA0, embB0, s0, decision, min_pixels=30,
) -> Dict[str, Any]:
    """Compute per-part occlusion deltas for one method (gray or blur).

    For each part, occludes both the reference (A) and probe (B) side
    independently, measuring how much each region contributes to the
    overall similarity score.
    """
    def apply_occ(rgb, mask01):
        if method == "gray":
            return occlude_rgb_gray(rgb, mask01)
        elif method == "blur":
            return occlude_rgb_blur(rgb, mask01)
        raise ValueError(f"Unknown method: {method}")

    eA0 = embA0.view(-1)
    eB0 = embB0.view(-1)
    out: Dict[str, Any] = {}

    for p in parts:
        mA = masksA.get(p)
        mB = masksB.get(p)

        if mA is None or mB is None:
            out[p] = dict(
                validA=False, validB=False,
                deltaA=None, deltaB=None, deltaMean=None,
                supportA=None, conflictA=None,
                supportB=None, conflictB=None,
                support=None, conflict=None,
                areaA_pixels=0, areaB_pixels=0,
                areaA_frac=0.0, areaB_frac=0.0,
                flags={"missing_mask": True},
            )
            continue

        mA01 = (mA > 0).astype(np.uint8)
        mB01 = (mB > 0).astype(np.uint8)
        areaA = int(mA01.sum())
        areaB = int(mB01.sum())
        total_px = mA01.shape[0] * mA01.shape[1]

        validA = areaA >= min_pixels
        validB = areaB >= min_pixels

        deltaA = None
        deltaB = None

        with torch.no_grad():
            if validA:
                rgbA_occ = apply_occ(rgbA, mA01)
                embA_occ = ada.embed(ada.preprocess_rgb_uint8(rgbA_occ)).view(-1)
                deltaA = float(torch.dot(embA_occ, eB0).item()) - s0

            if validB:
                rgbB_occ = apply_occ(rgbB, mB01)
                embB_occ = ada.embed(ada.preprocess_rgb_uint8(rgbB_occ)).view(-1)
                deltaB = float(torch.dot(eA0, embB_occ).item()) - s0

        # Side-specific support/conflict
        supportA, conflictA = (
            support_conflict_from_delta(deltaA, decision) if deltaA is not None else (None, None)
        )
        supportB, conflictB = (
            support_conflict_from_delta(deltaB, decision) if deltaB is not None else (None, None)
        )

        # deltaMean for cross-method Spearman rho
        if deltaA is not None and deltaB is not None:
            deltaMean = (deltaA + deltaB) / 2.0
        elif deltaA is not None:
            deltaMean = deltaA
        elif deltaB is not None:
            deltaMean = deltaB
        else:
            deltaMean = None

        # Aggregated support/conflict (mean of valid sides)
        valid_sup = [v for v in [supportA, supportB] if v is not None]
        valid_con = [v for v in [conflictA, conflictB] if v is not None]
        support = float(sum(valid_sup) / len(valid_sup)) if valid_sup else None
        conflict = float(sum(valid_con) / len(valid_con)) if valid_con else None

        out[p] = dict(
            validA=bool(validA), validB=bool(validB),
            deltaA=deltaA, deltaB=deltaB,
            deltaMean=float(deltaMean) if deltaMean is not None else None,
            supportA=float(supportA) if supportA is not None else None,
            conflictA=float(conflictA) if conflictA is not None else None,
            supportB=float(supportB) if supportB is not None else None,
            conflictB=float(conflictB) if conflictB is not None else None,
            support=float(support) if support is not None else None,
            conflict=float(conflict) if conflict is not None else None,
            areaA_pixels=areaA, areaB_pixels=areaB,
            areaA_frac=areaA / total_px, areaB_frac=areaB / total_px,
            flags={},
        )

    return out
def compute_global_metrics_for_method(*, s0, decision, part_block, topk=3):
    """Aggregate per-part signals into global metrics for one occlusion method."""
    valid = [(p, d) for p, d in part_block.items()
             if d.get("support") is not None or d.get("conflict") is not None]

    support_items = [(p, float(d["support"])) for p, d in valid if d.get("support") is not None]
    conflict_items = [(p, float(d["conflict"])) for p, d in valid if d.get("conflict") is not None]

    support_total = sum(v for _, v in support_items)
    conflict_total = sum(v for _, v in conflict_items)

    top_support = sorted(support_items, key=lambda x: x[1], reverse=True)[:topk]
    top_conflict = sorted(conflict_items, key=lambda x: x[1], reverse=True)[:topk]

    n_valid = sum(1 for _, d in part_block.items()
                  if d.get("deltaMean") is not None
                  or d.get("deltaA") is not None
                  or d.get("deltaB") is not None)

    return dict(
        sim0=float(s0), decision=decision,
        support_total=float(support_total),
        conflict_total=float(conflict_total),
        evidence_balance=float(support_total - conflict_total),
        top_3_support_parts=top_support,
        top_3_conflict_parts=top_conflict,
        n_parts_used=int(n_valid),
    )
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgA", type=str, required=True)
    parser.add_argument("--imgB", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results/adaface_run")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine threshold (overrides calibration file)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--do_parsing", action="store_true")
    parser.add_argument("--do_gradcam", action="store_true")
    parser.add_argument("--do_part_signals", action="store_true",
                        help="Per-region occlusion scoring (needs --do_parsing)")
    parser.add_argument("--parsing_model", type=str, default="jonathandinu/face-parsing")
    parser.add_argument("--ada_ckpt", type=str, default="pretrained/adaface_ir50_ms1mv2.ckpt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--do_operator_overlay", action="store_true")
    parser.add_argument("--overlay_method", type=str, default="gray", choices=["gray", "blur"])
    args = parser.parse_args()

    if args.do_part_signals and not args.do_parsing:
        raise ValueError("--do_part_signals requires --do_parsing")

    # Load calibrated threshold if available
    cal_path = Path("pretrained/threshold_lfw.json")
    if args.threshold is None and cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        effective_threshold = float(cal["threshold"])
        threshold_calibrated = bool(cal.get("threshold_calibrated", True))
        threshold_far_target = float(cal["far_target"]) if cal.get("far_target") is not None else None
        cal_split = cal.get("calibration_split") or {}
        threshold_dataset = cal_split.get("dataset", "LFW")
        threshold_source = str(cal_path.resolve())
        threshold_notes = None
        tar_at_far = float(cal_split["tar_at_far"]) if cal_split.get("tar_at_far") is not None else None
        actual_far = float(cal_split["actual_far"]) if cal_split.get("actual_far") is not None else None
    else:
        effective_threshold = float(args.threshold) if args.threshold is not None else 0.6
        threshold_calibrated = False
        threshold_far_target = None
        threshold_dataset = None
        threshold_source = None
        threshold_notes = "Placeholder threshold; not calibrated."
        tar_at_far = None
        actual_far = None

    if args.debug:
        cal_str = "calibrated" if threshold_calibrated else "UNCALIBRATED"
        print(f"[Threshold] {cal_str} threshold={effective_threshold:.6f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    imgA_path = Path(args.imgA).expanduser().resolve()
    imgB_path = Path(args.imgB).expanduser().resolve()
    if not imgA_path.exists():
        raise FileNotFoundError(f"imgA not found: {imgA_path}")
    if not imgB_path.exists():
        raise FileNotFoundError(f"imgB not found: {imgB_path}")

    device = torch.device(args.device)
    mtcnn = MTCNN(image_size=112, margin=10, keep_all=False, post_process=False, device=device)
    ada = AdaFaceEmbedder(AdaFaceConfig(
        architecture="ir_50", ckpt_path=args.ada_ckpt, device=str(device),
    ))

    imgA = load_image(str(imgA_path))
    imgB = load_image(str(imgB_path))

    faceA = align_face(mtcnn, imgA)
    faceB = align_face(mtcnn, imgB)

    alignedA_path = outdir / "aligned_A.png"
    alignedB_path = outdir / "aligned_B.png"
    save_tensor_as_image(faceA, alignedA_path)
    save_tensor_as_image(faceB, alignedB_path)

    rgbA = face_tensor_to_rgb_uint8(faceA)
    rgbB = face_tensor_to_rgb_uint8(faceB)

    pair = FacePair(imgA=str(imgA_path), imgB=str(imgB_path), pair_id="", label=None)
    pipeline_result = PipelineResult(pair=pair)

    pipeline_result.aligned = AlignedPair(
        A=AlignedFace(path=str(alignedA_path.resolve()), size=[112, 112]),
        B=AlignedFace(path=str(alignedB_path.resolve()), size=[112, 112]),
    )

    inpA = ada.preprocess_rgb_uint8(rgbA)
    inpB = ada.preprocess_rgb_uint8(rgbB)

    # Grad-CAM (before main embeddings so hooks don't interfere)
    def embed_for_gradcam(model, x):
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
            feat_raw = out[0]
        else:
            while isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                raise TypeError(f"Unexpected model output: {type(out)}")
            feat_raw = out
        return F.normalize(feat_raw, p=2, dim=1)

    gradcam_sim0 = None

    if args.do_gradcam:
        camB, gradcam_sim0 = pair_gradcam(
            ada.model, inpA, inpB, debug=args.debug, embed_fn=embed_for_gradcam,
        )
        heatB_path = outdir / "gradcam_B.npy"
        np.save(heatB_path, camB.cam)
        save_cam_overlay(rgbB, camB.cam, outdir / "gradcam_overlay_B.png")
        pipeline_result.xmap = XMapResult(
            method="gradcam",
            heatmapB_path=str(heatB_path.resolve()),
            signed=False, normalize="minmax", layer=camB.layer_name,
            notes="Probe-side (B) map only. Reference (A) shown clean.",
        )
        if args.debug:
            print(f"[GradCAM] layer: {camB.layer_name}")

    # Embeddings + similarity
    embA = ada.embed(inpA)
    qA = ada.last_quality_score
    embB = ada.embed(inpB)
    qB = ada.last_quality_score
    sim = cosine_similarity(embA, embB)

    if args.debug:
        print(f"[Embed] qA={qA:.4f}  qB={qB:.4f}  sim={sim:.6f}")

    if args.do_gradcam and gradcam_sim0 is not None:
        diff = abs(sim - gradcam_sim0)
        if args.debug:
            print(f"[Check] pipeline sim={sim:.6f}  gradcam sim0={gradcam_sim0:.6f}  |diff|={diff:.2e}")
        if diff >= 1e-6:
            print(f"[WARN] GradCAM/pipeline similarity mismatch: diff={diff:.2e}")

    decision = "match" if sim >= effective_threshold else "non-match"
    margin = compute_decision_margin(sim, effective_threshold)
    conf_label = margin_confidence_label(margin)

    embA_path = outdir / "embedding_A.npy"
    embB_path = outdir / "embedding_B.npy"
    np.save(embA_path, embA.numpy())
    np.save(embB_path, embB.numpy())

    pipeline_result.fv = FVResult(
        model="AdaFace ir_50 (ms1mv2)",
        embeddingA_path=str(embA_path.resolve()),
        embeddingB_path=str(embB_path.resolve()),
        similarity_cosine=sim,
        threshold=effective_threshold,
        decision=decision,
        device=str(device),
        backend="pytorch",
        qualityA=qA, qualityB=qB,
        threshold_calibrated=threshold_calibrated,
        threshold_far_target=threshold_far_target,
        threshold_dataset=threshold_dataset,
        threshold_source=threshold_source,
        threshold_notes=threshold_notes,
        tar_at_far=tar_at_far,
        actual_far=actual_far,
        decision_margin=margin,
        confidence_label=conf_label,
    )

    # Face parsing
    if args.do_parsing:
        face_parser = HFFaceParser(model_name=args.parsing_model, device=str(device))

        pilA = face_tensor_to_pil(faceA)
        pilB = face_tensor_to_pil(faceB)

        outA = face_parser.parse_pil(pilA)
        outB = face_parser.parse_pil(pilB)

        labelA_path = outdir / "parse_label_A.npy"
        labelB_path = outdir / "parse_label_B.npy"
        np.save(labelA_path, outA.label_map)
        np.save(labelB_path, outB.label_map)

        Image.fromarray(overlay_label_map(np.array(pilA, np.uint8), outA.label_map)).save(outdir / "parse_overlay_A.png")
        Image.fromarray(overlay_label_map(np.array(pilB, np.uint8), outB.label_map)).save(outdir / "parse_overlay_B.png")

        if outA.id2label and outB.id2label and outA.id2label != outB.id2label:
            print("[Parsing] WARNING: id2label differs between A and B")

        name_to_id = {v: k for k, v in outA.id2label.items()} if outA.id2label else {}
        parts = ["skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "hair"]
        resolved_parts, parts_report = validate_and_resolve_parts(parts, name_to_id)

        if parts_report["missing_parts"]:
            print("[Parsing] WARNING: missing parts:", parts_report["missing_parts"])

        masksA = face_parser.masks_from_label_map(outA.label_map, resolved_parts, name_to_id)
        masksB = face_parser.masks_from_label_map(outB.label_map, resolved_parts, name_to_id)

        # Merge lip sub-parts into composite mouth mask
        merge_masks(masksA, ["mouth", "u_lip", "l_lip"], "mouth")
        merge_masks(masksB, ["mouth", "u_lip", "l_lip"], "mouth")
        prune_keys(masksA, ["u_lip", "l_lip"])
        prune_keys(masksB, ["u_lip", "l_lip"])

        if args.debug:
            print("[Parsing] Mask pixels A:", mask_stats(masksA))
            print("[Parsing] Mask pixels B:", mask_stats(masksB))

        for side, stats in [("A", mask_stats(masksA)), ("B", mask_stats(masksB))]:
            empties = [k for k, n in stats.items() if n < 10]
            if empties:
                print(f"[Parsing] WARNING: nearly-empty masks on {side}: {empties}")

        masksA_path = outdir / "parse_masks_A.npz"
        masksB_path = outdir / "parse_masks_B.npz"
        np.savez_compressed(masksA_path, **masksA)
        np.savez_compressed(masksB_path, **masksB)

        pipeline_result.parsing = {
            "A": ParseResult(
                label_map_path=str(labelA_path.resolve()), parts=parts,
                masks_path=str(masksA_path.resolve()), parsing_model=args.parsing_model,
            ),
            "B": ParseResult(
                label_map_path=str(labelB_path.resolve()), parts=parts,
                masks_path=str(masksB_path.resolve()), parsing_model=args.parsing_model,
            ),
        }

        # Per-region occlusion scoring
        if args.do_part_signals:
            if args.debug:
                print("[PartSignals] Computing gray + blur...")

            raw_gray = compute_part_signals_for_method(
                method="gray", rgbA=rgbA, rgbB=rgbB,
                masksA=masksA, masksB=masksB, parts=parts, ada=ada,
                embA0=embA, embB0=embB, s0=sim, decision=decision,
            )
            raw_blur = compute_part_signals_for_method(
                method="blur", rgbA=rgbA, rgbB=rgbB,
                masksA=masksA, masksB=masksB, parts=parts, ada=ada,
                embA0=embA, embB0=embB, s0=sim, decision=decision,
            )

            pipeline_result.part_signals = PartSignals(methods={
                "gray": PartSignalsMethod(method="gray", parts=parts,
                    per_part={p: PartSignal(**raw_gray[p]) for p in parts}),
                "blur": PartSignalsMethod(method="blur", parts=parts,
                    per_part={p: PartSignal(**raw_blur[p]) for p in parts}),
            })

            metrics_gray = compute_global_metrics_for_method(s0=sim, decision=decision, part_block=raw_gray)
            metrics_blur = compute_global_metrics_for_method(s0=sim, decision=decision, part_block=raw_blur)

            # Cross-method agreement
            xs, ys = [], []
            for p in parts:
                dx = raw_gray[p].get("deltaMean")
                dy = raw_blur[p].get("deltaMean")
                if dx is not None and dy is not None:
                    xs.append(float(dx))
                    ys.append(float(dy))

            rho = spearman_rho(xs, ys)
            pipeline_result.metrics = GlobalMetrics(
                by_method={
                    "gray": GlobalMetricsMethod(**metrics_gray),
                    "blur": GlobalMetricsMethod(**metrics_blur),
                },
                agreement_gray_vs_blur_spearman_rho=rho,
                agreement_n_parts=len(xs),
            )

            if args.debug:
                print(f"[Metrics] gray  sup={metrics_gray['support_total']:.4f}  con={metrics_gray['conflict_total']:.4f}")
                print(f"[Metrics] blur  sup={metrics_blur['support_total']:.4f}  con={metrics_blur['conflict_total']:.4f}")
                print(f"[Metrics] rho(gray, blur) = {rho}")

    save_json(pipeline_result.to_dict(), str(outdir / "result.json"))

    if args.debug:
        print(json.dumps(pipeline_result.to_dict(), indent=2))
    else:
        print(f"Saved result.json -> {outdir / 'result.json'}")

    if args.do_operator_overlay:
        try:
            from make_operator_figure import run_operator_figure
            run_operator_figure(pair_dir=outdir, method=args.overlay_method)
        except Exception as exc:
            print(f"[Overlay] WARNING: operator figure failed: {exc}")
if __name__ == "__main__":
    main()
