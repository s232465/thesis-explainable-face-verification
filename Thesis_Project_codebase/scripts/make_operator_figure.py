"""
make_operator_figure.py

Generates the operator-facing explanation figure for a face pair result.

Layout: [Reference — clean] | [Probe — GradCAM + contours] | [Text panel]

The text panel has four blocks: decision summary, evidence summary,
reliability summary, and an operational note.

Usage:
    python scripts/make_operator_figure.py --pair_dir results/pair_001
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Constants

MARGIN_LOW  = 0.05
MARGIN_HIGH = 0.15

# AdaFace norm below this value flags reduced biometric sample quality
QUALITY_LOW = 15.0

# Human-readable part name mapping (SC37: "facial region")
PART_LABELS: Dict[str, str] = {
    "l_eye":  "left eye",
    "r_eye":  "right eye",
    "l_brow": "left eyebrow",
    "r_brow": "right eyebrow",
    "nose":   "nose",
    "mouth":  "mouth",
    "hair":   "hair",
    "skin":   "skin",
}
# I/O helpers
# Part-selection logic shared between text and outline rendering

DISPLAY_MIN_THRESHOLD = 0.002
DISPLAY_TOP_K = 3
DISPLAY_SKIP = {"skin"}
def _collect_display_parts(
    per_part: Dict[str, Any],
    decision: str,
    *,
    min_threshold: float = DISPLAY_MIN_THRESHOLD,
    top_k: int = DISPLAY_TOP_K,
    keep_both_groups: bool = True,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    """Select parts for the overlay and evidence text.

    Green (similar): deltaB < 0 — occluding reduces similarity.
    Red (different): deltaB > 0 — occluding increases similarity.

    Colour assignment is the same regardless of decision; the text blocks
    use decision-aware language to explain what each colour means.

    Excludes skin (too large for useful outlines), applies a minimum
    threshold, and keeps top-k per group. Falls back to the single
    strongest part if nothing passes the threshold.
    """
    similar_all:   Dict[str, float] = {}   # green: deltaB < 0
    different_all: Dict[str, float] = {}   # red:   deltaB > 0

    for part, data in per_part.items():
        if part in DISPLAY_SKIP:
            continue

        db = data.get("deltaB")
        if db is None:
            db = data.get("deltaMean")
        if db is None or float(db) == 0.0:
            continue

        db  = float(db)
        mag = abs(db)

        if db < 0:
            similar_all[part]   = mag   # green
        else:
            different_all[part] = mag   # red

    # Apply threshold
    similar   = {p: v for p, v in similar_all.items()   if v >= min_threshold}
    different = {p: v for p, v in different_all.items() if v >= min_threshold}

    # Top-k after thresholding
    top_similar   = sorted(similar.items(),   key=lambda x: x[1], reverse=True)[:top_k]
    top_different = sorted(different.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Fallback: if nothing survives threshold, keep strongest single part
    if allow_fallback and not top_similar and not top_different:
        candidates = (
            [("similar",   p, v) for p, v in similar_all.items()] +
            [("different", p, v) for p, v in different_all.items()]
        )
        candidates = [x for x in candidates if x[2] > 0]
        if candidates:
            kind, part, value = sorted(candidates, key=lambda x: x[2], reverse=True)[0]
            if kind == "similar":
                top_similar   = [(part, value)]
            else:
                top_different = [(part, value)]

    # Return under both naming conventions so call sites are readable
    return {
        "similar":        dict(top_similar),    # green
        "different":      dict(top_different),  # red
        "similar_list":   top_similar,
        "different_list": top_different,
        # legacy keys — kept so existing callers that use "supporting"/"conflicting"
        # continue to work; they now carry the absolute meaning (similar/different)
        "supporting":      dict(top_similar),
        "conflicting":     dict(top_different),
        "supporting_list": top_similar,
        "conflicting_list": top_different,
    }

def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def _load_npz_masks(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k].astype(np.uint8) for k in data.files}
def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))
def _load_cam(path: Path) -> np.ndarray:
    cam = np.load(path).astype(np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam
# Visual helpers

def _overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    cam_u8   = (cam * 255.0).clip(0, 255).astype(np.uint8)
    heat_rgb = np.stack([cam_u8, np.zeros_like(cam_u8), 255 - cam_u8], axis=-1)
    out      = img.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)

def _friendly(part: str) -> str:
    """Convert internal part key to human-readable label."""
    return PART_LABELS.get(part, part)
def _draw_part_outlines(
    ax,
    masks: Dict[str, np.ndarray],
    per_part: Dict[str, Any],
    decision: str,
    conf_label: str,
    min_threshold: float = DISPLAY_MIN_THRESHOLD,
    top_k: int = DISPLAY_TOP_K,
) -> None:
    """Draw contour outlines on the probe image, using the same part selection
    logic as the evidence text."""
    selected = _collect_display_parts(
        per_part,
        decision,
        min_threshold=min_threshold,
        top_k=top_k,
        keep_both_groups=True,
        allow_fallback=True,
    )

    top_supporting  = selected["similar"]    # green
    top_conflicting = selected["different"]  # red

    def draw_group(group: Dict[str, float], color: str, linestyle: str) -> None:
        if not group:
            return

        vals = list(group.values())
        vmax = max(vals) if vals else 1.0

        for part, value in group.items():
            mask = masks.get(part)
            if mask is None:
                continue

            m_float = mask.astype(np.float32)
            if m_float.max() <= 0:
                continue

            lw = 1.5 + 2.5 * (value / (vmax + 1e-8))
            ax.contour(
                m_float,
                levels=[0.5],
                colors=[color],
                linewidths=lw,
                linestyles=linestyle,
            )

    # Show BOTH groups always; this matches your interpretation framework better
    draw_group(top_supporting, color="green", linestyle="-")
    draw_group(top_conflicting, color="red", linestyle="--")

    from matplotlib.lines import Line2D
    legend_items = []
    if top_supporting:
        legend_items.append(
            Line2D([0], [0], color="green", lw=2, label="local similarity")
        )
    if top_conflicting:
        legend_items.append(
            Line2D([0], [0], color="red", lw=2, ls="--", label="local dissimilarity")
        )

    if legend_items:
        ax.legend(
            handles=legend_items,
            loc="lower center",
            fontsize=7,
            framealpha=0.8,
            frameon=True,
        )

# Confidence / quality helpers

def _confidence_label(margin: Optional[float]) -> str:
    if margin is None:
        return "N/A"
    am = abs(margin)
    if am < MARGIN_LOW:
        return "low"
    if am < MARGIN_HIGH:
        return "medium"
    return "high"
# SC37-aligned natural language explanation blocks

def _natural_join(parts: List[str]) -> str:
    """Join a list of strings naturally: 'a', 'a and b', 'a, b, and c'."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"
def _decision_block(decision: str, conf_label: str, margin: Optional[float]) -> str:
    """
    Block 1 — Decision summary.
    Uses SC37 terms: mated pair / non-mated pair / decision threshold.
    Adds a plain-English consequence sentence and a confidence-specific
    action prompt for the operator.
    """
    sc37_decision = "mated pair" if decision == "match" else "non-mated pair"
    plain_meaning = (
        "This means the system believes the reference and probe images "
        "show the same person."
        if decision == "match" else
        "This means the system believes the reference and probe images "
        "show two different people."
    )

    if conf_label == "low":
        margin_str = f"{abs(margin):.3f}" if margin is not None else "unknown"
        return (
            f"The biometric comparison score is very close to the decision "
            f"threshold — the score margin is only {margin_str}. "
            f"The system tentatively classifies this as a {sc37_decision}, "
            f"but the result is unreliable. {plain_meaning} "
            f"Given the borderline score, this decision should NOT be "
            f"accepted without manual review. Examine the highlighted "
            f"facial regions carefully before making a final determination."
        )
    if conf_label == "medium":
        margin_str = f"{abs(margin):.3f}" if margin is not None else "unknown"
        return (
            f"The biometric comparison indicates a {sc37_decision} "
            f"with moderate confidence (score margin: {margin_str}). "
            f"{plain_meaning} "
            f"The score provides reasonable but not definitive evidence. "
            f"Review the highlighted regions to confirm the decision."
        )
    margin_str = f"{abs(margin):.3f}" if margin is not None else "unknown"
    return (
        f"The biometric comparison indicates a {sc37_decision} "
        f"with high confidence (score margin: {margin_str}). "
        f"{plain_meaning} "
        f"The score is well clear of the decision threshold."
    )
def _skin_dominance_note(per_part: Dict[str, Any]) -> Optional[str]:
    """
    Returns a note when the skin region carries the strongest occlusion signal
    but is excluded from the overlay due to its large area.
    Simplified wording — tells the operator the figure is simplified.
    """
    skin_db = None
    skin_data = per_part.get("skin", {})
    db = skin_data.get("deltaB")
    if db is None:
        db = skin_data.get("deltaMean")
    if db is not None:
        skin_db = float(db)

    if skin_db is None:
        return None

    skin_mag = abs(skin_db)
    if skin_mag < DISPLAY_MIN_THRESHOLD:
        return None

    max_nonskin = 0.0
    for part, data in per_part.items():
        if part == "skin":
            continue
        v = data.get("deltaB")
        if v is None:
            v = data.get("deltaMean")
        if v is not None:
            max_nonskin = max(max_nonskin, abs(float(v)))

    if skin_mag <= max_nonskin:
        return None

    return (
        "A broad skin/facial-surface region showed the strongest local effect "
        "but is not outlined in the overlay to avoid visual clutter. "
        "The displayed outlines represent the strongest specific feature-level "
        "signals after excluding skin. The figure is therefore simplified — "
        "the skin signal is part of the story."
    )
def _global_evidence_note(
    per_part: Dict[str, Any],
    decision: str,
    similar_parts: List[str],
    different_parts: List[str],
) -> Optional[str]:
    """
    Returns a note when the local displayed regions do not obviously explain
    the final decision — the evidence is diffuse rather than concentrated.

    Non-match with no red outlines: all deltaB are negative, yet score is
    below threshold — dissimilarity is global, not localised.

    Match with no red outlines: all deltaB are negative, score above threshold
    — clean match with no conflicting regions.
    """
    if different_parts:
        return None  # red regions exist — no need for a global note

    # Check whether ALL non-skin deltaB values are negative
    all_negative = True
    any_valid    = False
    for part, data in per_part.items():
        if part == "skin":
            continue
        db = data.get("deltaB")
        if db is None:
            db = data.get("deltaMean")
        if db is not None:
            any_valid = True
            if float(db) >= 0:
                all_negative = False
                break

    if not any_valid:
        return None

    if decision == "match" and all_negative:
        return (
            "The visible local evidence is predominantly similarity-based. "
            "No above-threshold dissimilarity regions were identified on the "
            "displayed probe regions."
        )

    if decision != "match" and all_negative:
        return (
            "The model has decided this is a non-mated pair. However, no single "
            "facial region shows a clearly dominant dissimilarity signal. "
            "This suggests the non-match decision may depend on weakly distributed "
            "cues, broad excluded regions such as skin, or global score behaviour "
            "rather than one clearly isolated facial feature. Inspect the two "
            "images holistically rather than focusing on individual outlined regions."
        )

    return None
def _empty_group_note(
    similar_parts: List[str],
    different_parts: List[str],
    decision: str,
) -> Optional[str]:
    """
    Returns a note when one or both colour groups are empty after thresholding.
    Prevents the operator from expecting outlines that do not exist.
    """
    notes = []

    if not different_parts:
        if decision != "match":
            notes.append(
                "The system did not find any specific facial feature that looks "
                "clearly different. The non-match decision is based on the overall "
                "face rather than one identifiable part."
            )
        else:
            notes.append(
                "No facial region showed significant dissimilarity above the display "
                "threshold. No red outlines appear on the probe image."
            )

    if not similar_parts:
        notes.append(
            "The model found no significant local similarities to report. "
            "The faces look different across the board, so no green outlines appear."
        )

    return " ".join(notes) if notes else None
def _evidence_block(
    per_part: Dict[str, Any],
    decision: str,
    *,
    min_threshold: float = DISPLAY_MIN_THRESHOLD,
    top_k: int = DISPLAY_TOP_K,
) -> str:
    """Block 2 — Evidence summary. Green = similar regions, red = different regions.
    Appends notes for skin dominance, diffuse evidence, or empty groups."""
    selected = _collect_display_parts(
        per_part, decision,
        min_threshold=min_threshold, top_k=top_k,
        keep_both_groups=True, allow_fallback=True,
    )
    sim_parts  = [_friendly(p) for p, _ in selected["similar_list"]]
    diff_parts = [_friendly(p) for p, _ in selected["different_list"]]

    sim_txt  = _natural_join(sim_parts)  if sim_parts  else "none above display threshold"
    diff_txt = _natural_join(diff_parts) if diff_parts else "none above display threshold"

    if decision == "match":
        green_label = "Similar — supports match (green)"
        red_label   = "Dissimilar — conflicts with match (red)"
    else:
        green_label = "Similar — local similarity, did not overturn decision (green)"
        red_label   = "Dissimilar — supports non-match decision (red)"

    lines = (
        f"{green_label}: {sim_txt}\n"
        f"{red_label}:  {diff_txt}"
    )

    # Corner-case notes — appended in order of specificity
    skin_note   = _skin_dominance_note(per_part)
    global_note = _global_evidence_note(per_part, decision, sim_parts, diff_parts)
    empty_note  = _empty_group_note(sim_parts, diff_parts, decision)

    if skin_note:
        lines += f"\n{skin_note}"
    if global_note:
        lines += f"\n{global_note}"
    elif empty_note:
        # Only show empty_note if global_note did not already cover the emptiness
        lines += f"\n{empty_note}"

    return lines
def _reliability_block(
    rho: Optional[float],
    decision: str,
    similar_parts: List[str],
    different_parts: List[str],
) -> str:
    """Block 3 — Reliability summary. References specific regions and gives
    action-oriented guidance based on the gray/blur agreement rho."""
    sim_txt  = _natural_join(similar_parts)   if similar_parts   else "the visible green regions"
    diff_txt = _natural_join(different_parts)  if different_parts else "the visible red regions"

    if rho is None:
        return (
            "Explanation reliability could not be assessed — only one occlusion "
            "method produced valid results. Treat the highlighted regions as "
            "indicative only and rely primarily on your own visual comparison."
        )

    rho_str = f"{rho:.3f}"

    if rho < 0.5:
        if decision == "match":
            return (
                f"Explanation reliability is low (rho = {rho_str}). "
                f"The current decision may still be supportable, but manually "
                f"verify whether the green regions ({sim_txt}) genuinely look "
                f"similar, and check whether the red regions ({diff_txt}) reveal "
                f"meaningful differences that could weaken the mated-pair assessment."
            )
        return (
            f"Explanation reliability is low (rho = {rho_str}). "
            f"The current decision may still be supportable, but manually "
            f"verify whether the red regions ({diff_txt}) truly show convincing "
            f"differences, and whether the green regions ({sim_txt}) indicate "
            f"similarities strong enough to justify escalation."
        )
    if rho < 0.7:
        return (
            f"Explanation reliability is moderate (rho = {rho_str}). "
            f"Use the highlighted regions as a guide, but confirm that the "
            f"named regions are visually convincing before accepting the decision."
        )
    return (
        f"Explanation reliability is good (rho = {rho_str}). "
        f"The highlighted regions are a dependable guide for manual review."
    )
def _quality_block(
    qA: Optional[float],
    qB: Optional[float],
    decision: str,
    conf_lbl: str,
) -> str:
    """
    Quality note — neutral wording, connects quality to the decision.
    """
    qA_str = f"{qA:.1f}" if qA is not None else "N/A"
    qB_str = f"{qB:.1f}" if qB is not None else "N/A"
    both_good = (
        qA is not None and qA >= QUALITY_LOW and
        qB is not None and qB >= QUALITY_LOW
    )
    notes = []
    if both_good:
        if decision == "match":
            notes.append(
                f"Both biometric samples have adequate quality "
                f"(Reference: {qA_str}, Probe: {qB_str}). "
                f"Image quality does not provide an obvious quality-based "
                f"explanation for the current mated-pair result. "
                f"This is a high-quality comparison case."
            )
        else:
            notes.append(
                f"Both biometric samples have adequate quality "
                f"(Reference: {qA_str}, Probe: {qB_str}). "
                f"The non-mated decision is therefore unlikely to be explained "
                f"by blur, poor illumination, or pose. Assess whether the red "
                f"regions provide sufficient visible evidence to support the "
                f"current non-mated decision."
            )
    else:
        if qB is not None and qB < QUALITY_LOW:
            notes.append(
                f"The probe biometric sample has reduced quality (Score: {qB_str}). "
                f"This may have affected the comparison result. If possible, "
                f"acquire a new probe image under better conditions before "
                f"making a final determination."
            )
        if qA is not None and qA < QUALITY_LOW:
            notes.append(
                f"The reference biometric sample has reduced quality (Score: {qA_str}). "
                f"Consider whether the reference image on file is suitable "
                f"for a reliable comparison."
            )
    return " ".join(notes)
def _operational_block(
    decision: str,
    conf_lbl: str,
    similar_parts: List[str],
    different_parts: List[str],
    sim: Optional[float],
    thr: Optional[float],
    qA: Optional[float],
    qB: Optional[float],
) -> str:
    """Block 4 — Operational note. Decision-aware guidance with six branches
    (decision x confidence level)."""
    sim_str    = f"{sim:.3f}" if sim is not None else "N/A"
    thr_str    = f"{thr:.3f}" if thr is not None else "N/A"
    # Only generate colour references when the lists are actually populated
    sim_txt    = _natural_join(similar_parts)   if similar_parts   else None
    diff_txt   = _natural_join(different_parts) if different_parts else None
    quality_ok = (
        qA is not None and qA >= QUALITY_LOW and
        qB is not None and qB >= QUALITY_LOW
    )

    # High confidence match
    if decision == "match" and conf_lbl == "high":
        if sim_txt:
            base = (
                f"The system produced a high-confidence mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"Review the green solid outlines on {sim_txt} — these regions show "
                f"the strongest local similarity and support the match decision."
            )
        else:
            base = (
                f"The system produced a high-confidence mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"No specific non-skin region dominates the displayed signal. "
                f"See the evidence summary for possible skin-dominance or "
                f"diffuse-evidence notes."
            )
        if diff_txt:
            base += (
                f" The red dashed outlines on {diff_txt} indicate local "
                f"dissimilarity that conflicts with the match — assess whether "
                f"this represents a genuine structural difference."
            )
        if quality_ok:
            base += (
                " Both samples have adequate quality, so image degradation "
                "does not provide an obvious alternative explanation."
            )
        return base

    # Medium confidence match
    if decision == "match" and conf_lbl == "medium":
        if sim_txt:
            base = (
                f"The system produced a medium-confidence mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"Review the green solid outlines on {sim_txt} — these are the "
                f"strongest areas of local similarity supporting the match."
            )
        else:
            base = (
                f"The system produced a medium-confidence mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"No specific non-skin region dominates the displayed signal. "
                f"See the evidence summary for possible skin-dominance or "
                f"diffuse-evidence notes."
            )
        if diff_txt:
            base += (
                f" The red region ({diff_txt}) shows local dissimilarity that "
                f"conflicts with the match — assess whether this is a genuine "
                f"structural difference or an imaging artefact."
            )
        return base

    # Low confidence match
    if decision == "match" and conf_lbl == "low":
        green_clause = f"The green regions ({sim_txt})" if sim_txt else "No specific green region is above threshold"
        red_clause   = f"the red regions ({diff_txt})"  if diff_txt else "no specific red regions are above threshold"
        return (
            f"The system produced a low-confidence mated-pair decision "
            f"(Score: {sim_str}, Threshold: {thr_str}). "
            f"This is a borderline case. {green_clause} show local similarity "
            f"supporting the match, and {red_clause} show local dissimilarity "
            f"working against it. "
            f"If the similarities appear structurally convincing, the mated-pair "
            f"decision is more strongly supported. "
            f"If the differences appear structurally significant, consider "
            f"escalating this pair for further review."
        )

    # High confidence non-match
    if decision != "match" and conf_lbl == "high":
        if diff_txt:
            base = (
                f"The system produced a high-confidence non-mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"Review the red dashed outlines on {diff_txt} — these regions "
                f"show the strongest local dissimilarity and support the non-match decision."
            )
        else:
            base = (
                f"The system produced a high-confidence non-mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"No specific non-skin region dominates the displayed signal. "
                f"See the evidence summary for possible skin-dominance or "
                f"diffuse-evidence notes."
            )
        if sim_txt:
            base += (
                f" The green solid outlines on {sim_txt} show local similarity "
                f"that did not overturn the overall non-match decision."
            )
        if quality_ok:
            base += (
                " Both samples have adequate quality, so the decision is "
                "unlikely to be explained by image degradation alone."
            )
        return base

    # Medium confidence non-match
    if decision != "match" and conf_lbl == "medium":
        if diff_txt:
            base = (
                f"The system produced a medium-confidence non-mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"Review the red dashed outlines on {diff_txt} — these are the "
                f"strongest evidence of dissimilarity supporting the non-match. "
                f"Assess whether these features appear structurally different to you."
            )
        else:
            base = (
                f"The system produced a medium-confidence non-mated-pair decision "
                f"(Score: {sim_str}, Threshold: {thr_str}). "
                f"No specific non-skin region dominates the displayed signal. "
                f"See the evidence summary for possible skin-dominance or "
                f"diffuse-evidence notes."
            )
        if sim_txt:
            base += (
                f" The green region ({sim_txt}) shows local similarity that "
                f"did not overturn the decision — assess whether this similarity "
                f"is meaningful or coincidental."
            )
        return base

    # Low confidence non-match
    red_clause   = f"The red regions ({diff_txt})"  if diff_txt else "No specific red region is above threshold"
    green_clause = f"the green regions ({sim_txt})" if sim_txt  else "no specific green regions are above threshold"
    return (
        f"The system produced a low-confidence non-mated-pair decision "
        f"(Score: {sim_str}, Threshold: {thr_str}). "
        f"This is a borderline case. {red_clause} show local dissimilarity "
        f"supporting the non-match, and {green_clause} show local similarity "
        f"that did not overturn it. "
        f"If the red differences appear structurally convincing, the non-mated "
        f"decision is more strongly supported. "
        f"If the green similarities appear structurally significant, consider "
        f"escalating this pair for further review."
    )
def _one_liner(
    decision: str,
    conf_label: str,
    per_part: Dict[str, Any],
    rho: Optional[float],
    sim: Optional[float] = None,
    thr: Optional[float] = None,
) -> str:
    """Three-sentence operator instruction summary."""
    sc37_decision = "mated pair" if decision == "match" else "non-mated pair"
    selected = _collect_display_parts(
        per_part, decision,
        min_threshold=DISPLAY_MIN_THRESHOLD, top_k=2,
        keep_both_groups=True, allow_fallback=True,
    )
    sim_parts  = [_friendly(p) for p, _ in selected["similar_list"]]
    diff_parts = [_friendly(p) for p, _ in selected["different_list"]]
    sim_txt  = _natural_join(sim_parts[:2])  if sim_parts  else None
    diff_txt = _natural_join(diff_parts[:2]) if diff_parts else None

    # Sentence 1 — decision + confidence
    parts = [f"The system indicates a {sc37_decision} with {conf_label} confidence."]

    # Sentence 2 — what to inspect, decision-aware, only reference colours that exist
    if decision == "match":
        if sim_txt:
            s2 = f"Check whether the green regions ({sim_txt}) show convincing similarity supporting the match."
            if diff_txt:
                s2 += f" Also verify that the red regions ({diff_txt}) do not represent significant structural differences."
        else:
            s2 = (
                "No specific non-skin region dominates the displayed signal. "
                "See the evidence summary for possible skin-dominance or diffuse-evidence notes."
            )
    else:
        if diff_txt:
            s2 = f"Check whether the red regions ({diff_txt}) show convincing dissimilarity supporting the non-match."
            if sim_txt:
                s2 += f" Also review the green regions ({sim_txt}) — these show local similarity that did not overturn the decision."
        else:
            s2 = (
                "No specific non-skin region dominates the displayed signal. "
                "See the evidence summary for possible skin-dominance or diffuse-evidence notes."
            )
    parts.append(s2)

    # Sentence 3 — reliability
    if rho is not None:
        rho_str = f"{rho:.3f}"
        if rho < 0.5:
            parts.append(f"Explanation reliability is low (rho = {rho_str}) — manual verification is especially important.")
        elif rho < 0.7:
            parts.append(f"Explanation reliability is moderate (rho = {rho_str}).")
        else:
            parts.append(f"Explanation reliability is good (rho = {rho_str}).")

    return " ".join(parts)
# Plain-text summary file

def _write_summary_txt(
    outdir: Path,
    decision: str,
    sim: float,
    thresh: float,
    margin: float,
    conf_lbl: str,
    per_part: Dict[str, Any],
    rho: Optional[float],
    qA: Optional[float],
    qB: Optional[float],
    method: str,
) -> None:
    sc37_decision = "MATED PAIR" if decision == "match" else "NON-MATED PAIR"
    conf_color_tag = (
        "[BORDERLINE]" if conf_lbl == "low" else
        "[MODERATE]"   if conf_lbl == "medium" else
        ""
    )

    # Compute selected parts ONCE — shared across all blocks for consistency
    _selected = _collect_display_parts(
        per_part, decision,
        min_threshold=DISPLAY_MIN_THRESHOLD,
        top_k=DISPLAY_TOP_K,
        keep_both_groups=True, allow_fallback=True,
    )
    similar_parts   = [_friendly(p) for p, _ in _selected["similar_list"]]
    different_parts = [_friendly(p) for p, _ in _selected["different_list"]]

    lines = [
        "=" * 60,
        "BIOMETRIC COMPARISON — OPERATOR EXPLANATION",
        "=" * 60,
        "",
        "--- Comparison metrics ---",
        f"  Decision            : {sc37_decision}  {conf_color_tag}",
        f"  Comparison score    : {sim:.4f}",
        f"  Decision threshold  : {thresh:.4f}",
        f"  Score margin        : {margin:+.4f}",
        f"  Decision confidence : {conf_lbl.upper()}",
        f"  Sample quality A    : {f'{qA:.2f}' if qA is not None else 'N/A'}",
        f"  Sample quality B    : {f'{qB:.2f}' if qB is not None else 'N/A'}",
        f"  Explanation method  : occlusion ({method})",
        "",
    ]

    # Block 1
    lines += [
        "--- 1. Decision summary ---",
        _decision_block(decision, conf_lbl, margin),
        "",
    ]

    # Block 2
    lines += [
        "--- 2. Evidence summary ---",
        _evidence_block(per_part, decision),
        "",
    ]

    # Block 3
    lines += [
        "--- 3. Reliability summary ---",
        _reliability_block(rho, decision, similar_parts, different_parts),
        "",
    ]

    # Quality note — always shown when quality is relevant
    q_note = _quality_block(qA, qB, decision, conf_lbl)
    if q_note:
        lines += [
            "--- Biometric sample quality ---",
            q_note,
            "",
        ]

    # Block 4
    lines += [
        "--- 4. Operational note ---",
        _operational_block(
            decision, conf_lbl,
            similar_parts, different_parts,
            sim, thresh, qA, qB,
        ),
        "",
        "=" * 60,
        "",
        "One-line summary:",
        _one_liner(decision, conf_lbl, per_part, rho, sim, thresh),
        "",
    ]

    (outdir / "operator_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[Operator] Saved: {outdir / 'operator_summary.txt'}")
# Figure text panel

def _build_text_panel(
    ax,
    decision: str,
    sim: Optional[float],
    thr: Optional[float],
    margin: Optional[float],
    conf_lbl: str,
    per_part: Dict[str, Any],
    qA: Optional[float],
    qB: Optional[float],
    rho: Optional[float],
    method: str,
) -> None:
    ax.axis("off")

    sc37_decision = "Mated pair" if decision == "match" else "Non-mated pair"
    conf_color = (
        "red"        if conf_lbl == "low"    else
        "darkorange" if conf_lbl == "medium" else
        "darkgreen"
    )

    sim_str    = f"{sim:.4f}"    if sim    is not None else "N/A"
    thr_str    = f"{thr:.4f}"    if thr    is not None else "N/A"
    margin_str = f"{margin:+.4f}" if margin is not None else "N/A"
    qA_str     = f"{qA:.1f}"    if qA    is not None else "N/A"
    qB_str     = f"{qB:.1f}"    if qB    is not None else "N/A"
    rho_str    = f"{rho:.3f}"   if rho   is not None else "N/A"

    y = 1.0

    def txt(text, dy, fontsize=10, color="black", bold=False):
        nonlocal y
        y -= dy
        ax.text(0.0, y, text, va="top", fontsize=fontsize,
                color=color,
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes)

    # Decision
    txt(f"{sc37_decision}", 0.0, fontsize=12, color=conf_color, bold=True)
    txt(f"Confidence: {conf_lbl.upper()}", 0.07, fontsize=10,
        color=conf_color, bold=True)

    if conf_lbl == "low":
        txt("Operator verification required", 0.06, fontsize=9, color="red")
        y -= 0.01

    # Metrics
    txt(f"Score: {sim_str}   Threshold: {thr_str}", 0.08, fontsize=9)
    txt(f"Margin: {margin_str}", 0.05, fontsize=9)

    # Evidence — absolute colour semantics, decision-aware labels
    selected = _collect_display_parts(
        per_part, decision,
        min_threshold=DISPLAY_MIN_THRESHOLD,
        top_k=DISPLAY_TOP_K,
        keep_both_groups=True, allow_fallback=True,
    )
    sim_parts  = [_friendly(p) for p, _ in selected["similar_list"]]
    diff_parts = [_friendly(p) for p, _ in selected["different_list"]]
    sim_txt  = _natural_join(sim_parts)  if sim_parts  else "—"
    diff_txt = _natural_join(diff_parts) if diff_parts else "—"

    if decision == "match":
        green_label = "Similar — supports match (green)"
        red_label   = "Dissimilar — conflicts with match (red)"
    else:
        green_label = "Similar — did not overturn decision (green)"
        red_label   = "Dissimilar — supports non-match (red)"

    txt("Evidence summary", 0.08, fontsize=10, color="dimgray", bold=True)
    txt(f"{green_label}: {sim_txt}",  0.06, fontsize=9, color="#127A1A")
    txt(f"{red_label}:  {diff_txt}", 0.05, fontsize=9, color="#C9332A")

    # Corner-case notes on figure panel (compact single-line versions)
    skin_note   = _skin_dominance_note(per_part)
    global_note = _global_evidence_note(per_part, decision, sim_parts, diff_parts)
    empty_note  = _empty_group_note(sim_parts, diff_parts, decision)

    if skin_note:
        txt("⚠ Skin signal dominant — see summary", 0.05, fontsize=8, color="dimgray")
    if global_note:
        txt("⚠ Diffuse evidence — see summary", 0.05, fontsize=8, color="dimgray")
    elif empty_note and not global_note:
        txt("⚠ No outlines for one colour — see summary", 0.05, fontsize=8, color="dimgray")

    # Quality
    txt("Biometric sample quality", 0.09, fontsize=9, color="dimgray", bold=True)
    qA_color = "firebrick" if qA is not None and qA < QUALITY_LOW else "black"
    qB_color = "firebrick" if qB is not None and qB < QUALITY_LOW else "black"
    txt(f"Reference: {qA_str}", 0.05, fontsize=9, color=qA_color)
    txt(f"Probe:     {qB_str}", 0.05, fontsize=9, color=qB_color)

    # Reliability
    txt("Explanation reliability", 0.09, fontsize=9, color="dimgray", bold=True)
    rho_color = (
        "firebrick"  if rho is not None and rho < 0.5 else
        "darkorange" if rho is not None and rho < 0.7 else
        "black"
    )
    rho_verdict = (
        "Low — interpret with caution" if rho is not None and rho < 0.5 else
        "Moderate"                      if rho is not None and rho < 0.7 else
        "Good"
    )
    txt(f"Gray/blur rho: {rho_str}  ({rho_verdict})", 0.05,
        fontsize=9, color=rho_color)
# Core figure builder

def run_operator_figure(
    pair_dir: Path,
    method:   str  = "gray",
    out_path: Optional[Path] = None,
) -> None:
    """
    Build the operator figure from a completed result folder.

    Parameters
    ----------
    pair_dir  : folder containing result.json, aligned_*.png, parse_masks_B.npz
    method    : occlusion method for part signal colours ("gray" or "blur")
    out_path  : override output PNG path
    """
    pair_dir = Path(pair_dir)
    out_path = out_path or pair_dir / "operator_prototype_probe_only.png"

    # Load data
    result  = _load_json(pair_dir / "result.json")
    imgA    = _load_image(pair_dir / "aligned_A.png")
    imgB    = _load_image(pair_dir / "aligned_B.png")
    masksB  = _load_npz_masks(pair_dir / "parse_masks_B.npz")

    if (pair_dir / "gradcam_B.npy").exists():
        camB = _load_cam(pair_dir / "gradcam_B.npy")
        visB = _overlay_cam(imgB, camB, alpha=0.35)
    elif (pair_dir / "gradcam_overlay_B.png").exists():
        visB = _load_image(pair_dir / "gradcam_overlay_B.png")
    else:
        visB = imgB

    # Extract fields
    fv           = result.get("fv") or {}
    metrics      = result.get("metrics") or {}
    by_method    = metrics.get("by_method") or {}
    metric_block = by_method.get(method) or {}

    decision = fv.get("decision", "N/A")
    sim      = fv.get("similarity_cosine")
    thr      = fv.get("threshold")
    qA       = fv.get("qualityA")
    qB       = fv.get("qualityB")
    rho      = metrics.get("agreement_gray_vs_blur_spearman_rho")

    margin = fv.get("decision_margin")
    if margin is None and sim is not None and thr is not None:
        margin = float(sim) - float(thr)
    margin = float(margin) if margin is not None else None

    conf_lbl = fv.get("confidence_label") or _confidence_label(margin)

    # Per-part data — lives under part_signals.methods.<method>.per_part
    per_part = (
        (result.get("part_signals") or {})
        .get("methods", {})
        .get(method, {})
        .get("per_part", {})
    )
    # Build figure
    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[0, 2])

    # Reference A: plain, clean
    axA.imshow(imgA)
    axA.set_title("Reference biometric sample", fontsize=10, pad=4)
    axA.axis("off")

    # Probe B: GradCAM heatmap + part outlines
    axB.imshow(visB)
    axB.set_title("Probe biometric sample", fontsize=10, pad=4)
    axB.axis("off")
    _draw_part_outlines(axB, masksB, per_part, decision, conf_lbl)

    # Red border on probe if borderline
    if conf_lbl == "low":
        for spine in axB.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2.5)
            spine.set_visible(True)
        axB.text(0.5, 0.97, "BORDERLINE — MANUAL REVIEW",
                 transform=axB.transAxes, ha="center", va="top",
                 fontsize=7, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="red",
                           edgecolor="none", alpha=0.9))

    # Reliability badge
    if rho is not None:
        if rho >= 0.7:
            rho_label, rho_col = "GOOD STABILITY",     "#1D9E75"
        elif rho >= 0.5:
            rho_label, rho_col = "MODERATE STABILITY", "#BA7517"
        else:
            rho_label, rho_col = "LOW STABILITY",      "#A32D2D"
        axB.text(0.02, 0.02, rho_label,
                 transform=axB.transAxes, ha="left", va="bottom",
                 fontsize=6.5, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=rho_col,
                           edgecolor="none", alpha=0.9))

    # Quality badge on probe if reduced
    if qB is not None and qB < QUALITY_LOW:
        axB.text(0.98, 0.02, "LOW PROBE QUALITY",
                 transform=axB.transAxes, ha="right", va="bottom",
                 fontsize=6.5, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#A32D2D",
                           edgecolor="none", alpha=0.9))

    # Text panel
    _build_text_panel(
        ax=axT,
        decision=decision,
        sim=sim,
        thr=thr,
        margin=margin,
        conf_lbl=conf_lbl,
        per_part=per_part,
        qA=qA,
        qB=qB,
        rho=rho,
        method=method,
    )

    sc37_decision = "Mated Pair" if decision == "match" else "Non-Mated Pair"
    title_color   = "red" if conf_lbl == "low" else "black"
    fig.suptitle(
        f"Biometric Comparison — {sc37_decision}",
        fontsize=13, color=title_color, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Operator] Saved: {out_path}")

    # Plain-text summary
    _write_summary_txt(
        outdir=pair_dir,
        decision=decision,
        sim=float(sim) if sim is not None else 0.0,
        thresh=float(thr) if thr is not None else 0.315,
        margin=margin if margin is not None else 0.0,
        conf_lbl=conf_lbl,
        per_part=per_part,
        rho=rho,
        qA=qA,
        qB=qB,
        method=method,
    )
# CLI

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate operator figure from a completed pair result folder."
    )
    parser.add_argument("--pair_dir", required=True)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--method",   default="gray", choices=["gray", "blur"])
    args = parser.parse_args()

    run_operator_figure(
        pair_dir=Path(args.pair_dir),
        method=args.method,
        out_path=Path(args.out) if args.out else None,
    )
if __name__ == "__main__":
    main()