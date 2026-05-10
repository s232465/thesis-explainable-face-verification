"""
dataset_batch_runner.py

Batch runner for any manifest CSV. Runs the pipeline (run_pair.py) on every
pair, then produces a review table and accuracy report.

Usage:
    python scripts/dataset_batch_runner.py \
        --manifest  assets/gorilla/manifest.csv \
        --outdir    results/gorilla \
        --dataset   gorilla \
        --device    cuda

    python scripts/dataset_batch_runner.py \
        --manifest  assets/hda/manifest.csv \
        --outdir    results/hda \
        --dataset   hda \
        --device    cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(f)]


def _top_part(items: List[Any]) -> str:
    if not items:
        return "-"
    name, val = items[0]
    return str(name) if val and float(val) > 0 else "-"


def load_result(result_json: Path) -> Dict[str, Any]:
    with open(result_json, encoding="utf-8") as f:
        data = json.load(f)
    fv      = data.get("fv") or {}
    metrics = data.get("metrics") or {}
    by_m    = metrics.get("by_method") or {}
    gray    = by_m.get("gray") or {}

    return {
        "sim":               fv.get("similarity_cosine"),
        "threshold":         fv.get("threshold"),
        "decision":          fv.get("decision"),
        "decision_margin":   fv.get("decision_margin"),
        "confidence_label":  fv.get("confidence_label"),
        "qualityA":          fv.get("qualityA"),
        "qualityB":          fv.get("qualityB"),
        "top_support_gray":  _top_part(gray.get("top_3_support_parts") or []),
        "top_conflict_gray": _top_part(gray.get("top_3_conflict_parts") or []),
        "evidence_balance":  gray.get("evidence_balance"),
        "spearman_rho":      metrics.get("agreement_gray_vs_blur_spearman_rho"),
        "align_ok":          data.get("aligned") is not None,
        "parsing_ok":        data.get("parsing") is not None,
        "part_signals_ok":   data.get("part_signals") is not None,
        "error":             None,
    }


def _correct(gt: str, decision: Optional[str]) -> str:
    if decision is None:
        return "?"
    gt_n  = gt.strip().lower().replace("-", "")
    dec_n = decision.strip().lower().replace("-", "")
    return "Y" if gt_n == dec_n else "N"


def run_pair(
    pipeline_script: str,
    img_a: str,
    img_b: str,
    outdir: Path,
    device: str,
    extra_flags: List[str],
    timeout: int,
    verbose: bool,
) -> Tuple[bool, str]:
    cmd = [
        sys.executable, pipeline_script,
        "--imgA",   img_a,
        "--imgB",   img_b,
        "--outdir", str(outdir),
        "--device", device,
        "--do_parsing",
        "--do_part_signals",
    ] + extra_flags

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = getattr(result, "stderr", "") or ""
            return False, f"Exit {result.returncode}. {stderr[:400]}"
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as exc:
        return False, str(exc)


# Review table formatting

def _fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


_COLS = {
    "pair_id":    22, "category": 18, "gt": 9, "sim": 7,
    "decision":   11, "correct":  7,  "margin": 8,
    "confidence": 9,  "sup":      12, "con":    12,
    "rho":        6,
}
_SEP = "-" * (sum(_COLS.values()) + 2 * len(_COLS))
_HDR = ["pair_id","category","gt","sim","decision","ok","margin","conf","top_sup","top_con","rho"]


def _row_str(vals: List[str]) -> str:
    keys = list(_COLS.keys())
    return "  ".join(v[:_COLS[k]].ljust(_COLS[k]) for k, v in zip(keys, vals))


def print_table(rows: List[Dict]) -> None:
    print("\n" + _SEP)
    print(_row_str(_HDR))
    print(_SEP)
    for r in rows:
        print(_row_str([
            r["pair_id"],
            r["category"],
            r["gt"],
            _fmt(r.get("sim"), 4),
            r.get("decision") or "N/A",
            r.get("correct") or "?",
            _fmt(r.get("decision_margin"), 4),
            r.get("confidence_label") or "N/A",
            r.get("top_support_gray") or "-",
            r.get("top_conflict_gray") or "-",
            _fmt(r.get("spearman_rho"), 3),
        ]))
    print(_SEP + "\n")


def save_review_csv(rows: List[Dict], path: Path) -> None:
    fields = [
        "pair_id","category","ground_truth","sim","threshold","decision","correct",
        "decision_margin","confidence_label","qualityA","qualityB",
        "top_support_gray","top_conflict_gray","evidence_balance",
        "spearman_rho","align_ok","parsing_ok","part_signals_ok","error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_review_txt(rows: List[Dict], path: Path) -> None:
    lines = [_SEP, _row_str(_HDR), _SEP]
    for r in rows:
        lines.append(_row_str([
            r["pair_id"], r["category"], r["gt"],
            _fmt(r.get("sim"), 4),
            r.get("decision") or "N/A",
            r.get("correct") or "?",
            _fmt(r.get("decision_margin"), 4),
            r.get("confidence_label") or "N/A",
            r.get("top_support_gray") or "-",
            r.get("top_conflict_gray") or "-",
            _fmt(r.get("spearman_rho"), 3),
        ]))
    lines.append(_SEP)
    path.write_text("\n".join(lines), encoding="utf-8")


# Accuracy report

def build_accuracy_report(rows: List[Dict], dataset_name: str) -> str:
    total   = [r for r in rows if r.get("sim") is not None]
    correct = [r for r in total if r.get("correct") == "Y"]

    lines = [
        "=" * 56,
        f"ACCURACY REPORT — {dataset_name.upper()}",
        "=" * 56,
        "",
        f"Total pairs evaluated : {len(total)}",
        f"Correct decisions     : {len(correct)}",
        f"Overall accuracy      : {len(correct)/max(len(total),1)*100:.1f}%",
        "",
    ]

    # Per-category breakdown
    lines.append("Per-category breakdown")
    lines.append("-" * 40)
    cats: Dict[str, List] = defaultdict(list)
    for r in total:
        cats[r["category"]].append(r)

    for cat in sorted(cats.keys()):
        cat_rows = cats[cat]
        n_cor    = sum(1 for r in cat_rows if r.get("correct") == "Y")
        acc      = n_cor / len(cat_rows) * 100
        sims     = [r["sim"] for r in cat_rows if r.get("sim") is not None]
        mean_sim = sum(sims) / len(sims) if sims else 0.0
        lines.append(
            f"  {cat:<25} {n_cor:>2}/{len(cat_rows):<2}  "
            f"acc={acc:5.1f}%  mean_sim={mean_sim:.4f}"
        )

    # Match vs non-match split
    lines += ["", "Match vs non-match", "-" * 40]
    for gt_val in ["match", "nonmatch"]:
        subset = [r for r in total if r["gt"].lower().replace("-","") == gt_val]
        if not subset:
            continue
        n_cor   = sum(1 for r in subset if r.get("correct") == "Y")
        sims    = [r["sim"] for r in subset if r.get("sim") is not None]
        mean_s  = sum(sims) / len(sims) if sims else 0.0
        min_s   = min(sims) if sims else 0.0
        max_s   = max(sims) if sims else 0.0
        lines.append(
            f"  {gt_val:<10}  {n_cor}/{len(subset)}  acc={n_cor/len(subset)*100:.1f}%  "
            f"sim: mean={mean_s:.4f} min={min_s:.4f} max={max_s:.4f}"
        )

    # Threshold info
    thresholds = list({r["threshold"] for r in total if r.get("threshold") is not None})
    if thresholds:
        lines += ["", "Threshold", "-" * 40,
                  f"  {thresholds[0]:.6f}  (LFW calibrated, FMR=0.001)"]

    # Confidence distribution
    conf_counts: Dict[str, int] = defaultdict(int)
    for r in total:
        conf_counts[r.get("confidence_label") or "unknown"] += 1
    lines += ["", "Confidence label distribution", "-" * 40]
    for lbl in ["high", "medium", "low", "unknown"]:
        if conf_counts[lbl]:
            lines.append(f"  {lbl:<10} {conf_counts[lbl]}")

    # Failures
    failed = [r for r in rows if r.get("error")]
    if failed:
        lines += ["", f"Failed pairs ({len(failed)})", "-" * 40]
        for r in failed:
            lines.append(f"  {r['pair_id']}: {r['error'][:80]}")

    bad_align = [r for r in total if not r.get("align_ok")]
    if bad_align:
        lines += ["", "Alignment failures", "-" * 40]
        for r in bad_align:
            lines.append(f"  {r['pair_id']}")

    lines += ["", "=" * 56, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner for face verification pipeline."
    )
    parser.add_argument("--manifest",  required=True)
    parser.add_argument("--outdir",    required=True)
    parser.add_argument("--dataset",   default="dataset",
                        help="Name for report headers (e.g. gorilla, hda)")
    parser.add_argument("--pipeline_script",
                        default="scripts/run_pair.py")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--timeout",   type=int, default=600)
    parser.add_argument("--do_gradcam",           action="store_true")
    parser.add_argument("--do_operator_overlay",  action="store_true")
    parser.add_argument("--skip_existing",         action="store_true")
    parser.add_argument("--debug",                 action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")

    pipeline = Path(args.pipeline_script)
    if not pipeline.exists():
        sys.exit(f"Pipeline script not found: {pipeline}")

    pairs    = read_manifest(str(manifest_path))
    outdir_b = Path(args.outdir)
    outdir_b.mkdir(parents=True, exist_ok=True)

    extra: List[str] = []
    if args.do_gradcam:
        extra.append("--do_gradcam")
    if args.do_operator_overlay:
        extra.append("--do_operator_overlay")
    if args.debug:
        extra.append("--debug")

    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}  ({len(pairs)} pairs)")
    print(f"  Output  : {outdir_b.resolve()}")
    print(f"  Device  : {args.device}")
    print(f"{'='*60}\n")

    table_rows: List[Dict] = []
    t0_total = time.time()

    for i, row in enumerate(pairs, 1):
        pair_id  = row["pair_id"]
        category = row["category"]
        gt       = row["ground_truth"]
        img_a    = row["img_a"]
        img_b    = row["img_b"]

        pair_dir    = outdir_b / pair_id
        result_path = pair_dir / "result.json"

        print(f"[{i:03d}/{len(pairs)}] {pair_id}  ({category}, gt={gt})")

        if args.skip_existing and result_path.exists():
            print("       -> skipping (exists)")
            summary = load_result(result_path)
            summary.update({"pair_id": pair_id, "category": category,
                            "gt": gt, "ground_truth": gt})
            summary["correct"] = _correct(gt, summary.get("decision"))
            table_rows.append(summary)
            continue

        if row.get("notes") == "FILE_MISSING":
            print("       -> SKIPPED (file missing in manifest)")
            table_rows.append({
                "pair_id": pair_id, "category": category,
                "gt": gt, "ground_truth": gt,
                "correct": "?", "error": "FILE_MISSING",
            })
            continue

        pair_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        ok, msg = run_pair(
            pipeline_script=str(pipeline),
            img_a=img_a, img_b=img_b,
            outdir=pair_dir, device=args.device,
            extra_flags=extra, timeout=args.timeout,
            verbose=args.debug,
        )

        elapsed = time.time() - t0
        status  = "ok" if ok else "FAIL"
        print(f"       -> {status}  ({elapsed:.1f}s)  {'' if ok else msg[:100]}")

        summary: Dict[str, Any] = {
            "pair_id": pair_id, "category": category,
            "gt": gt, "ground_truth": gt,
            "align_ok": False, "parsing_ok": False,
            "part_signals_ok": False,
            "error": None if ok else msg,
        }

        if ok and result_path.exists():
            try:
                summary.update(load_result(result_path))
            except Exception as exc:
                summary["error"] = f"JSON error: {exc}"

        summary["correct"] = _correct(gt, summary.get("decision"))
        table_rows.append(summary)

    total_elapsed = time.time() - t0_total
    print(f"\nBatch complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    print_table(table_rows)

    report = build_accuracy_report(table_rows, args.dataset)
    print(report)

    csv_out    = outdir_b / "review_table.csv"
    txt_out    = outdir_b / "review_table.txt"
    report_out = outdir_b / "accuracy_report.txt"

    save_review_csv(table_rows, csv_out)
    save_review_txt(table_rows, txt_out)
    report_out.write_text(report, encoding="utf-8")

    print(f"Saved:")
    print(f"  {csv_out}")
    print(f"  {txt_out}")
    print(f"  {report_out}")


if __name__ == "__main__":
    main()
