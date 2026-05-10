import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

from src.models.fr.adaface import AdaFaceEmbedder, AdaFaceConfig
@dataclass(frozen=True)
class ImgRec:
    path: Path
    name: str
def list_lfw_images(images_dir: Path) -> Tuple[List[ImgRec], Dict[str, List[Path]]]:
    recs: List[ImgRec] = []
    by_name: Dict[str, List[Path]] = {}
    for person_dir in images_dir.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        imgs = sorted(p for p in person_dir.glob("*.jpg") if p.is_file())
        if not imgs:
            continue
        by_name[name] = imgs
        for p in imgs:
            recs.append(ImgRec(path=p, name=name))
    return recs, by_name
def align_to_rgb_uint8(mtcnn: MTCNN, path: Path) -> Optional[np.ndarray]:
    """MTCNN-aligned 112x112 RGB uint8."""
    img = Image.open(path).convert("RGB")
    with torch.inference_mode():
        face = mtcnn(img)
    if face is None:
        return None
    x = face.detach().cpu().numpy()
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0
    if x.max() > 1.5:
        x = x / 255.0
    return (np.clip(x, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
def embed_images(
    ada: AdaFaceEmbedder,
    mtcnn: MTCNN,
    recs: List[ImgRec],
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str], List[Path]]:
    """Embed all images, returning (N,512) L2-normalised embeddings,
    identity names, and paths. Skips images where MTCNN finds no face."""
    embs: List[np.ndarray] = []
    names: List[str] = []
    paths: List[Path] = []
    skipped = 0

    for i in tqdm(range(0, len(recs), batch_size), desc="Embedding"):
        batch = recs[i : i + batch_size]
        tensors: List[torch.Tensor] = []
        batch_names: List[str] = []
        batch_paths: List[Path] = []

        for r in batch:
            rgb = align_to_rgb_uint8(mtcnn, r.path)
            if rgb is None:
                skipped += 1
                continue
            tensors.append(ada.preprocess_rgb_uint8(rgb))  # (1,3,112,112) on GPU
            batch_names.append(r.name)
            batch_paths.append(r.path)

        if not tensors:
            continue

        x = torch.cat(tensors, dim=0)  # (B,3,112,112)

        with torch.inference_mode():
            out = ada.model(x)

            # Extract features (handles AdaFace returning (feat, norm) tuple)
            if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
                feat_raw = out[0]
            else:
                while isinstance(out, (tuple, list)):
                    out = out[0]
                if not torch.is_tensor(out):
                    raise TypeError(f"Unexpected model output type after unwrapping: {type(out)}")
                feat_raw = out

            feat = torch.nn.functional.normalize(feat_raw, p=2, dim=1)

        embs.append(feat.detach().cpu().numpy().astype(np.float32))
        names.extend(batch_names)
        paths.extend(batch_paths)

    print(f"[Embed] Skipped {skipped} images (no face detected)")
    embs_np = np.vstack(embs) if embs else np.zeros((0, 512), dtype=np.float32)
    return embs_np, names, paths
def sample_genuine_pairs(
    by_name: Dict[str, List[Path]],
    embedded_paths: List[Path],
    num_pairs: int,
    rng: random.Random,
) -> List[Tuple[Path, Path]]:
    embedded_set = set(embedded_paths)
    candidates = {n: [p for p in imgs if p in embedded_set] for n, imgs in by_name.items()}
    candidates = {n: imgs for n, imgs in candidates.items() if len(imgs) >= 2}
    if not candidates:
        raise RuntimeError("No identities with >=2 embedded images.")
    names = list(candidates.keys())
    pairs = []
    for _ in range(num_pairs):
        name = rng.choice(names)
        a, b = rng.sample(candidates[name], 2)
        pairs.append((a, b))
    return pairs
def sample_impostor_indices(
    names: List[str],
    num_pairs: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Sample unique impostor (different-identity) index pairs."""
    N = len(names)
    pairs: List[Tuple[int, int]] = []
    seen = set()
    attempts = 0
    max_attempts = num_pairs * 10

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        i = rng.randrange(N)
        j = rng.randrange(N)
        if i == j or names[i] == names[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, j))

    if len(pairs) < num_pairs:
        print(f"[WARN] Only sampled {len(pairs)}/{num_pairs} unique impostor pairs.")
    return pairs
def conservative_threshold_at_far(
    imp_scores: np.ndarray,
    far_target: float,
) -> Tuple[float, float, int]:
    """
    Choose threshold so that FAR <= far_target, handling ties safely.
    Returns: (threshold, actual_far, accepted_impostor_count)
    """
    imp_sorted = np.sort(imp_scores)[::-1]  # descending
    N = len(imp_sorted)

    k = int(np.floor(far_target * N))
    k = max(1, min(k, N))  # clamp to [1, N]

    thr = float(imp_sorted[k - 1])

    accepted = int(np.sum(imp_scores >= thr))
    if accepted > k:
        # Nudge threshold upward by 1 ulp to break ties
        thr_nudged = float(np.nextafter(thr, np.inf))
        accepted_nudged = int(np.sum(imp_scores >= thr_nudged))
        if accepted_nudged > 0:
            thr = thr_nudged
            accepted = accepted_nudged

    actual_far = accepted / N
    return thr, actual_far, accepted
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ada_ckpt", default="pretrained/adaface_ir50_ms1mv2.ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_images", type=int, default=0, help="Cap on images to embed (0 = all)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_impostor_pairs", type=int, default=200_000)
    ap.add_argument("--num_genuine_pairs", type=int, default=20_000)
    ap.add_argument("--far_target", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outfile", default="pretrained/threshold_lfw.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    images_dir = Path(args.images_dir)

    mtcnn = MTCNN(image_size=112, margin=10, keep_all=False, post_process=False, device=device)
    ada = AdaFaceEmbedder(AdaFaceConfig(architecture="ir_50", ckpt_path=args.ada_ckpt, device=str(device)))

    all_recs, by_name = list_lfw_images(images_dir)
    print(f"Found {len(all_recs)} images across {len(by_name)} identities")

    if args.max_images and args.max_images < len(all_recs):
        rng.shuffle(all_recs)
        all_recs = all_recs[: args.max_images]

    embs, names, paths = embed_images(ada, mtcnn, all_recs, args.batch_size)
    print(f"Embedded {len(embs)} images successfully")

    if len(embs) < 2:
        raise RuntimeError("Too few embeddings produced. Try increasing max_images or check MTCNN detection.")

    path_to_idx = {p: i for i, p in enumerate(paths)}

    # Genuine scores
    genuine_pairs = sample_genuine_pairs(by_name, paths, args.num_genuine_pairs, rng)
    gen_scores = []
    for a, b in tqdm(genuine_pairs, desc="Genuine scores"):
        ia, ib = path_to_idx.get(a), path_to_idx.get(b)
        if ia is None or ib is None:
            continue
        gen_scores.append(float(np.dot(embs[ia], embs[ib])))
    gen_scores = np.array(gen_scores, dtype=np.float32)

    # Impostor scores
    imp_idx_pairs = sample_impostor_indices(names, args.num_impostor_pairs, rng)
    imp_scores = np.array(
        [float(np.dot(embs[i], embs[j])) for i, j in tqdm(imp_idx_pairs, desc="Impostor scores")],
        dtype=np.float32,
    )

    # Threshold
    thr, actual_far, accepted = conservative_threshold_at_far(imp_scores, args.far_target)
    tar = float(np.mean(gen_scores >= thr))
    frr = 1.0 - tar

    print("\n--- Calibration result ---")
    print(f"  Threshold : {thr:.6f}")
    print(f"  FAR       : {actual_far*100:.4f}%  (target: {args.far_target*100:.3f}%)")
    print(f"  TAR       : {tar*100:.2f}%")
    print(f"  FRR       : {frr*100:.2f}%")

    out = {
        "threshold": thr,
        "threshold_calibrated": True,
        "far_target": args.far_target,
        "calibration_split": {
            "dataset": "LFW unrestricted random pairs (MTCNN aligned)",
            "images_dir": str(images_dir.resolve()),
            "n_images_embedded": int(len(embs)),
            "n_genuine_pairs": int(len(gen_scores)),
            "n_impostor_pairs": int(len(imp_scores)),
            "accepted_impostors": int(accepted),
            "actual_far": float(actual_far),
            "tar_at_far": float(tar),
            "frr_at_far": float(frr),
            "seed": args.seed,
        },
        "model": "adaface_ir50_ms1mv2",
        "notes": "MTCNN-aligned 112×112 crops. Threshold selected conservatively with tie-handling.",
    }

    outpath = Path(args.outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {outpath.resolve()}")
if __name__ == "__main__":
    main()