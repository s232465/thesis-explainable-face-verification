import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

from src.contracts.io import save_json
from src.contracts.types import FacePair, FVResult, PipelineResult, ParseResult
from src.models.parsing.hf_face_parsing import HFFaceParser
from src.models.fr.adaface import AdaFaceEmbedder, AdaFaceConfig


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.inference_mode()
def align_face(mtcnn: MTCNN, img: Image.Image) -> torch.Tensor:
    """
    Returns a single aligned face tensor (3, 112, 112).
    Raises ValueError if no face is detected.
    """
    face = mtcnn(img)
    if face is None:
        raise ValueError("No face detected.")
    return face


def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    # embeddings are already normalized
    return float(torch.dot(e1, e2).item())


def _normalize_face_tensor_to_01(face: torch.Tensor) -> np.ndarray:
    """
    Convert face tensor to float HWC RGB in [0,1], robust to:
      - [-1,1] normalization
      - [0,1]
      - [0,255]
    Input face: (3,H,W) torch tensor
    Output: (H,W,3) float32 in [0,1]
    """
    x = face.detach().cpu().numpy()  # (3,H,W)

    # If looks like [-1,1], map to [0,1]
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0

    # If looks like [0,255], map to [0,1]
    if x.max() > 1.5:
        x = x / 255.0

    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (1, 2, 0))  # HWC RGB
    return x.astype(np.float32)


def save_tensor_as_image(face: torch.Tensor, out_path: Path) -> None:
    """
    Save aligned face tensor as PNG (visual/debug).
    """
    x01 = _normalize_face_tensor_to_01(face)  # HWC float [0,1]
    x = (x01 * 255.0).astype(np.uint8)
    Image.fromarray(x, mode="RGB").save(out_path)


def face_tensor_to_rgb_uint8(face: torch.Tensor) -> np.ndarray:
    """
    Convert aligned face tensor to RGB uint8 (112x112x3) for AdaFace preprocessing.
    """
    x01 = _normalize_face_tensor_to_01(face)
    return (x01 * 255.0).astype(np.uint8)


def face_tensor_to_pil(face: torch.Tensor) -> Image.Image:
    """
    Convert aligned face tensor to a PIL image for parsing, independent of what was saved to disk.
    This prevents 'white aligned images' from breaking parsing.
    """
    x01 = _normalize_face_tensor_to_01(face)
    x = (x01 * 255.0).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")


def overlay_label_map(img_rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Simple overlay for sanity-checking parsing.
    img_rgb: HxWx3 uint8
    label_map: HxW int
    Returns HxWx3 uint8.
    """
    H, W = label_map.shape
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    colors[..., 0] = (label_map * 37) % 255
    colors[..., 1] = (label_map * 91) % 255
    colors[..., 2] = (label_map * 151) % 255
    out = (img_rgb.astype(np.float32) * (1 - alpha) + colors.astype(np.float32) * alpha)
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgA", type=str, required=True, help="Path to probe image")
    parser.add_argument("--imgB", type=str, required=True, help="Path to reference image")
    parser.add_argument("--outdir", type=str, default="results/adaface_run", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine threshold for match decision (placeholder)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--do_parsing",
        action="store_true",
        help="If set, run face parsing on aligned crops and store masks in PipelineResult.parsing",
    )
    parser.add_argument(
        "--parsing_model",
        type=str,
        default="jonathandinu/face-parsing",
        help="HuggingFace face parsing model name",
    )
    parser.add_argument(
        "--ada_ckpt",
        type=str,
        default="pretrained/adaface_ir50_ms1mv2.ckpt",
        help="Path (relative to repo root) to AdaFace checkpoint",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve paths early (prevents relative path confusion)
    imgA_path = Path(args.imgA).expanduser().resolve()
    imgB_path = Path(args.imgB).expanduser().resolve()
    if not imgA_path.exists():
        raise FileNotFoundError(f"imgA not found: {imgA_path}")
    if not imgB_path.exists():
        raise FileNotFoundError(f"imgB not found: {imgB_path}")

    device = torch.device(args.device)

    # MTCNN does detection + alignment; we disable post_process so the returned crop isn't FaceNet-normalized
    mtcnn = MTCNN(image_size=112, margin=10, keep_all=False, post_process=False, device=device)

    # AdaFace embedder
    ada = AdaFaceEmbedder(
        AdaFaceConfig(
            architecture="ir_50",
            ckpt_path=args.ada_ckpt,
            device=str(device),
        )
    )

    # Load images
    imgA = load_image(str(imgA_path))
    imgB = load_image(str(imgB_path))

    # Align faces
    faceA = align_face(mtcnn, imgA)
    faceB = align_face(mtcnn, imgB)

    # Save aligned crops (debug artifact)
    alignedA_path = outdir / "aligned_A.png"
    alignedB_path = outdir / "aligned_B.png"
    save_tensor_as_image(faceA, alignedA_path)
    save_tensor_as_image(faceB, alignedB_path)

    # Embeddings + similarity (AdaFace)
    rgbA = face_tensor_to_rgb_uint8(faceA)
    rgbB = face_tensor_to_rgb_uint8(faceB)

    inpA = ada.preprocess_rgb_uint8(rgbA)
    inpB = ada.preprocess_rgb_uint8(rgbB)

    embA = ada.embed(inpA)
    embB = ada.embed(inpB)

    sim = cosine_similarity(embA, embB)
    decision = "match" if sim >= args.threshold else "non-match"

    # Save embeddings
    embA_path = outdir / "embedding_A.npy"
    embB_path = outdir / "embedding_B.npy"
    np.save(embA_path, embA.numpy())
    np.save(embB_path, embB.numpy())

    # Build contract objects
    pair = FacePair(
        imgA=str(imgA_path),
        imgB=str(imgB_path),
        pair_id="",
        label=None,
    )

    fv = FVResult(
        model="AdaFace ir_50 (ms1mv2)",
        embeddingA_path=str(embA_path.resolve()),
        embeddingB_path=str(embB_path.resolve()),
        similarity_cosine=sim,
        threshold=args.threshold,
        decision=decision,
        device=str(device),
        backend="pytorch",
    )

    pipeline_result = PipelineResult(pair=pair, fv=fv)

    # -------------------------
    # Optional: face parsing (parse from in-memory aligned tensors, NOT from saved PNGs)
    # -------------------------
    if args.do_parsing:
        face_parser = HFFaceParser(model_name=args.parsing_model, device=str(device))

        pilA = face_tensor_to_pil(faceA)
        pilB = face_tensor_to_pil(faceB)

        outA = face_parser.parse_pil(pilA)
        outB = face_parser.parse_pil(pilB)

        # Save label maps
        labelA_path = outdir / "parse_label_A.npy"
        labelB_path = outdir / "parse_label_B.npy"
        np.save(labelA_path, outA.label_map)
        np.save(labelB_path, outB.label_map)

        # Save a quick overlay image for sanity-checking
        imgA_np = np.array(pilA, dtype=np.uint8)
        imgB_np = np.array(pilB, dtype=np.uint8)
        overlayA = overlay_label_map(imgA_np, outA.label_map)
        overlayB = overlay_label_map(imgB_np, outB.label_map)
        Image.fromarray(overlayA).save(outdir / "parse_overlay_A.png")
        Image.fromarray(overlayB).save(outdir / "parse_overlay_B.png")

        # Map label names to ids
        name_to_id: Dict[str, int] = {v: k for k, v in outA.id2label.items()} if outA.id2label else {}

        # Small interpretable set of parts (must match id2label strings exactly)
        parts = ["skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "hair"]

        masksA = face_parser.masks_from_label_map(outA.label_map, parts, name_to_id)
        masksB = face_parser.masks_from_label_map(outB.label_map, parts, name_to_id)

        masksA_path = outdir / "parse_masks_A.npz"
        masksB_path = outdir / "parse_masks_B.npz"
        np.savez_compressed(masksA_path, **masksA)
        np.savez_compressed(masksB_path, **masksB)

        pipeline_result.parsing = {
            "A": ParseResult(
                label_map_path=str(labelA_path.resolve()),
                parts=parts,
                masks_path=str(masksA_path.resolve()),
                parsing_model=args.parsing_model,
                parsing_confidence=None,
            ),
            "B": ParseResult(
                label_map_path=str(labelB_path.resolve()),
                parts=parts,
                masks_path=str(masksB_path.resolve()),
                parsing_model=args.parsing_model,
                parsing_confidence=None,
            ),
        }

        print("Parsing id2label keys (sample):", list(outA.id2label.items())[:10])

    # Save final result
    save_json(pipeline_result.to_dict(), str(outdir / "result.json"))
    print(json.dumps(pipeline_result.to_dict(), indent=2))


if __name__ == "__main__":
    main()