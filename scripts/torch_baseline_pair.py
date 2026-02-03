import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from src.contracts.io import save_json
from src.contracts.types import FacePair, FVResult, PipelineResult, ParseResult
from src.models.parsing.hf_face_parsing import HFFaceParser


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.inference_mode()
def align_face(mtcnn: MTCNN, img: Image.Image) -> torch.Tensor:
    """
    Returns a single aligned face tensor (3, 160, 160).
    Raises ValueError if no face is detected.
    """
    face = mtcnn(img)
    if face is None:
        raise ValueError("No face detected.")
    return face


@torch.inference_mode()
def embed_face(resnet: InceptionResnetV1, face: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    face: (3,160,160) float tensor
    returns embedding: (512,)
    """
    face = face.unsqueeze(0).to(device)  # (1,3,160,160)
    emb = resnet(face)                   # (1,512)
    emb = F.normalize(emb, p=2, dim=1)   # normalize for cosine
    return emb.squeeze(0).detach().cpu() # (512,)


def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    return float(torch.dot(e1, e2).item())


def save_tensor_as_image(face: torch.Tensor, out_path: Path) -> None:
    """
    face is (3,160,160) in roughly [0,1]. Save as PNG for debugging.
    """
    x = face.detach().cpu().clamp(0, 1).numpy()
    x = (np.transpose(x, (1, 2, 0)) * 255.0).astype(np.uint8)
    Image.fromarray(x).save(out_path)


def overlay_label_map(img_rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Simple overlay for sanity-checking parsing.
    img_rgb: HxWx3 uint8
    label_map: HxW int
    Returns HxWx3 uint8.
    """
    # deterministic pseudo-colors by label id
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
    parser.add_argument("--outdir", type=str, default="results/torch_baseline_run", help="Output directory")
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

    # MTCNN does detection + alignment; InceptionResnetV1 gives embeddings
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

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

    # Embeddings + similarity
    embA = embed_face(resnet, faceA, device)
    embB = embed_face(resnet, faceB, device)

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
        label=None
    )

    fv = FVResult(
        model="facenet-pytorch/InceptionResnetV1(vggface2)",
        embeddingA_path=str(embA_path.resolve()),
        embeddingB_path=str(embB_path.resolve()),
        similarity_cosine=sim,
        threshold=args.threshold,
        decision=decision,
        device=str(device),
        backend="pytorch"
    )

    pipeline_result = PipelineResult(pair=pair, fv=fv)

    # -------------------------
    # Step 4: optional face parsing
    # -------------------------
    if args.do_parsing:
        face_parser = HFFaceParser(model_name=args.parsing_model, device=str(device))

        pilA = Image.open(alignedA_path).convert("RGB")
        pilB = Image.open(alignedB_path).convert("RGB")

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

        # Start with a small, interpretable set of parts.
        # NOTE: label names must match the model's id2label exactly.
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

        # Optional: print the model's label mapping once for debugging
        # (so you can adjust "parts" names if needed)
        print("Parsing id2label keys (sample):", list(outA.id2label.items())[:10])

    # Save final result
    save_json(pipeline_result.to_dict(), str(outdir / "result.json"))
    print(json.dumps(pipeline_result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
