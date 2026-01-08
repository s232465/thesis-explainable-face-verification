import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


@torch.inference_mode()
def align_face(mtcnn: MTCNN, img: Image.Image) -> torch.Tensor:
    """
    Returns a single aligned face tensor (3, 160, 160) in range [0, 1] normalized by MTCNN internal transforms.
    Raises ValueError if no face is detected.
    """
    face = mtcnn(img)  # either Tensor [3,160,160] or None (or batch if keep_all=True)
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
    # embeddings are already normalized
    return float(torch.dot(e1, e2).item())


def save_tensor_as_image(face: torch.Tensor, out_path: Path) -> None:
    """
    face is (3,160,160) in roughly [0,1]. Save as PNG for debugging.
    """
    x = face.detach().cpu().clamp(0, 1).numpy()
    x = (np.transpose(x, (1, 2, 0)) * 255.0).astype(np.uint8)
    Image.fromarray(x).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgA", type=str, required=True, help="Path to probe image")
    parser.add_argument("--imgB", type=str, required=True, help="Path to reference image")
    parser.add_argument("--outdir", type=str, default="results/torch_baseline_run", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine threshold for match decision (placeholder)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # MTCNN does detection + alignment; InceptionResnetV1 gives embeddings
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    imgA = load_image(args.imgA)
    imgB = load_image(args.imgB)

    # Align faces
    faceA = align_face(mtcnn, imgA)
    faceB = align_face(mtcnn, imgB)

    # Save aligned crops (debug artifact)
    save_tensor_as_image(faceA, outdir / "aligned_A.png")
    save_tensor_as_image(faceB, outdir / "aligned_B.png")

    # Embeddings + similarity
    embA = embed_face(resnet, faceA, device)
    embB = embed_face(resnet, faceB, device)

    sim = cosine_similarity(embA, embB)
    decision = "match" if sim >= args.threshold else "non-match"

    result: Dict[str, Any] = {
        "imgA": str(Path(args.imgA).resolve()),
        "imgB": str(Path(args.imgB).resolve()),
        "model": "facenet-pytorch/InceptionResnetV1(vggface2)",
        "similarity_cosine": sim,
        "threshold": args.threshold,
        "decision": decision,
        "device": str(device),
        "torch_version": torch.__version__,
    }

    # Save embeddings (optional, but useful later)
    np.save(outdir / "embedding_A.npy", embA.numpy())
    np.save(outdir / "embedding_B.npy", embB.numpy())

    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
