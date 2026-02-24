import numpy as np
import torch
from src.models.fr.adaface import AdaFaceEmbedder, AdaFaceConfig

def main():
    ada = AdaFaceEmbedder(AdaFaceConfig(
        architecture="ir_50",
        ckpt_path="pretrained/adaface_ir50_ms1mv2.ckpt",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ))

    dummy = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    x = ada.preprocess_rgb_uint8(dummy)
    emb = ada.embed(x)
    print("embedding shape:", emb.shape, "norm:", float(torch.linalg.norm(emb)))

if __name__ == "__main__":
    main()