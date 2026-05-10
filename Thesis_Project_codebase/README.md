# Semantic Part-Based Explainability for Face Verification

This repository contains the implementation of a face verification pipeline with semantic segmentation explainability. Given two face images (a reference and a probe), the system produces a match/non-match decision and highlights which facial regions contributed most to that decision.

The pipeline is built around three main components:

- **AdaFace** (IR-50, pretrained on MS1MV2) for face embedding and cosine similarity scoring
- **Grad-CAM** on the deepest spatial convolutional layer for pixel-level attribution
- **SegFormer** (fine-tuned on CelebAMask-HQ) for semantic face parsing into 8 regions: skin, nose, left/right eye, left/right eyebrow, mouth, and hair

Region-level importance is measured via semantic occlusion: each parsed region is occluded (gray fill or Gaussian blur) and the resulting similarity shift quantifies that region's contribution to the overall score. Two independent occlusion methods (gray and blur) are used, and their cross-method agreement (Spearman ρ) serves as a built-in reliability indicator.


## Repository Structure

```
Thesis_Project_codebase/
├── README.md
├── requirements.txt
│
├── pretrained/
│   ├── adaface_ir50_ms1mv2.ckpt       # AdaFace IR-50 weights (MS1MV2)
│   └── threshold_lfw.json             # Calibrated threshold (FMR=0.001)
│
├── src/
│   ├── contracts/
│   │   ├── types.py                   # Dataclass definitions for pipeline I/O
│   │   └── io.py                      # JSON save/load helpers
│   └── models/
│       ├── fr/
│       │   └── adaface.py             # AdaFace wrapper (loading, preprocessing, embedding)
│       ├── parsing/
│       │   └── hf_face_parsing.py     # SegFormer face parser
│       └── explain/
│           └── gradcam.py             # Grad-CAM for pairwise similarity
│
├── scripts/
│   ├── run_pair.py                    # Main pipeline (single pair)
│   ├── dataset_batch_runner.py        # Batch runner over a manifest CSV
│   ├── calibrate_threshold.py         # Threshold calibration on LFW
│   ├── fidelity_eval.py               # Explanation fidelity evaluation
│   ├── make_operator_figure.py        # Operator-facing figure generation
│   ├── generate_gorilla_manifest.py   # Build manifest for Gorilla 41-pair set
│   └── generate_hda_manifest.py       # Build manifest for HDA Doppelgänger set
│
├── assets/
│   ├── demo_images/                   # Sample images for quick testing
│   ├── gorilla/
│   │   └── gor_manifest.csv           # Gorilla curated set (41 pairs)
│   ├── hda/
│   │   └── hda_manifest.csv           # HDA Doppelgänger (399 pairs)
│   └── lfw/
│       └── lfw_300_manifest.csv       # LFW 300-pair evaluation subset
│
├── gorilla/                           # Gorilla batch results
│   ├── accuracy_report.txt
│   ├── review_table.csv
│   ├── fidelity_results.csv
│   ├── fidelity_summary.txt
│   └── fidelity_plots.png
│
├── hda/                               # HDA batch results
│   ├── hda_accuracy_report.txt
│   ├── hda_review_table.csv
│   └── hda_review_table.txt
│
└── lfw/                               # LFW batch results
    ├── lfw_accuracy_report.txt
    ├── lfw_review_table.csv
    ├── fidelity_results.csv
    ├── fidelity_summary.txt
    └── fidelity_plots.png


## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended; CPU works but is slow)

### Installation

With pip:
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yml
conda activate xface_thesis
```

### Third-party model setup

1. **AdaFace**: Clone the AdaFace repository into `third_party/adaface/` and download the IR-50 checkpoint:

```bash
mkdir -p third_party
git clone https://github.com/mk-minchul/AdaFace.git third_party/adaface
mkdir -p pretrained
# Place adaface_ir50_ms1mv2.ckpt in pretrained/
```

2. **SegFormer**: The face parsing model (`jonathandinu/face-parsing`) is downloaded automatically from HuggingFace on first use.

3. **MTCNN**: Provided by `facenet-pytorch`, no manual setup needed.


## Usage

### Running the full pipeline on a pair

```bash
python scripts/run_pair.py \
    --imgA path/to/reference.jpg \
    --imgB path/to/probe.jpg \
    --outdir results/my_pair \
    --do_gradcam \
    --do_parsing \
    --do_part_signals \
    --do_operator_overlay
```

This will:
1. Detect and align both faces with MTCNN (112×112 crops)
2. Compute AdaFace embeddings and cosine similarity
3. Generate a Grad-CAM heatmap on the probe image
4. Parse both faces into semantic regions
5. Run per-region occlusion scoring (gray + blur)
6. Produce the operator-facing explanation figure

Output files in the result directory:
- `result.json` — full pipeline output with all scores and metadata
- `aligned_A.png`, `aligned_B.png` — MTCNN-aligned crops
- `gradcam_B.npy`, `gradcam_overlay_B.png` — Grad-CAM heatmap
- `parse_masks_A.npz`, `parse_masks_B.npz` — per-region binary masks
- `operator_prototype_probe_only.png` — three-panel explanation figure
- `operator_summary.txt` — plain-text explanation

### Threshold calibration

The decision threshold is calibrated on LFW at a target FMR of 0.001:

```bash
python scripts/calibrate_threshold.py \
    --images_dir /path/to/lfw/ \
    --num_impostor_pairs 2000000 \
    --num_genuine_pairs 80000 \
    --far_target 0.001 \
    --seed 123
```

This saves `pretrained/threshold_lfw.json`, which `run_pair.py` picks up automatically.

### Generating dataset manifests

Before running the batch runner, generate a manifest CSV for your dataset:

```bash
# Gorilla curated set (41 pairs)
python scripts/generate_gorilla_manifest.py \
    --img_dir dataset_for_gorilla/resized_512_crop \
    --out assets/gorilla/manifest.csv

# HDA Doppelgänger (399 non-mated pairs)
python scripts/generate_hda_manifest.py \
    --hda_dir dataset_for_gorilla/HDA \
    --out assets/hda/manifest.csv
```

### Running on a full dataset

To process every pair in a manifest CSV (e.g. the Gorilla curated set or HDA Doppelgänger):

```bash
python scripts/dataset_batch_runner.py \
    --manifest assets/gorilla/manifest.csv \
    --outdir results/gorilla \
    --dataset gorilla \
    --do_gradcam \
    --device cuda
```

The manifest CSV must have columns: `pair_id`, `category`, `ground_truth`, `img_a`, `img_b`. The batch runner calls `run_pair.py` as a subprocess for each pair, then aggregates results into `review_table.csv` and `accuracy_report.txt`.

Use `--skip_existing` to resume interrupted runs without re-processing completed pairs.

### Fidelity evaluation

To evaluate whether the occlusion-based explanations are internally consistent (i.e. whether importance rankings generalise across gray and blur perturbation methods):

```bash
python scripts/fidelity_eval.py \
    --manifest assets/gorilla/manifest.csv \
    --results results/gorilla \
    --outdir results/fidelity \
    --device cuda
```

This produces:
- `fidelity_results.csv` — per-pair metrics
- `fidelity_summary.txt` — aggregate statistics
- `fidelity_plots.png` — diagnostic histograms


## Pipeline Overview

### Step 1: Face Detection and Alignment
MTCNN detects faces and produces 112×112 aligned crops. Both reference and probe are processed identically.

### Step 2: Embedding and Comparison
AdaFace (IR-50) maps each aligned crop to a 512-dimensional L2-normalised embedding. The comparison score is the cosine similarity between the two embeddings. A decision margin and confidence label (high/medium/low) are derived from the distance to the calibrated threshold.

AdaFace also returns a quality norm for each image, which is reported as the biometric sample quality score.

### Step 3: Grad-CAM Attribution
Grad-CAM is computed on the probe image with respect to the cosine similarity to the reference embedding. The heatmap highlights which spatial regions of the probe had the most influence on the similarity score. Only the probe-side map is generated since the reference is shown clean to the operator.

### Step 4: Semantic Parsing
SegFormer segments each face into semantic regions. The raw labels are mapped to 8 canonical parts (with `u_lip` and `l_lip` merged into `mouth`).

### Step 5: Per-Region Occlusion Scoring
For each semantic region on the probe, the region is occluded and the similarity is recomputed. The delta (occluded minus baseline) tells us whether that region was contributing to similarity (delta < 0) or dissimilarity (delta > 0). This is done twice — once with gray fill and once with Gaussian blur — to check robustness.

The cross-method Spearman ρ between gray and blur deltas across parts serves as a reliability indicator: high ρ means the explanation is stable regardless of the perturbation method used.

### Step 6: Operator Figure
A three-panel figure is generated showing the reference (clean), the probe (with Grad-CAM overlay and colour-coded contours), and a text panel with the decision summary, evidence breakdown, reliability assessment, and operational guidance.

Green contours mark regions contributing to similarity, red contours mark regions contributing to dissimilarity. The colour assignment is absolute (not decision-dependent), but the text panel interprets them in the context of the decision.
