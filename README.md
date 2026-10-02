# 🧵 Saree Design ID

### Find the weave of a design—across changing color palettes.

A PyTorch image-retrieval prototype that embeds saree surface patterns and ranks reference images by design similarity. It supports **gallery identification** and **same-design verification**, with grayscale preprocessing intended to reduce reliance on color.

> **Project status:** runnable research prototype. The supplied DeepLure labels below are provisional visual annotations; results are exploratory and should not be read as production accuracy.

---

## ✨ What it does

- 🔎 **Identification:** ranks a gallery for each query using cosine similarity.
- 🤝 **Verification:** predicts whether a pair shares a design, using a threshold calibrated on validation pairs.
- 🎨 **Palette robustness:** converts inputs to luminance before embedding, emphasizing motif structure over color.
- 📊 **Evaluation:** reports Recall@1/5, mAP, verification ROC-AUC, EER, thresholded accuracy, split audit, and inference timing.

## 🧠 Approach

**Approach note (470 characters):** ImageNet-pretrained EfficientNet-B0 embeds 224² cloth crops as normalized 256-D vectors. Decode RGB; resize 256, center-crop 224, convert to luminance, replicate channels and ImageNet-normalize. Train with supervised contrastive loss and P×K identity batches; augment crops, flips, rotations, brightness/contrast, hue/saturation and blur. Rank by cosine; calibrate verification threshold on validation pairs. Grayscale suppresses palette while retaining motif structure.

Backbone weights: torchvision EfficientNet-B0 `IMAGENET1K_V1`. Training uses supervised contrastive loss with identity-balanced batches. See `train.py`, `data.py`, and `model.py` for implementation details.

## 📈 Results

Primary benchmark: manually grouped DeepLure handloom images. The corpus has no design labels, so 99 of 165 images were provisionally grouped into 20 repeated-motif IDs; 66 ambiguous or unmatched images were excluded. Train/validation/test design IDs are disjoint. The held-out test has 8 gallery images and 12 queries across 4 designs.

| Evaluation | Result |
|---|---:|
| Identification Recall@1 / Recall@5 | 1.000 / 1.000 |
| Identification mAP | 0.978 |
| Gallery-query verification ROC-AUC | 0.971 |
| Gallery-query verification EER | 0.091 |
| Cross-palette verification ROC-AUC | 0.980 |
| Cross-palette positives | 16 pairs across 3 designs |

These scores come from a small, curated set with labels assigned by visual review, not textile experts. Obvious repeated motifs may be overrepresented. The cross-palette subset is especially small; treat these results as an initial demonstration, not a robust generalization claim. Full metrics, protocol, and caveats: [`RESULTS.md`](RESULTS.md) and [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md).

A secondary evaluation on the Kaggle dataset uses four broad pattern categories, not exact design identities. Its metrics are a category proxy and do not prove color-invariant exact-design retrieval.

## 🚀 Run locally (VS Code / PowerShell)

Open this folder in VS Code, then open **Terminal → New Terminal**. The commands below assume the supplied handloom archive has been extracted so the JPG files are directly inside `work/data/handloom_sarees/handloom_sarees` relative to the project root. Adjust `--data-root` if your images are elsewhere.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe train.py `
  --manifest manifest_deeplure_visual.csv `
  --data-root ..\..\work\data\handloom_sarees\handloom_sarees `
  --output-dir run `
  --epochs 5 --batch-size 32 --batches-per-epoch 10 `
  --workers 0 --finetune-last-blocks 2

.\.venv\Scripts\python.exe evaluate.py `
  --manifest manifest_deeplure_visual.csv `
  --data-root ..\..\work\data\handloom_sarees\handloom_sarees `
  --checkpoint run\best.pt `
  --output-dir run\eval
```

On first training, torchvision may download pretrained weights, so internet access is needed unless they are already cached. CPU training works but may take longer; a GPU runtime such as Kaggle can speed it up.

### 🔍 Use a trained model

```powershell
python infer.py --checkpoint run\best.pt --gallery-csv gallery.csv `
  --query path\to\query.jpg --data-root path\to\images --top-k 5

python verify.py --checkpoint run\best.pt --image-a a.jpg --image-b b.jpg `
  --threshold run\eval\threshold.json
```

See `manifest.example.csv` for the manifest format. For a new dataset, create rows with `path,design_id,colorway_id,split`; paths can be relative to `--data-root`.

## 🗂️ Repository contents

- `train.py` — supervised-contrastive training and checkpoint selection.
- `evaluate.py`, `metrics.py` — retrieval, verification, audit, and efficiency evaluation.
- `infer.py`, `verify.py` — gallery ranking and pair verification.
- `model.py`, `data.py`, `common.py` — architecture, transforms, manifests, and utilities.
- `manifest_deeplure_visual.csv`, `deeplure_visual_labels.csv` — provisional labels and split assignments; no images are included.
- `manifest_indian_patterns.csv` — broad-category proxy manifest.
- `*_metrics.json`, `*_split_audit.json`, `*_efficiency.json` — machine-readable reports.

## 🔐 Dataset and reproducibility notes

- **DeepLure images are proprietary. Do not upload or redistribute the images or their ZIP archives.** Obtain them only through the source link in the project brief, and remove your local copy when the exercise is over.
- This repository contains code, manifests, and reported metrics—not the image corpora or trained model weights.
- The provided Kaggle export has broad category labels and cannot by itself supervise exact design identity.
- Training/evaluation values can vary with package versions, hardware, and nondeterministic operations. The included JSON files record the reported run.
- No FLOP count was measured. The reported model has 4,335,484 parameters, 256-D embeddings, and 102.85 ms CPU batch-1 p50 model-only latency in the recorded environment.

## 📦 Dependencies

Python packages are listed in [`requirements.txt`](requirements.txt): PyTorch, torchvision, pandas, Pillow, NumPy, and scikit-learn.

---

**Built for the saree-design retrieval assignment** · PyTorch · Image retrieval · Metric learning
