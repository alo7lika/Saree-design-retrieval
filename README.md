# Saree surface-design retrieval

**Approach note (470 characters):** ImageNet-pretrained EfficientNet-B0 embeds 224² cloth crops as normalized 256-D vectors. Decode RGB; resize 256, center-crop 224, convert to luminance, replicate channels and ImageNet-normalize. Train with supervised contrastive loss and P×K identity batches; augment crops, flips, rotations, brightness/contrast, hue/saturation and blur. Rank by cosine; calibrate verification threshold on validation pairs. Grayscale suppresses palette while retaining motif structure.

This directory contains a runnable PyTorch implementation for design retrieval, pair verification, training and evaluation. Images are not included. The code uses the project's Kaggle and DeepLure corpora plus torchvision's public ImageNet EfficientNet-B0 weights (`IMAGENET1K_V1`). DeepLure is proprietary: do not redistribute it, and delete your local copy when the exercise ends.

## Supplied archive audit and prepared labels

The supplied Kaggle dataset ([Indian Saree Patterns](https://www.kaggle.com/datasets/div456/indian-saree-patterns)) is represented by `archive.zip`: 1,468 images in Banarasi, Bandhani, Ikat and Pichwai folders, split into 1,293 train, 115 validation and 60 test images. These are broad pattern-category labels, not exact design IDs. `manifest_indian_patterns.csv` supports a category-proxy experiment only. Kaggle lists CC0, while the embedded Roboflow README says MIT; source metadata conflicts, so no source images or derived image bundle are redistributed.

The project also supplies the [DeepLure Drive corpus](https://drive.google.com/open?id=1V_DcXJ50QYV9nEqvq7fqKno_lhbSPdBM). Its `handloom_sarees` folder contains 165 images without design/colorway metadata; `normal_sarees` is empty. I visually reviewed the handloom images and created provisional motif groups for 99 images (20 design IDs); 66 ambiguous or unmatched images were excluded. `deeplure_visual_labels.csv` records the manual assignments. Group identities are disjoint across train (11 designs), validation (5), and test (4). In test, gallery/query share the four held-out design IDs; three designs also have manually separated palette groups, yielding 16 cross-palette positive pairs. This small visual pseudo-label benchmark is exploratory, not expert-verified ground truth.

To reproduce the measured DeepLure run in Kaggle, add the handloom images, `manifest_deeplure_visual.csv`, and code files to the notebook, then run:

```bash
python train.py --manifest /kaggle/input/deeplure/manifest_deeplure_visual.csv \
  --data-root /kaggle/input/deeplure/handloom_sarees --output-dir /kaggle/working/run \
  --epochs 5 --batch-size 32 --batches-per-epoch 10 --workers 0 --finetune-last-blocks 2
python evaluate.py --manifest /kaggle/input/deeplure/manifest_deeplure_visual.csv \
  --data-root /kaggle/input/deeplure/handloom_sarees --checkpoint /kaggle/working/run/best.pt \
  --output-dir /kaggle/working/run/eval
```

## Data manifest

Create `manifest.csv` with columns `path,design_id,colorway_id,split`. Paths may be absolute or relative to `--data-root`. Each row is one image. `design_id` names the actual surface design (shared by all palette variants); `colorway_id` identifies a palette/photographic variant. Recommended split values are `train`, `gallery`, `query`, and optionally `val`. For color-invariance training, include at least two distinct colorways per design where possible; gallery and query should contain different colorways of the same held-out designs. The supplied corpora lack train colorway labels, so the prepared run cannot enforce cross-colorway positive sampling. Keep all near-duplicates and images from one physical product/session in one colorway group. Review ambiguous labels before training.

Neither source contains exact design IDs. The prepared DeepLure manifest uses visually checked motif groups and hand-assigned palette groups for three test designs. Review the per-file decisions in `deeplure_visual_labels.csv`; they are provisional labels, not an official annotation. Unmatched/ambiguous images are excluded from that benchmark.

## Kaggle setup and training

Upload the project and the dataset to a Kaggle Notebook (GPU recommended), then install/use PyTorch, torchvision, pandas, Pillow and scikit-learn. Make the manifest accessible and run:

```bash
python train.py --manifest /kaggle/input/my-data/manifest.csv \
  --data-root /kaggle/input/my-data --output-dir /kaggle/working/run \
  --epochs 20 --batch-size 32
```

The sampler forms batches with several examples per design; supervised contrastive loss attracts same-design embeddings and repels other designs. The default CPU-friendly fine-tuning policy updates the final two EfficientNet feature blocks and projection; change `--finetune-last-blocks` to tune more or fewer blocks (0 trains the projection only). If a batch contains only one unique design, training fails fast. Save `best.pt` by validation nearest-neighbor Recall@1 when `val` rows exist, otherwise by training loss (the latter is not model-selection evidence). Set `--pretrained false` for offline environments without cached weights. The first pretrained run needs internet/cache access to fetch torchvision weights.

## Inference and evaluation

```bash
python evaluate.py --manifest manifest.csv --data-root /path/to/images \
  --checkpoint run/best.pt --output-dir run/eval
python infer.py --checkpoint run/best.pt --gallery-csv gallery.csv \
  --query query.jpg --data-root /path/to/images --top-k 5
python verify.py --checkpoint run/best.pt --image-a a.jpg --image-b b.jpg \
  --threshold run/eval/threshold.json
```

Evaluation reports query-to-gallery Recall@1/5, mAP, and pair verification ROC-AUC, equal-error rate and thresholded accuracy. The threshold is selected on validation pairs only and stored in `threshold.json`; if there is no validation split, the script uses a clearly labeled development subset of query pairs and its reported verification result is optimistic. Prefer a disjoint validation split for final reporting. Gallery/query split is image-manifest-driven; inspect the generated `split_audit.json` for design/colorway overlap. The script refuses designs absent from gallery and can audit whether query/gallery colorways are disjoint.

## Evaluation protocol and result status

Use a design-disjoint train/validation/test split where corpus size permits; within each held-out design, put one or more colorways in gallery and distinct colorways in query. For small corpora, disclose that the design identities recur across gallery/query but the physical/colorway groups do not. Report the number of designs, colorways and images per split, class imbalance, exclusions and any repeated photos. Identification metrics are macro-averaged by query design; verification uses balanced same/different pairs and reports ROC-AUC plus EER and thresholded accuracy at a validation-selected threshold. Add a palette-matched hard-negative set (different designs with similar colors) and cross-colorway positive set.

**Measured results:** [RESULTS.md](RESULTS.md) reports both the small provisional DeepLure visual-design benchmark and the secondary Kaggle category-proxy result. Treat the manually curated DeepLure result as exploratory because its labels and test set are small.

## Efficiency

The backbone is torchvision EfficientNet-B0 plus a 1280→256 linear projection and L2 normalization. `train.py` prints parameter count; `evaluate.py` writes model parameters, embedding size and measured batch-1 latency to `efficiency.json` on the run device. FLOPs are omitted because profiler FLOP conventions differ; latency is meaningful only alongside device, input size and batch size.
