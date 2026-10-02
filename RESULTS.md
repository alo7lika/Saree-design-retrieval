# Measured results

## Primary: provisional DeepLure visual-design benchmark

The DeepLure Drive `handloom_sarees` folder contains 165 unlabelled images; `normal_sarees` is empty. I visually grouped 99 images into 20 repeated-motif IDs and excluded 66 images that were unmatched or ambiguous. The annotations are provisional visual labels rather than corpus ground truth. Three held-out test IDs have palette groups manually assigned from the images.

| Split | Design IDs | Images | Protocol |
|---|---:|---:|---|
| Train | 11 | 54 | Supervised contrastive training |
| Validation | 5 | 25 | Nearest-neighbor Recall@1 selection; verification threshold calibration |
| Test gallery | 4 | 8 | One reference or more per held-out design |
| Test query | 4 | 12 | Includes query palettes disjoint from gallery for three designs |

Train, validation, and test design IDs are disjoint. Gallery and query intentionally share the four test design IDs. The evaluation contains 22 positive gallery-query design pairs and 74 different-design pairs. The cross-palette subset contains 16 positive pairs (three designs) and 74 possible negatives; metrics below balance these to 16 pairs per class. The threshold is selected on validation pairs: cosine 0.35768.

| Task | Metric | Result |
|---|---|---:|
| Identification | Recall@1 | 1.0000 |
| Identification | Recall@5 | 1.0000 |
| Identification | mAP | 0.9778 |
| Gallery-query verification | ROC-AUC | 0.9711 |
| Gallery-query verification | EER | 0.0909 |
| Gallery-query verification | Accuracy at validation threshold | 0.8409 |
| Cross-palette verification | ROC-AUC | 0.9805 |
| Cross-palette verification | EER | 0.0625 |
| Cross-palette verification | Accuracy at validation threshold | 0.9063 |
| Validation selection | Nearest-neighbor Recall@1 | 1.0000 |

This is a small curated proof-of-concept, not a reliable estimate of deployment accuracy. The 99-image visual grouping and palette assignments have not been checked by a textile expert; selecting visually obvious repeated motifs can bias scores upward. The cross-palette result has only 16 positives from three designs. It provides initial evidence that the embedding handles these examples, not a definitive validation of the full color-invariance requirement. See `deeplure_visual_labels.csv` for every included file and assignment.

Training used ImageNet-1K pretrained EfficientNet-B0, grayscale preprocessing, supervised contrastive loss, 5 epochs × 10 identity-balanced batches, batch size 32, CPU, and fine-tuned the final two feature blocks plus projection. The checkpoint was selected by validation nearest-neighbor Recall@1.

## Secondary: specified Kaggle broad-category proxy

The Kaggle dataset has four pattern-category labels, not design identities. On its 31-query category proxy split, the earlier run obtained Recall@1 0.8065, Recall@5 1.0000, mAP 0.7446, gallery-query verification ROC-AUC 0.8831, EER 0.1974, and thresholded accuracy 0.8004 (validation-calibrated threshold 0.21545). This does not test exact-design color invariance. `manifest_indian_patterns.csv` and `proxy_*.json` document this secondary result.

## Efficiency

| Measure | Result |
|---|---:|
| Parameters | 4,335,484 |
| Embedding | 256 dimensions |
| Batch-1 model-only latency | 102.85 ms p50; 119.55 ms mean (CPU, 4 threads, 224×224 input) |

Latency covers 100 model-only forwards after warmup and excludes decode/preprocessing. FLOPs were not measured. Numbers depend on CPU and runtime. The image data and DeepLure-trained checkpoint are not redistributed; the checkpoint remains in temporary workspace storage for local reproduction.
