# Dataset audit and evaluation status

## Project sources

The Kaggle link in the project document points to *Indian Saree Patterns*: 1,000+ images in four folder classes. The supplied `archive.zip` is the corresponding Roboflow export with 1,468 images: 1,293 train, 115 validation, 60 test across Banarasi, Bandhani, Ikat and Pichwai. The labels describe broad pattern categories, not exact surface designs or colorways. The Kaggle page lists CC0, while the embedded export README says MIT; the metadata conflict is documented and no source images are redistributed.

The specified DeepLure Drive folder contains 165 `handloom_sarees` images and an empty `normal_sarees` folder. The images have no design/colorway labels. DeepLure is proprietary; its images remain only in temporary workspace storage and must be deleted once the exercise concludes.

## Visual relabeling for a design-level proof of concept

After viewing contact sheets and checking likely motif matches, I assigned provisional design IDs to repeated visual motifs among 99 handloom images. Twenty design groups were created; 66 images without a confident match were excluded. The split is by design ID: 11 train IDs (54 images), 5 validation IDs (25 images), and 4 held-out test IDs (20 images). The held-out test set has an 8-image gallery and 12-image query split. Three test designs have separately annotated gallery/query palette groups, forming 16 cross-palette positive pairs; one test design has no palette annotations. The row-level labels and split assignments are in `deeplure_visual_labels.csv` and `manifest_deeplure_visual.csv`.

These are visual pseudo-labels made for this exercise, not labels supplied by DeepLure or checked by a textile expert. The small and curated set can overstate accuracy. Report it as exploratory evidence; it does not establish production-level accuracy or broad generalization. For a stronger benchmark, have an expert independently verify design and palette groups and gather many more products/colorways.

## Measured results

The visually labeled DeepLure benchmark produced Recall@1 1.0000, Recall@5 1.0000, mAP 0.9778, gallery-query verification ROC-AUC 0.9711, and cross-palette verification ROC-AUC 0.9805 on only 16 cross-palette positives. Threshold selection used validation pairs only. Exact values and pair counts are in `RESULTS.md` and `deeplure_*.json`.

The separate Kaggle category proxy produced Recall@1 0.8065, Recall@5 1.0000, mAP 0.7446, gallery-query ROC-AUC 0.8831, EER 0.1974, and thresholded accuracy 0.8004. These values are category recognition, not exact design retrieval.

Efficiency: 4,335,484 parameters, 256-D embeddings, and measured CPU batch-1 p50 latency in `deeplure_efficiency.json` (224×224 input, model only). FLOPs were not measured.
