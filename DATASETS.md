# Dataset Preparation

This repository does not include image datasets, trained weights, or generated dataset copies.

## Class Scheme

The unified six-class scheme is:

| ID | Class |
|---:|---|
| 0 | abnormal |
| 1 | empty |
| 2 | overripe |
| 3 | ripe |
| 4 | under_ripe |
| 5 | unripe |

The external `preprocessed-ffb` evaluation uses a four-class masked protocol:

| Evaluated | Ignored |
|---|---|
| abnormal, ripe, under_ripe, unripe | empty, overripe |

## Expected Local Inputs

The full pipeline expects local paths for:

- `TRAINED_MODEL`: a YOLOv8 checkpoint such as `best.pt`
- `ORIGINAL_DATASET`: labeled source-domain oil palm FFB dataset
- `NEW_IMAGES`: unlabeled or weakly labeled images for pseudo-labeling
- `EXTERNAL_TEST_DATASET`: external preprocessed FFB dataset

Use the command-line arguments in `src/full_ssod_ffb_pipeline.py` and related runner scripts to point to your local dataset directories.

## Reproducibility Notes

- Keep external test images locked and out of training.
- Do not mix adjacent video frames across train/test when reporting scene-disjoint results.
- Use the four-class masked external protocol when the external dataset lacks `empty` and `overripe`.
- Keep generated prepared datasets outside Git, or archive them separately on Zenodo/OneDrive if required.

