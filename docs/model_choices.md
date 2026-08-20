# Model choices

This document records the models used by the public reconstruction, where
they came from, and how they are loaded.

## Bundled: YOLO26 bubble detector

`Yolo26BubbleDetector` (`src/toontra/modules/yolo26_bubble_detector.py`) is
the default and only bundled `BubbleDetector` implementation. Its checkpoint,
`src/toontra/weights/speech_bubble_yolo26s.pt`, was trained for this public
rewrite with Ultralytics, initialized from the official pretrained `yolo26s.pt`,
and fine-tuned on the public Roboflow "speech-bubbles-detection" dataset
(manga pages; several annotated bubble-shape classes merged into Toontra's
single `speech_bubble` label). See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the Ultralytics license.

`Toontra`'s default processing chain for this detector is: tiled inference on
tall pages, cross-tile ownership, cross-tile NMS, then one ROI
expansion of each surviving box (`Toontra(detection_expansion_ratio=0.05)`,
the default). The ratio is applied independently per axis -- left/right
padding equals 5% of the box's own width, top/bottom padding equals 5% of its
own height -- giving roughly 10% width and 10% height growth. This reproduces
the ROI expansion of the original, unpublished pipeline this repository
rebuilds. Expansion occurs after deduplication so IoU comparisons use the
original boxes. Nothing downstream, including `ManetBubbleMasker`, expands the
box again; see
[custom_models.md](custom_models.md#optional-ma-net-masker).

Training-run numbers (from the raw Kaggle artifacts) and an independent,
manually annotated webtoon benchmark are both under
[evaluation/](../evaluation/); the README's Evaluation section summarizes
both, kept as separate tables since they measure different things.

## Bubble masking

`WhiteBubbleMasker` is a deterministic, weight-free OpenCV baseline: it looks
for the white bubble interior around a detection and fills it (and any text
holes inside it). It works well for light bubbles with a closed outline and
needs no model download, GPU, or account.

## Optional: MA-Net bubble masking

`ManetBubbleMasker` (`src/toontra/modules/manet_bubble_masker.py`) is the
learned alternative to `WhiteBubbleMasker`. Its separately distributed
checkpoint, `toontra_manet_resnet34_bubble_segmentation.pth`, was trained for
this public reconstruction using Segmentation Models PyTorch's `MAnet` with a
`resnet34` encoder, fine-tuned on the public Roboflow
[`manga-segment_v2`](https://universe.roboflow.com/), version 5,
speech-bubble segmentation dataset. It is a new checkpoint produced by this
training run, not the original production checkpoint.

Evaluation numbers from
[docs/training/manet_results.json](training/manet_results.json) at the best
checkpoint epoch:

| Metric | Value |
| --- | --- |
| Validation Dice | 0.9828 |
| Test Dice | 0.9805 |
| Test IoU | 0.9622 |
| Decision threshold | 0.45 (stored in the checkpoint) |

Full per-epoch training history is in
[docs/training/manet_training_history.csv](training/manet_training_history.csv).
These results apply to the recorded training run and dataset split.

The adapter and its runtime (`torch`, `segmentation-models-pytorch`) are
optional and are installed with `pip install -e ".[manet]"`; the rest of
Toontra does not require them. The checkpoint is distributed separately and
is not stored in Git or included in package data. `ManetBubbleMasker` requires
an explicit checkpoint path, does not download weights, and does not select
`WhiteBubbleMasker` automatically. See
[custom_models.md](custom_models.md#optional-ma-net-masker) for the
preprocessing and construction details.

## Other optional adapters

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) is available as an optional
  OCR adapter. Its dependency and model cache stay outside the core
  installation, and downloads require explicit consent.
- `CallableBubbleDetector` and the other component protocols accept a
  private, public, Torch, Paddle, or ONNX implementation without leaking
  framework types into the pipeline.

## Stages with no public model

`src/toontra/interfaces.py` documents text detection, inpainting, font
matching, text reinsertion, and image enhancement. These stages existed in
the original pipeline but are not implemented in this public reconstruction.
An adapter implements the matching protocol and is passed to the pipeline in
the same way as a custom `BubbleDetector` or `Translator`.
