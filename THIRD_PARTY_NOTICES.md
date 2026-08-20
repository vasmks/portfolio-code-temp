# Third-party notices

Toontra does not redistribute training datasets. It bundles one trained
checkpoint: `src/toontra/weights/speech_bubble_yolo26s.pt`.

## Required dependencies

### NumPy

- Project: <https://numpy.org/>
- Source: <https://github.com/numpy/numpy>
- License: BSD-3-Clause

### OpenCV

- Project: <https://opencv.org/>
- Source: <https://github.com/opencv/opencv>
- License: Apache-2.0 for OpenCV 4.5 and later

### Ultralytics and the bundled YOLO26 checkpoint

- Source: <https://github.com/ultralytics/ultralytics>
- License: Ultralytics publishes the project under AGPL-3.0-only and offers a
  commercial license. `Yolo26BubbleDetector` is Toontra's default detector,
  so Ultralytics is a base dependency.
- Bundled checkpoint: `src/toontra/weights/speech_bubble_yolo26s.pt` was
  trained for this public reconstruction, initialized from Ultralytics' official
  pretrained `yolo26s.pt`, and fine-tuned on the public Roboflow
  "speech-bubbles-detection" dataset (manga pages, several bubble-shape
  classes merged into Toontra's single `speech_bubble` label). See
  [docs/model_choices.md](docs/model_choices.md) for training details.

## Optional adapters

### EasyOCR

- Source: <https://github.com/JaidedAI/EasyOCR>
- License: Apache-2.0
- Behavior: EasyOCR may download selected language weights, but Toontra
  permits that only after the user explicitly opts in. No download occurs
  during import or automated tests.

### PyTorch, Segmentation Models PyTorch, and the optional MA-Net checkpoint

- PyTorch source: <https://github.com/pytorch/pytorch>
- PyTorch license: BSD-3-Clause; binary distributions may include components
  covered by additional notices.
- Segmentation Models PyTorch source:
  <https://github.com/qubvel-org/segmentation_models.pytorch>
- Segmentation Models PyTorch license: MIT
- Behavior: both packages are installed only through the optional `manet`
  extra (`pip install -e ".[manet]"`) and are never required by the base
  install. `ManetBubbleMasker` performs no model download; the checkpoint
  path is always supplied explicitly.
- Optional checkpoint: `toontra_manet_resnet34_bubble_segmentation.pth` was
  trained for this public reconstruction (MA-Net, `resnet34` encoder) on the public
  Roboflow "manga-segment_v2" dataset, version 5. The checkpoint is
  distributed separately and is excluded from Git and package data. See
  [docs/model_choices.md](docs/model_choices.md) for evaluation numbers and
  [docs/training/](docs/training/) for the training record.

Applicable upstream dependency, model, and dataset terms should be reviewed
separately.

## The AI-generated example image

`examples/source/webtoon_sample_original.png` and the metadata-stripped
`examples/input/sample_webtoon.png` derived from it were generated with
ChatGPT as synthetic demonstration artwork. They were not produced by, or
used to train or evaluate, anything in this repository. See
[README.md](README.md) for the full note.

## User-provided components

Custom models are supplied by the user through Toontra's component
interfaces. Toontra does not verify their source, license, training-data
provenance, or file integrity. Pickled or otherwise executable checkpoints
can run code when loaded and should come from a trusted source.
