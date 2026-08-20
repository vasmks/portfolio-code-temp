# Custom model contract

Toontra keeps model-specific tensor work inside adapters. The public pipeline
uses RGB NumPy images and a small set of data classes.

## Image input

- Type: `numpy.ndarray`
- Shape: `[height, width, 3]`
- Dtype: `uint8`
- Channel order: RGB

## Bubble detector output

`BubbleDetector.detect(image)` returns a sequence of `Detection` objects.

- Coordinates use `Box(x1, y1, x2, y2)`.
- Coordinates are pixel values relative to the supplied image.
- `x2` and `y2` are exclusive, matching NumPy slicing.
- `score` must be between 0 and 1.
- Boxes must have positive width and height.

The pipeline clips detections at page boundaries and rejects invalid output with
`ModelContractError`. A custom adapter should still perform its own model resize,
letterboxing, and inverse coordinate transform.

## Bubble masker output

`BubbleMasker.create_mask(bubble_crop)` returns a two-dimensional `uint8` mask
with exactly the same height and width as the crop it receives. The crop is
the reported `Detection.box` region, unmodified: after ownership and NMS,
Toontra expands a box exactly once in `detect_bubbles()` (see
`DETECTION_EXPANSION_RATIO` in `pipeline.py`). That
same expanded box is what gets reported, cropped for OCR, and passed to the
masker -- a masker never receives a tighter or looser crop than what
`Detection.box` records, and must not expand it again.

- `0` preserves the source pixel.
- `255` fully replaces it with the configured fill color.
- Values between 0 and 255 are supported as partial coverage.

Local masks are combined into the page mask using a pixel-wise maximum. The
completed mask is then applied inside detected regions, so overlaps cannot
restore pixels covered by another bubble.

## OCR and translation

`TextRecognizer.recognize(crop, language=...)` returns one `Recognition`.
`Translator.translate(texts, source_language=..., target_language=...)` must
return the same number of strings in the same order.

See `examples/custom_components.py` for small working adapters.

## Replacing the bundled YOLO26 detector

`Yolo26BubbleDetector` is used by default. A replacement implements the same
`BubbleDetector` protocol
(`detect(image) -> Sequence[Detection]`) and is passed to the constructor:

```python
from toontra import Toontra

toontra = Toontra(detector=MyBubbleDetector())
```

The rest of the pipeline -- tiling, ownership, cross-tile NMS, box
expansion, masking, OCR, translation -- only ever sees the `Detection`
objects your detector's `detect()` returns; it never depends on YOLO26 or
Ultralytics types. See `docs/model_choices.md` for the bundled checkpoint's
provenance.

## Stages with no bundled model

These interfaces document stages that existed in the original pipeline but
are not implemented in this public reconstruction: `TextDetector`,
`Inpainter`, `FontMatcher`, `TextReinserter`, and `ImageEnhancer`. `Toontra`
does not accept them as constructor arguments; they are technical contracts,
not executable pipeline stages.

## Optional MA-Net masker

`ManetBubbleMasker` is a ready-made adapter for the optional MA-Net/ResNet34
speech-bubble checkpoint (Segmentation Models PyTorch `MAnet`, `resnet34`
encoder, one output class). The checkpoint is distributed separately and is
not stored in Git or included in package data. Install the optional runtime:

```console
pip install -e ".[manet]"
```

Pass the checkpoint path explicitly -- there is no default and the adapter
never downloads weights:

```python
from toontra import Toontra
from toontra.modules import ManetBubbleMasker

masker = ManetBubbleMasker(
    "weights/toontra_manet_resnet34_bubble_segmentation.pth",
    device="auto",
)
toontra = Toontra(masker=masker)
```

The adapter accepts one checkpoint format. Construction:
`torch.load(..., weights_only=True)`, verify
the embedded metadata (`architecture="MAnet"`, `encoder="resnet34"`,
`in_channels=3`, `classes=1`, `image_size=512`) matches what the code
constructs, then a strict `load_state_dict` into
`smp.MAnet(encoder_name="resnet34", encoder_weights=None, in_channels=3,
classes=1, activation=None)`. A mismatched or malformed checkpoint raises
`ModelContractError` instead of being silently coerced.

Preprocessing reproduces the training pipeline: RGB input, resized to fit
within 512 x 512 with aspect ratio preserved, centered on a white 512 x 512
canvas (letterbox padding), ImageNet mean/std normalization. Inference runs
the model, applies sigmoid, thresholds at the checkpoint's own recorded
threshold (pass `threshold=` to override it), strips the letterbox padding,
and resizes the binary mask back to the original crop size with
nearest-neighbor interpolation -- so the returned mask lines up with the crop
pixel-for-pixel. See [model_choices.md](model_choices.md) for where that
checkpoint came from and its evaluation numbers.

The crop this adapter receives has already been expanded by the pipeline
(`DETECTION_EXPANSION_RATIO` in `pipeline.py`); `ManetBubbleMasker` does not
expand it again.

To run the opt-in local integration test without putting a weight path in the
repository, set `TOONTRA_MANET_CHECKPOINT` (optionally `TOONTRA_MANET_DEVICE`).

Input files are decoded as 8-bit RGB. PNG transparency is flattened onto white
at the file boundary. A model adapter never needs to interpret BGR or alpha
channels unless it chooses to use them internally.

## Model metadata checklist

Before sharing an adapter, record:

- model name and version or immutable revision;
- source and model-card URLs;
- code and weight licenses;
- training dataset and its usage rights;
- expected input normalization and output schema;
- SHA-256 checksum for each downloaded artifact;
- tested hardware, runtime, and known failure modes.

Toontra never downloads a model at import time. Keep weights outside Git.
