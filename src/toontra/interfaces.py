"""Stage contracts for the rest of the original Toontra pipeline.

`contracts.py` defines the stages this repository actually implements and
wires into `Toontra`. The stages below existed in the original, company-owned
pipeline too, but no public model backs them here, so they are contracts
only -- nothing in `pipeline.py` calls them. A future adapter that trains or
licenses a real model for one of these stages implements the matching
protocol and is passed to it directly, the same way a custom `BubbleDetector`
or `Translator` is today.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import Box, RGBImage


@runtime_checkable
class TextDetector(Protocol):
    def detect(self, image: RGBImage) -> Sequence[Box]:
        """Return text-region boxes for the supplied image."""


@runtime_checkable
class Inpainter(Protocol):
    def inpaint(self, image: RGBImage, mask: RGBImage) -> RGBImage:
        """Return the image with masked text removed, background reconstructed."""


@runtime_checkable
class FontMatcher(Protocol):
    def match(self, text_region: RGBImage) -> str:
        """Return the best-matching font identifier for a text region's appearance."""


@runtime_checkable
class TextReinserter(Protocol):
    def reinsert(
        self,
        image: RGBImage,
        texts: Sequence[str],
        boxes: Sequence[Box],
        fonts: Sequence[str],
    ) -> RGBImage:
        """Return the image with translated text typeset into the given boxes."""


@runtime_checkable
class ImageEnhancer(Protocol):
    def enhance(self, images: Sequence[RGBImage]) -> list[RGBImage]:
        """Return one enhanced image per input image, in order."""
