from __future__ import annotations

import json
from pathlib import Path

from .errors import OutputExistsError
from .imaging import save_gray, save_rgb, save_rgba
from .models import PageResult


def save_page_result(
    result: PageResult,
    output_dir: str | Path,
    *,
    save_crops: bool = False,
    force: bool = False,
) -> Path:
    """Save documented artifacts without deleting or modifying source files."""
    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()) and not force:
        raise OutputExistsError(
            f"Output directory is not empty: {destination}. "
            "Pass --force in the CLI or force=True in Python to overwrite files."
        )

    owned_files = [
        destination / "cleaned.png",
        destination / "mask.png",
        destination / "overlay.png",
        destination / "result.json",
    ]
    crops_dir = destination / "crops"
    if crops_dir.is_dir():
        owned_files.extend(path for path in crops_dir.glob("bubble_*.png") if path.is_file())
    _refuse_source_conflict(result, owned_files)

    if force:
        for path in owned_files:
            if path.is_file():
                path.unlink()

    destination.mkdir(parents=True, exist_ok=True)
    save_rgb(destination / "cleaned.png", result.cleaned)
    save_gray(destination / "mask.png", result.mask)
    save_rgba(destination / "overlay.png", result.overlay)
    metadata_path = destination / "result.json"
    metadata_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if save_crops:
        crops_dir = destination / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for bubble in result.bubbles:
            rows, columns = bubble.detection.box.as_slices()
            crop = result.original[rows, columns]
            save_rgb(crops_dir / f"bubble_{bubble.index:03d}.png", crop)

    return destination


def _refuse_source_conflict(result: PageResult, targets: list[Path]) -> None:
    if result.source is None:
        return
    source = result.source.expanduser().resolve()
    for target in targets:
        if source == target.expanduser().resolve():
            raise OutputExistsError(
                f"Refusing to overwrite the input image with an output artifact: {source}"
            )
