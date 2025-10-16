"""Utility helpers for persisting Matplotlib figures alongside WandB logging."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from matplotlib.figure import Figure

__all__ = ["save_figure"]

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_key(key: str) -> str:
    safe = _SANITIZE_PATTERN.sub("_", key).strip("_")
    return safe or "figure"


def save_figure(
    fig: Figure,
    *,
    output_dir: Optional[str],
    key: str,
    step: int,
    suffix: str = "pdf",
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
) -> Optional[Path]:
    """
    Save a Matplotlib figure to ``output_dir`` using a sanitized WandB key.

    Args:
        fig: Matplotlib figure to persist.
        output_dir: Directory where the file should be written. When ``None`` or
            empty, the figure is not saved and ``None`` is returned.
        key: WandB logging key (used to derive the filename).
        step: Global step used to disambiguate the filename.
        suffix: File suffix to use (default: ``\"pdf\"``).
        dpi: Optional DPI override passed to ``Figure.savefig``.
        bbox_inches: ``bbox_inches`` argument for ``Figure.savefig``.

    Returns:
        ``Path`` of the saved figure when ``output_dir`` is provided, otherwise ``None``.
    """
    if fig is None or not output_dir:
        return None

    safe_key = _sanitize_key(key)
    ext = suffix.lstrip(".")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"{safe_key}_step_{step:06d}.{ext}"
    fig.savefig(filename, format=ext, dpi=dpi, bbox_inches=bbox_inches)
    return filename

