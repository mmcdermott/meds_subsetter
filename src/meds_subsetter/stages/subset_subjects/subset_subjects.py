"""Placeholder MEDS-Transforms stage; replaced during implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from MEDS_transforms.stages import Stage

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl
    from omegaconf import DictConfig


@Stage.register
def subset_subjects(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Placeholder."""

    def fn(data: pl.LazyFrame) -> pl.LazyFrame:
        return data

    return fn
