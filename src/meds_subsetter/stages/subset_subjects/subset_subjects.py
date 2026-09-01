r"""The ``subset_subjects`` MEDS-Transforms stage: subject subsetting inside a pipeline.

This stage exists so subject subsetting *composes*. It drops into an existing MEDS-Transforms pipeline
like any other stage and parallelizes over shards with ``--multirun worker="range(0,N)"`` exactly like
the rest of the ecosystem, so a cohort can be subsetted in the same run that extracts, filters, and
normalizes it.

It is deliberately simpler than the ``meds-subset`` CLI, and does strictly less:

- It is a **pure per-shard map**. Each input shard is filtered to the selected subjects and written back
  out under the same shard name.
- It does **not reshard**, and therefore does **not** produce the disk-sharing nested family that is the
  point of this package. Rank-bucketed sharding needs the *whole cohort's* rank order at write time to
  decide which subject lands in which output file; a per-shard map only ever sees one shard. Use the
  ``meds-subset`` CLI (:mod:`meds_subsetter.subset`) when you want a family of nested subsets whose full
  shards are byte-identical and shared on disk.

Selection itself is still the nested, salted rank order of :func:`meds_subsetter.selection.select_nested`,
so the subset this stage produces holds exactly the subjects ``meds-subset`` would have chosen for the
same ``n_subjects`` and ``salt`` -- only the sharding differs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import meds
import polars as pl
from MEDS_transforms.stages import Stage

from meds_subsetter.config import DEFAULT_SALT
from meds_subsetter.selection import select_nested

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

#: The MEDS data column that names a subject. Read off the schema rather than hardcoded.
SUBJECT_ID = meds.DataSchema.subject_id_name

#: The column of ``subject_splits.parquet`` that names a subject's split.
SPLIT = meds.SubjectSplitSchema.split_name

#: The basename of ``subject_splits.parquet``; ``meds.subject_splits_filepath`` is root-relative.
SUBJECT_SPLITS_FILE = Path(meds.subject_splits_filepath).name


def _parquet_paths(directory: Path) -> list[Path]:
    """List the parquet files under ``directory``, sorted, skipping dot-prefixed paths.

    MEDS-Transforms writes ``.logs/`` and ``.hydra/`` *inside* the data directory, so a bare
    ``rglob("*.parquet")`` -- or a ``scan_parquet`` pointed at the directory -- would read log shards as
    data. Every path component is checked, not just the file name, and the result is sorted so that
    filesystem iteration order can never reach the selection.

    A dangling symlink is an error rather than a shard to skip. ``meds-subset`` links its subset roots
    into a shared shard store, so an unfollowable link is the routine consequence of moving a subset
    without its store -- and silently dropping the shard would shrink the candidate pool, which silently
    changes *which* subjects the rank order selects.

    Args:
        directory: The directory to scan recursively. Need not exist.

    Returns:
        The parquet files under ``directory``, sorted.

    Raises:
        FileNotFoundError: If a non-hidden ``*.parquet`` entry is a symlink whose target is missing.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     data = Path(tmp) / "train"
        ...     (data / ".logs").mkdir(parents=True)
        ...     pl.DataFrame({"a": [1]}).write_parquet(data / "1.parquet")
        ...     pl.DataFrame({"a": [1]}).write_parquet(data / "0.parquet")
        ...     pl.DataFrame({"a": [1]}).write_parquet(data / ".logs" / "0.parquet")
        ...     [p.name for p in _parquet_paths(data)]
        ['0.parquet', '1.parquet']

        A directory that does not exist yields nothing rather than raising:

        >>> _parquet_paths(Path("/does/not/exist"))
        []

        A shard linked into a shard store that did not come with it is an error:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     data = Path(tmp) / "train"
        ...     data.mkdir()
        ...     (data / "0.parquet").symlink_to(Path(tmp) / "store" / "r0.parquet")
        ...     _parquet_paths(data)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Dangling shard symlink: ...train/0.parquet -> .../store/r0.parquet
    """
    candidates = sorted(
        p
        for p in directory.rglob("*.parquet")
        if not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )
    for path in candidates:
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(f"Dangling shard symlink: {path} -> {path.readlink()}")
    return [p for p in candidates if p.is_file()]


def resolve_train_subjects(
    *,
    data_input_dir: str | Path | None,
    metadata_input_dir: str | Path | None,
    train_split: str,
) -> list[int]:
    """Resolve the *whole cohort's* train subject ids from a stage's pipeline-level input directories.

    Ranking ``n_subjects`` requires the full candidate pool: a subject's rank is its position among
    *every* train subject, so a per-shard map that ranked only the subjects in its own shard would select
    a different set in every worker. MEDS-Transforms does expose the pipeline-level directories to a
    stage functor -- ``stage_cfg.data_input_dir`` and ``stage_cfg.metadata_input_dir`` are resolved by
    ``PipelineConfig`` before the functor is bound -- so the pool can be read honestly, once per worker,
    from the same files every worker sees.

    Two sources are tried, in order, and the first that yields a non-empty pool wins:

    1. ``<metadata_input_dir>/subject_splits.parquet``, the spec-sanctioned record of split membership.
       Cheap: one small file, no data scan.
    2. The shards under ``<data_input_dir>/<train_split>/``. The ``train/`` shard-name prefix is a
       convention rather than part of the MEDS spec, but it is the same convention MEDS-Transforms' own
       ``shard_iterator`` uses to honor ``train_only``, and it is the only source left once an earlier
       metadata stage has repointed ``metadata_input_dir`` away from the dataset's ``metadata/``.

    Args:
        data_input_dir: The stage's data input directory, or ``None`` if unset.
        metadata_input_dir: The stage's metadata input directory, or ``None`` if unset.
        train_split: The split whose subjects are subsetted.

    Returns:
        The train subject ids, sorted ascending and deduplicated.

    Raises:
        ValueError: If neither source yields a subject, with the two ways out spelled out.
        FileNotFoundError: If a train shard is a symlink whose target is missing, which would otherwise
            shrink the pool and silently change the selection.

    Examples:
        >>> resolve_train_subjects(
        ...     data_input_dir=simple_static_MEDS / "data",
        ...     metadata_input_dir=simple_static_MEDS / "metadata",
        ...     train_split="train",
        ... )
        [68729, 239684, 814703, 1195293]

        The non-train splits are exactly the ones left out:

        >>> resolve_train_subjects(
        ...     data_input_dir=simple_static_MEDS / "data",
        ...     metadata_input_dir=simple_static_MEDS / "metadata",
        ...     train_split="held_out",
        ... )
        [1500733]

        With no split file in reach -- the shape a stage placed after a metadata stage takes, since its
        ``metadata_input_dir`` then points at that stage's output -- the ``train/`` shard prefix answers
        identically:

        >>> resolve_train_subjects(
        ...     data_input_dir=simple_static_MEDS / "data",
        ...     metadata_input_dir=None,
        ...     train_split="train",
        ... )
        [68729, 239684, 814703, 1195293]

        And when neither source is in reach, the error says what to do instead:

        >>> resolve_train_subjects(data_input_dir=None, metadata_input_dir=None, train_split="train")
        Traceback (most recent call last):
            ...
        ValueError: Cannot resolve a non-empty 'train' subject pool for `n_subjects` to be ranked
        against: no subject_splits.parquet rows under metadata_input_dir=None and no shards under
        data_input_dir=None/train. Pass an explicit `subject_ids` list to this stage, or use the
        `meds-subset` CLI, which reads the whole dataset directly.

        A split name that matches nothing raises the same actionable error rather than handing back an
        empty pool and failing later with an opaque "pool of 0":

        >>> resolve_train_subjects(
        ...     data_input_dir=simple_static_MEDS / "data",
        ...     metadata_input_dir=simple_static_MEDS / "metadata",
        ...     train_split="training",
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cannot resolve a non-empty 'training' subject pool ...
    """
    if metadata_input_dir is not None:
        splits_fp = Path(metadata_input_dir) / SUBJECT_SPLITS_FILE
        if splits_fp.is_file():
            ids = (
                pl.scan_parquet(splits_fp, glob=False)
                .filter(pl.col(SPLIT) == train_split)
                .select(SUBJECT_ID)
                .collect()
                .get_column(SUBJECT_ID)
                .to_list()
            )
            if ids:
                return sorted(set(ids))

    if data_input_dir is not None:
        paths = _parquet_paths(Path(data_input_dir) / train_split)
        if paths:
            # An explicit list of already-resolved files, so `glob=False`: a shard whose name holds
            # `*`, `?` or `[` would otherwise be re-expanded as a pattern. MEDS shards legitimately
            # differ in width -- `numeric_value` is optional -- hence `missing_columns="insert"`.
            ids = (
                pl.scan_parquet(paths, glob=False, missing_columns="insert", extra_columns="ignore")
                .select(SUBJECT_ID)
                .unique()
                .collect()
                .get_column(SUBJECT_ID)
                .to_list()
            )
            if ids:
                return sorted(set(ids))

    raise ValueError(
        f"Cannot resolve a non-empty {train_split!r} subject pool for `n_subjects` to be ranked against: "
        f"no {SUBJECT_SPLITS_FILE} rows under metadata_input_dir={metadata_input_dir} and no shards "
        f"under data_input_dir={data_input_dir}/{train_split}. Pass an explicit `subject_ids` list to "
        f"this stage, or use the `meds-subset` CLI, which reads the whole dataset directly."
    )


def filter_to_subjects(df: pl.LazyFrame, subject_ids: Collection[int]) -> pl.LazyFrame:
    """Keep only the rows belonging to ``subject_ids``.

    Args:
        df: The shard to filter.
        subject_ids: The subjects to keep.

    Returns:
        ``df`` restricted to ``subject_ids``.

    Examples:
        >>> df = pl.DataFrame({"subject_id": [1, 1, 2, 3, 3], "code": ["a", "b", "c", "d", "e"]})
        >>> filter_to_subjects(df, [1, 3])
        shape: (4, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ code │
        │ ---        ┆ ---  │
        │ i64        ┆ str  │
        ╞════════════╪══════╡
        │ 1          ┆ a    │
        │ 1          ┆ b    │
        │ 3          ┆ d    │
        │ 3          ┆ e    │
        └────────────┴──────┘

        Subjects that are not in the shard cost nothing, and an empty selection empties the shard:

        >>> filter_to_subjects(df, [3, 99]).height
        2
        >>> filter_to_subjects(df, []).height
        0
    """
    return df.filter(pl.col(SUBJECT_ID).is_in(list(subject_ids)))


def filter_cohort_to_subjects(
    df: pl.LazyFrame, subject_ids: Collection[int], cohort: Collection[int]
) -> pl.LazyFrame:
    """Subset ``cohort`` down to ``subject_ids``, passing every other subject through untouched.

    This is the split policy of :mod:`meds_subsetter` (design decision D3) expressed as a single
    per-shard filter: only the train split is subsetted, and ``tuning`` / ``held_out`` -- and any subject
    the split assignment does not mention at all -- survive byte-constant, so a scaling sweep holds its
    evaluation sets fixed along the size axis. Because the map runs over every shard of the dataset, a
    strict filter would instead empty the non-train shards.

    Args:
        df: The shard to filter.
        subject_ids: The selected subjects, a subset of ``cohort``.
        cohort: The subjects eligible for subsetting -- i.e. the whole train split.

    Returns:
        ``df`` with the ``cohort`` rows restricted to ``subject_ids`` and all other rows kept.

    Examples:
        >>> df = pl.DataFrame({"subject_id": [1, 2, 3, 4], "code": ["a", "b", "c", "d"]})
        >>> filter_cohort_to_subjects(df, [1], cohort=[1, 2])
        shape: (3, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ code │
        │ ---        ┆ ---  │
        │ i64        ┆ str  │
        ╞════════════╪══════╡
        │ 1          ┆ a    │
        │ 3          ┆ c    │
        │ 4          ┆ d    │
        └────────────┴──────┘

        A shard holding no cohort member at all is returned unchanged -- that is the passthrough of a
        ``tuning`` or ``held_out`` shard:

        >>> filter_cohort_to_subjects(df, [1], cohort=[1, 2]).filter(pl.col("subject_id") > 2).height
        2
        >>> filter_cohort_to_subjects(df, [], cohort=[1, 2]).height
        2
    """
    keep = list(subject_ids)
    pool = list(cohort)
    return df.filter(pl.col(SUBJECT_ID).is_in(keep) | ~pl.col(SUBJECT_ID).is_in(pool))


@Stage.register
def subset_subjects(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Filter each shard to a nested, salted subject subset of the train split.

    A pure per-shard map: it selects subjects and drops rows, and never moves a subject between shards.
    Shard names, shard membership, and every non-train split are left exactly as they were found. See
    the module docstring for why resharding lives in the ``meds-subset`` CLI instead.

    Args:
        stage_cfg: The stage configuration. Recognized keys:

            - ``subject_ids``: an explicit list of subjects to keep. Takes precedence over
              ``n_subjects`` whenever it is given, and filters the shard to *exactly* those subjects --
              non-train subjects are not passed through, because an explicit list carries no split
              information for the stage to reason with. Include them in the list if you want them.
            - ``n_subjects``: how many train subjects to select, ranked by
              :func:`meds_subsetter.selection.select_nested`. Selections nest: ``n_subjects=100`` is a
              prefix of ``n_subjects=1000`` for the same pool and salt. Subjects outside the train split
              pass through untouched.
            - ``salt``: the family salt fixing the rank order. Defaults to ``meds_subsetter``.
            - ``train_split``: the split that is subsetted. Defaults to ``train``.

    Returns:
        The per-shard filter.

    Raises:
        ValueError: If neither ``n_subjects`` nor ``subject_ids`` is set, or if the train subject pool
            cannot be resolved (see :func:`resolve_train_subjects`).
        TypeError: If ``n_subjects`` is not an integer.

    Examples:
        >>> from omegaconf import DictConfig
        >>> df = pl.DataFrame({"subject_id": [1, 1, 2, 3], "code": ["a", "b", "c", "d"]})

        An explicit list filters to exactly those subjects:

        >>> stage_cfg = DictConfig({"subject_ids": [1, 3], "n_subjects": None})
        >>> subset_subjects(stage_cfg)(df)
        shape: (3, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ code │
        │ ---        ┆ ---  │
        │ i64        ┆ str  │
        ╞════════════╪══════╡
        │ 1          ┆ a    │
        │ 1          ┆ b    │
        │ 3          ┆ d    │
        └────────────┴──────┘

        ``subject_ids`` wins over ``n_subjects`` when both are given:

        >>> stage_cfg = DictConfig({"subject_ids": [2], "n_subjects": 3})
        >>> subset_subjects(stage_cfg)(df)["subject_id"].to_list()
        [2]

        With ``n_subjects``, the pool is read from the pipeline-level input directories that
        MEDS-Transforms resolves into ``stage_cfg``, so every worker ranks against the same whole cohort.
        Against the static fixture, whose train split holds four subjects:

        >>> stage_cfg = DictConfig({
        ...     "n_subjects": 2,
        ...     "subject_ids": None,
        ...     "salt": "meds_subsetter",
        ...     "train_split": "train",
        ...     "data_input_dir": str(simple_static_MEDS / "data"),
        ...     "metadata_input_dir": str(simple_static_MEDS / "metadata"),
        ... })
        >>> def survivors(shard, n=2):
        ...     lf = pl.scan_parquet(simple_static_MEDS / "data" / f"{shard}.parquet")
        ...     fn = subset_subjects(DictConfig({**stage_cfg, "n_subjects": n}))
        ...     return sorted(fn(lf).collect()["subject_id"].unique().to_list())

        One of ``train/0``'s two subjects and one of ``train/1``'s two subjects are selected:

        >>> survivors("train/0")
        [239684]
        >>> survivors("train/1")
        [814703]

        The non-train shards pass through untouched -- that is what keeps an evaluation set fixed across
        a scaling sweep, and it is why the map can safely run over every shard of the dataset:

        >>> survivors("tuning/0")
        [754281]
        >>> survivors("held_out/0")
        [1500733]

        Selections nest, so a larger sweep point is a superset of every smaller one and no worker's
        answer depends on which shard it happens to hold:

        >>> [survivors("train/1", n) for n in (1, 2, 3, 4)]
        [[], [814703], [68729, 814703], [68729, 814703]]

        Setting neither key is an error rather than a silent no-op:

        >>> subset_subjects(DictConfig({"n_subjects": None, "subject_ids": None}))
        Traceback (most recent call last):
            ...
        ValueError: subset_subjects needs either `n_subjects` or `subject_ids`; both are unset. Set one,
        e.g. `stage_configs.subset_subjects.n_subjects=100`.

        As is a non-integer ``n_subjects``:

        >>> subset_subjects(DictConfig({"n_subjects": "100"}))
        Traceback (most recent call last):
            ...
        TypeError: n_subjects must be an integer; got <class 'str'> 100

        Asking for more subjects than the train split holds is an error too, not a short answer:

        >>> subset_subjects(DictConfig({**stage_cfg, "n_subjects": 9}))
        Traceback (most recent call last):
            ...
        ValueError: Cannot select 9 subjects from a pool of 4
    """
    subject_ids = stage_cfg.get("subject_ids", None)
    if subject_ids is not None:
        listed = [int(s) for s in subject_ids]
        logger.info(f"Filtering to the {len(listed)} explicitly listed subjects.")

        def filter_to_listed(data: pl.LazyFrame) -> pl.LazyFrame:
            return filter_to_subjects(data, listed)

        return filter_to_listed

    n_subjects = stage_cfg.get("n_subjects", None)
    if n_subjects is None:
        raise ValueError(
            "subset_subjects needs either `n_subjects` or `subject_ids`; both are unset. Set one, e.g. "
            "`stage_configs.subset_subjects.n_subjects=100`."
        )
    if not isinstance(n_subjects, int):
        raise TypeError(f"n_subjects must be an integer; got {type(n_subjects)} {n_subjects}")

    train_split = stage_cfg.get("train_split", None) or meds.train_split
    salt = stage_cfg.get("salt", None) or DEFAULT_SALT

    cohort = resolve_train_subjects(
        data_input_dir=stage_cfg.get("data_input_dir", None),
        metadata_input_dir=stage_cfg.get("metadata_input_dir", None),
        train_split=train_split,
    )
    keep = select_nested(cohort, n_subjects, salt)
    logger.info(
        f"Selected {len(keep)} of the {len(cohort)} {train_split!r} subjects with salt {salt!r}; "
        f"subjects outside that split pass through untouched."
    )

    def filter_to_selection(data: pl.LazyFrame) -> pl.LazyFrame:
        return filter_cohort_to_subjects(data, keep, cohort)

    return filter_to_selection
