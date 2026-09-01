r"""Resampled / bootstrap evaluation manifests.

A resample is **never** a materialized MEDS root. Multiplicity -- which is the entire content of a
bootstrap -- cannot survive one, and each of the four reasons below was measured rather than assumed:

* ``meds.SubjectSplitSchema`` is **closed** (``allow_extra_columns = False``) and holds exactly
  ``subject_id`` and ``split``. There is nowhere to put a weight: a MEDS root can say *which* subjects
  are in a split, never *how many times* each one counts.
* Duplicate rows in a subject-schema frame corrupt MEDS-Torch-Data's ``end_event_index``: a subject with
  3 events listed twice was reported as having 6, and every offset after it was wrong.
* ``MEDS_transforms``' ``reshard_to_split`` runs the split's subjects through ``np.unique``, so repeated
  draws are discarded before a single row is written.
* Four independent consumers silently deduplicate, so a duplicated subject is not even an error --- it
  is a resample that quietly is not the resample that was asked for.

So the draws are emitted as a **manifest over the unmodified parent**: the repeated subject ids
themselves, plus -- when the parent's task/index labels are given -- a copy of those labels in which
each drawn subject's rows are repeated once per draw. Duplicated *label* rows, unlike duplicated
subjects, pass through MTD unharmed (verified end to end: ``len(dataset)`` equals the number of draws,
``subj_locations`` is untouched, and duplicate items yield byte-identical tensors). That is where
multiplicity lives.

The layout, under the ``out_root`` handed to :func:`build_resamples`::

    <out_root>/resamples/
      family.json                      # the config, the parent ids, and every replicate's ids
      <resample_name>/
        subjects.parquet               # subject_id (repeated for multiplicity), draw_index
        labels/<shard>.parquet         # only when an index directory was given
        manifest.json                  # this test set's ids and provenance

Examples:
    >>> from meds_subsetter.config import ResampleConfig
    >>> parent = simple_static_MEDS_dataset_with_task
    >>> index_dir = parent / "task_labels" / "boolean_value_task"
    >>> tmp = tempfile.TemporaryDirectory()
    >>> out = Path(tmp.name)
    >>> cfg = ResampleConfig(n_resamples=2, size=4, salt="boot")
    >>> family = build_resamples(parent, out, cfg, index_dir=index_dir)
    >>> print_directory(out)
    └── resamples
        ├── boot_000
        │   ├── labels
        │   │   ├── labels_A.parquet.parquet
        │   │   └── labels_B.parquet.parquet
        │   ├── manifest.json
        │   └── subjects.parquet
        ├── boot_001
        │   ├── labels
        │   │   ├── labels_A.parquet.parquet
        │   │   └── labels_B.parquet.parquet
        │   ├── manifest.json
        │   └── subjects.parquet
        └── family.json

    The parent is untouched, and every replicate names itself so an evaluation table can join on
    ``test_set_id``:

    >>> [m.test_set_id for m in family.resamples]
    ['boot_000', 'boot_001']
    >>> tmp.cleanup()
"""

from __future__ import annotations

import dataclasses
import shutil
from typing import TYPE_CHECKING, Any

import meds
import polars as pl

from .digests import digest, frame_digest, subject_set_digest
from .materialize import atomic_write_json, atomic_write_parquet
from .selection import draw_with_replacement, draw_without_replacement

# Not a second implementation of the same scan: every recursive scan in this package must skip
# dot-directories (MEDS-Transforms writes `.logs/` inside the directories it fills) and must refuse a
# dangling symlink rather than quietly report fewer shards. `sizes.meds_shard_paths` is this helper
# bound to `<root>/data`; a label directory is not a MEDS root, so the shared helper is used directly.
from .sizes import _parquet_shard_paths

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from .config import ResampleConfig

#: Directory, under ``out_root``, that holds a resample family.
RESAMPLES_SUBDIR = "resamples"

#: Family-level manifest, written beside the replicate directories.
FAMILY_FILENAME = "family.json"

#: Per-replicate manifest, written in the replicate's directory.
MANIFEST_FILENAME = "manifest.json"

#: Per-replicate draw list, written in the replicate's directory.
SUBJECTS_FILENAME = "subjects.parquet"

#: Subdirectory of a replicate that holds its label shards.
LABELS_SUBDIR = "labels"

#: Column of :data:`SUBJECTS_FILENAME` giving each draw its position in the draw order.
DRAW_INDEX_COLUMN = "draw_index"

#: The MEDS label columns, in schema order. Only those present in the input are ever written, and
#: ``meds.LabelSchema.align`` is never called --- real ACES output has all-null ``integer_value`` /
#: ``float_value`` / ``categorical_value`` columns, which the closed schema declares non-nullable.
LABEL_COLUMNS = tuple(meds.LabelSchema.columns())

_SUBJECT_ID = meds.SubjectSplitSchema.subject_id_name
_SPLIT = meds.SubjectSplitSchema.split_name
_ROW_COLUMN = "__row__"


def _source_split_subjects(parent: Path, split: str) -> list[int]:
    """Return the parent's subjects in ``split``, deduplicated and ascending.

    Args:
        parent: The parent MEDS root.
        split: The split label to read.

    Returns:
        The subject ids carrying ``split``, sorted ascending.

    Raises:
        FileNotFoundError: If the parent has no ``metadata/subject_splits.parquet``.
        ValueError: If no subject carries ``split``. An empty pool is always a mistake -- a typo in the
            split name, or a parent that never had one -- and it would otherwise surface much later as
            an error about drawing from an empty pool.

    Examples:
        >>> _source_split_subjects(simple_static_MEDS, "train")
        [68729, 239684, 814703, 1195293]
        >>> _source_split_subjects(simple_static_MEDS, "held_out")
        [1500733]

        Any string is a legal MEDS split label, so a split that nobody is in is reported with the
        labels that do exist rather than as an empty list:

        >>> _source_split_subjects(simple_static_MEDS, "test")
        Traceback (most recent call last):
            ...
        ValueError: No subject is in split 'test';
        ...subject_splits.parquet has ['held_out', 'train', 'tuning']

        >>> _source_split_subjects(Path("/nonexistent"), "train")
        Traceback (most recent call last):
            ...
        FileNotFoundError: /nonexistent/metadata/subject_splits.parquet does not exist
    """
    path = parent / meds.subject_splits_filepath
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    splits = pl.read_parquet(path, glob=False)
    subjects = splits.filter(pl.col(_SPLIT) == split)[_SUBJECT_ID].unique().sort().to_list()
    if not subjects:
        present = sorted(splits[_SPLIT].unique().to_list())
        raise ValueError(f"No subject is in split {split!r}; {path} has {present}")
    return subjects


def _eligible_pool(
    subjects: Sequence[int], exclude: Iterable[int] | None, allow_overlap: bool
) -> tuple[list[int], int, bool]:
    """Narrow a split's subjects to those a replicate may draw.

    Excluding is the default and the honest construction: nothing in the MEDS ecosystem enforces
    train/test disjointness, so an evaluation resample that quietly overlaps the training subset is an
    easy and invisible mistake.

    Args:
        subjects: The source split's subjects.
        exclude: Subjects to hold out -- normally the training subset's. ``None`` excludes nothing.
        allow_overlap: Whether to ignore ``exclude`` and draw from the whole split.

    Returns:
        The pool, how many subjects the exclusion removed from it, and whether exclusion was applied.

    Examples:
        >>> _eligible_pool([1, 2, 3], [2, 9], False)
        ([1, 3], 1, True)

        Only the subjects actually in the split are counted as excluded -- ``9`` above was never in the
        pool to begin with. With ``allow_overlap``, nothing is removed and the manifest says so:

        >>> _eligible_pool([1, 2, 3], [2, 9], True)
        ([1, 2, 3], 0, False)
        >>> _eligible_pool([1, 2, 3], None, False)
        ([1, 2, 3], 0, False)

        ``exclude`` is consumed exactly once, so a generator is fine:

        >>> _eligible_pool([1, 2, 3], iter([2, 3]), False)
        ([1], 2, True)
    """
    if exclude is None or allow_overlap:
        return list(subjects), 0, False
    held_out = set(exclude)
    pool = [subject_id for subject_id in subjects if subject_id not in held_out]
    return pool, len(subjects) - len(pool), True


def _subjects_frame(draws: Sequence[int]) -> pl.DataFrame:
    """Render the draws as the frame written to ``subjects.parquet``.

    Rows are in draw order, which ``draw_index`` makes a total order, so the file's bytes are a
    function of the draws alone.

    Args:
        draws: The drawn subject ids, in draw order, with repeats.

    Returns:
        A two-column frame of ``subject_id`` and :data:`DRAW_INDEX_COLUMN`.

    Examples:
        >>> _subjects_frame([7, 7, 3])
        shape: (3, 2)
        ┌────────────┬────────────┐
        │ subject_id ┆ draw_index │
        │ ---        ┆ ---        │
        │ i64        ┆ i64        │
        ╞════════════╪════════════╡
        │ 7          ┆ 0          │
        │ 7          ┆ 1          │
        │ 3          ┆ 2          │
        └────────────┴────────────┘

        The empty draw is still a well-typed frame:

        >>> _subjects_frame([]).schema
        Schema({'subject_id': Int64, 'draw_index': Int64})
    """
    return pl.DataFrame(
        {_SUBJECT_ID: list(draws), DRAW_INDEX_COLUMN: list(range(len(draws)))},
        schema={_SUBJECT_ID: pl.Int64, DRAW_INDEX_COLUMN: pl.Int64},
    )


def _repeat_labels(labels: pl.DataFrame, draws: Sequence[int]) -> pl.DataFrame:
    """Repeat each drawn subject's label rows once per draw, in draw order.

    This is where a bootstrap's multiplicity is expressed: a subject drawn three times contributes
    three copies of every label row it has. The output is the concatenation, in draw order, of each
    draw's rows, so it lines up one block per row of ``subjects.parquet``.

    Only :data:`LABEL_COLUMNS` that the input actually carries are kept. ``meds.LabelSchema.align`` is
    deliberately not called: it is a closed schema whose four value columns are declared non-nullable,
    while real label files have three of them all-null.

    Args:
        labels: One shard of the parent's labels.
        draws: The drawn subject ids, in draw order, with repeats.

    Returns:
        The repeated rows, with the input's label columns in :data:`LABEL_COLUMNS` order.

    Raises:
        ValueError: If ``labels`` has no ``subject_id`` column.

    Examples:
        >>> labels = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 2],
        ...         "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...         "boolean_value": [True, False, True],
        ...     }
        ... )
        >>> _repeat_labels(labels, [1, 2, 1])
        shape: (5, 3)
        ┌────────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ prediction_time     ┆ boolean_value │
        │ ---        ┆ ---                 ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ bool          │
        ╞════════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ true          │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ false         │
        │ 2          ┆ 2021-01-03 00:00:00 ┆ true          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ true          │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ false         │
        └────────────┴─────────────────────┴───────────────┘

        A subject drawn three times contributes three copies of each of its rows:

        >>> _repeat_labels(labels, [1, 1, 1]).equals(pl.concat([labels.head(2)] * 3))
        True

        A subject that was not drawn contributes nothing, and a shard none of whose subjects were drawn
        yields an empty frame rather than no frame:

        >>> _repeat_labels(labels, [2]).height, _repeat_labels(labels, [99]).height
        (1, 0)

        Columns outside the MEDS label schema are dropped, and the survivors are put in schema order:

        >>> wide = labels.select("boolean_value", "subject_id", pl.lit("x").alias("note"))
        >>> _repeat_labels(wide, [2]).columns
        ['subject_id', 'boolean_value']

        >>> _repeat_labels(labels.drop("subject_id"), [1])
        Traceback (most recent call last):
            ...
        ValueError: Label shard has no 'subject_id' column; got ['prediction_time', 'boolean_value']
    """
    if _SUBJECT_ID not in labels.columns:
        raise ValueError(f"Label shard has no {_SUBJECT_ID!r} column; got {labels.columns}")
    columns = [name for name in LABEL_COLUMNS if name in labels.columns]
    return (
        labels.select(columns)
        .with_row_index(_ROW_COLUMN)
        .join(_subjects_frame(draws), on=_SUBJECT_ID, how="inner")
        .sort(DRAW_INDEX_COLUMN, _ROW_COLUMN)
        .select(columns)
    )


def _write_labels(
    label_paths: Sequence[Path], index_dir: Path, draws: Sequence[int], out_dir: Path
) -> list[Path]:
    """Write one repeated label shard per input label shard.

    Each output keeps its input's path relative to ``index_dir``, so the output is sharded exactly the
    way the input was -- including the doubled ``.parquet.parquet`` that MEDS task label files really
    carry. A shard none of whose subjects were drawn is written as an empty frame rather than skipped,
    so the output sharding mirrors the input's unconditionally and a missing file always means a
    missing input.

    Args:
        label_paths: The input label shards.
        index_dir: The directory ``label_paths`` are relative to.
        draws: The drawn subject ids, in draw order, with repeats.
        out_dir: The replicate's ``labels/`` directory.

    Returns:
        The written shards, in the order ``label_paths`` arrived in.

    Examples:
        >>> index_dir = simple_static_MEDS_dataset_with_task / "task_labels" / "boolean_value_task"
        >>> shards = sorted(index_dir.glob("*.parquet"))
        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "labels"
        >>> [p.name for p in _write_labels(shards, index_dir, [239684, 239684], out)]
        ['labels_A.parquet.parquet', 'labels_B.parquet.parquet']
        >>> print_directory(out)
        ├── labels_A.parquet.parquet
        └── labels_B.parquet.parquet
        >>> pl.read_parquet(out / "labels_A.parquet.parquet").height
        6
        >>> pl.read_parquet(out / "labels_B.parquet.parquet").height
        0
        >>> tmp.cleanup()
    """
    written = []
    for src in label_paths:
        dst = out_dir / src.relative_to(index_dir)
        atomic_write_parquet(_repeat_labels(pl.read_parquet(src, glob=False), draws), dst)
        written.append(dst)
    return written


def _index_id(paths: Sequence[Path]) -> str:
    """Digest a replicate's written label shards as one id.

    The shards are scanned together, at one pinned column set, so the id is a function of the label
    *rows* and not of how they were split into files. :func:`~meds_subsetter.digests.frame_digest`
    preserves multiplicity, so two replicates that drew the same subjects different numbers of times
    get different ids.

    Args:
        paths: The written label shards.

    Returns:
        A ``"sha256:<hex>"`` string.

    Examples:
        >>> index_dir = simple_static_MEDS_dataset_with_task / "task_labels" / "boolean_value_task"
        >>> shards = sorted(index_dir.glob("*.parquet"))
        >>> from meds_subsetter.digests import short_id
        >>> short_id(_index_id(shards))
        'c79f91f0bcb16f46'

        Sharding is invisible to it, multiplicity is not:

        >>> short_id(_index_id(shards[::-1]))
        'c79f91f0bcb16f46'
        >>> short_id(_index_id([*shards, shards[0]]))
        'ccac977efad8344f'
    """
    lf = pl.scan_parquet(list(paths), glob=False, missing_columns="insert", extra_columns="ignore")
    return frame_digest(lf, schema=lf.collect_schema())


def _prune_stale(family_dir: Path, keep: Sequence[str]) -> list[Path]:
    """Remove what an earlier build of this family left behind and this one will not overwrite.

    ``subjects.parquet``, ``manifest.json`` and ``family.json`` are each rewritten unconditionally, so
    they cannot go stale. Two things can. A replicate's ``labels/`` takes its file names from the
    *input's* sharding, which is free to change between runs -- re-running the upstream task pipeline
    and getting one shard where there were two is enough -- and a shard left over from the previous
    sharding is then counted a second time by anything that reads ``labels/`` as a directory. That
    multiplies a drawn subject's rows, which breaks the one invariant a bootstrap has, and no id
    detects it: :func:`_index_id` digests the shards *this* run wrote. Likewise a replicate directory
    that a smaller ``n_resamples`` no longer names is a test set that ``family.json`` does not list.

    Only directories this package wrote are removed: a replicate directory is recognized by its
    :data:`MANIFEST_FILENAME`, so anything else under ``family_dir`` is left alone, as is any symlink.

    Args:
        family_dir: The family directory, ``<out_root>/resamples``. Need not exist.
        keep: The replicate names this build is about to write.

    Returns:
        The paths removed: each kept replicate's stale ``labels/`` in ``keep`` order, then the
        replicate directories this build does not name, sorted.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> family_dir = Path(tmp.name) / "resamples"
        >>> for name in ("boot_000", "boot_001"):
        ...     (family_dir / name / "labels").mkdir(parents=True)
        ...     (family_dir / name / "labels" / "old.parquet").touch()
        ...     (family_dir / name / "manifest.json").touch()
        >>> (family_dir / "notes.txt").touch()
        >>> (family_dir / "scratch").mkdir()
        >>> [p.relative_to(family_dir).as_posix() for p in _prune_stale(family_dir, ["boot_000"])]
        ['boot_000/labels', 'boot_001']

        The replicate the build still names keeps everything the build rewrites, and a directory this
        package did not write -- no ``manifest.json`` in it -- is not this function's to delete:

        >>> print_directory(family_dir)
        ├── boot_000
        │   └── manifest.json
        ├── notes.txt
        └── scratch

        Nothing stale is nothing removed, and a family that has never been built is not an error:

        >>> _prune_stale(family_dir, ["boot_000"])
        []
        >>> _prune_stale(Path(tmp.name) / "never_built", ["boot_000"])
        []
        >>> tmp.cleanup()
    """
    removed: list[Path] = []
    for name in keep:
        labels = family_dir / name / LABELS_SUBDIR
        if labels.is_dir() and not labels.is_symlink():
            shutil.rmtree(labels)
            removed.append(labels)

    if not family_dir.is_dir():
        return removed
    named = set(keep)
    for entry in sorted(family_dir.iterdir()):
        if entry.name in named or entry.is_symlink() or not entry.is_dir():
            continue
        if (entry / MANIFEST_FILENAME).is_file():
            shutil.rmtree(entry)
            removed.append(entry)
    return removed


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ResampleManifest:
    """One replicate: what was drawn, from what, and the ids that identify it.

    Attributes:
        test_set_id: The replicate's name, and the key an evaluation table joins on.
        replicate: The zero-based replicate index. With the config's ``salt`` this *is* the seed: the
            two of them reproduce the draw exactly, with no state carried between replicates.
        config: The family config, verbatim.
        parent: The parent MEDS root the draws came from.
        parent_subject_set_id: Digest of every subject in the source split, before exclusion. A parent
            whose split membership has changed is then detectable rather than silently re-drawn.
        pool_subject_set_id: Digest of the eligible pool, after exclusion.
        pool_size: How many subjects were eligible.
        n_excluded: How many subjects the exclusion removed from the pool.
        excluded_training_subset: Whether the exclusion was applied at all.
        n_draws: How many draws were taken. Equals ``config.size``.
        n_distinct: How many distinct subjects those draws cover. Below ``n_draws`` exactly when a
            subject was drawn more than once.
        subject_multiset_id: Digest of the draws **with** multiplicity, so two different multisets of
            the same subjects differ.
        subject_set_id: Digest of the distinct drawn subjects, ignoring multiplicity. Together with
            ``subject_multiset_id`` this separates "different subjects" from "same subjects, different
            weights".
        index_id: Digest of the replicate's written label rows, or ``None`` when no labels were given.
        task_name: The task the labels define, or ``None`` when no labels were given.

    Examples:
        >>> from meds_subsetter.config import ResampleConfig
        >>> manifest = ResampleManifest(
        ...     test_set_id="boot_000",
        ...     replicate=0,
        ...     config=ResampleConfig(n_resamples=1, size=3),
        ...     parent="/data/meds",
        ...     parent_subject_set_id=digest(["1", "2", "3"]),
        ...     pool_subject_set_id=digest(["2", "3"]),
        ...     pool_size=2,
        ...     n_excluded=1,
        ...     excluded_training_subset=True,
        ...     n_draws=3,
        ...     n_distinct=2,
        ...     subject_multiset_id=digest(["2", "3", "3"]),
        ...     subject_set_id=digest(["2", "3"]),
        ...     index_id=None,
        ...     task_name=None,
        ... )
        >>> manifest.test_set_id, manifest.n_draws, manifest.n_distinct
        ('boot_000', 3, 2)

        Frozen, like the configs it carries:

        >>> manifest.n_draws = 4
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'n_draws'
    """

    test_set_id: str
    replicate: int
    config: ResampleConfig
    parent: str
    parent_subject_set_id: str
    pool_subject_set_id: str
    pool_size: int
    n_excluded: int
    excluded_training_subset: bool
    n_draws: int
    n_distinct: int
    subject_multiset_id: str
    subject_set_id: str
    index_id: str | None
    task_name: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form written to ``manifest.json``.

        The config is embedded whole, and the two fields a downstream table reads most -- the salt and
        the source split -- are also lifted out flat beside it.

        Returns:
            A dict of JSON-native values.

        Examples:
            >>> from meds_subsetter.config import ResampleConfig
            >>> manifest = ResampleManifest(
            ...     test_set_id="boot_000",
            ...     replicate=0,
            ...     config=ResampleConfig(n_resamples=1, size=3, salt="s"),
            ...     parent="/data/meds",
            ...     parent_subject_set_id="sha256:aa",
            ...     pool_subject_set_id="sha256:bb",
            ...     pool_size=2,
            ...     n_excluded=1,
            ...     excluded_training_subset=True,
            ...     n_draws=3,
            ...     n_distinct=2,
            ...     subject_multiset_id="sha256:cc",
            ...     subject_set_id="sha256:dd",
            ...     index_id=None,
            ...     task_name=None,
            ... )
            >>> from pprint import pprint
            >>> pprint(manifest.to_dict())
            {'config': {'allow_overlap': False,
                        'n_resamples': 1,
                        'name_template': 'boot_{index:03d}',
                        'replace': True,
                        'salt': 's',
                        'size': 3,
                        'source_split': 'train'},
             'excluded_training_subset': True,
             'index_id': None,
             'n_distinct': 2,
             'n_draws': 3,
             'n_excluded': 1,
             'parent': '/data/meds',
             'parent_subject_set_id': 'sha256:aa',
             'pool_size': 2,
             'pool_subject_set_id': 'sha256:bb',
             'replicate': 0,
             'salt': 's',
             'source_split': 'train',
             'subject_multiset_id': 'sha256:cc',
             'subject_set_id': 'sha256:dd',
             'task_name': None,
             'test_set_id': 'boot_000'}

            Every value is JSON-native, so the dict round-trips:

            >>> json.loads(json.dumps(manifest.to_dict())) == manifest.to_dict()
            True
        """
        return {
            "test_set_id": self.test_set_id,
            "replicate": self.replicate,
            "config": self.config.to_dict(),
            "salt": self.config.salt,
            "source_split": self.config.source_split,
            "parent": self.parent,
            "parent_subject_set_id": self.parent_subject_set_id,
            "pool_subject_set_id": self.pool_subject_set_id,
            "pool_size": self.pool_size,
            "n_excluded": self.n_excluded,
            "excluded_training_subset": self.excluded_training_subset,
            "n_draws": self.n_draws,
            "n_distinct": self.n_distinct,
            "subject_multiset_id": self.subject_multiset_id,
            "subject_set_id": self.subject_set_id,
            "index_id": self.index_id,
            "task_name": self.task_name,
        }


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ResampleFamilyManifest:
    """A whole family of replicates: one pool, one config, and every replicate's ids.

    Attributes:
        config: The family config, verbatim.
        parent: The parent MEDS root the draws came from.
        parent_subject_set_id: Digest of every subject in the source split, before exclusion.
        pool_subject_set_id: Digest of the eligible pool, after exclusion.
        pool_size: How many subjects were eligible.
        n_excluded: How many subjects the exclusion removed from the pool.
        excluded_training_subset: Whether the exclusion was applied at all.
        task_name: The task the labels define, or ``None`` when no labels were given.
        resamples: One :class:`ResampleManifest` per replicate, in replicate order.

    Examples:
        >>> from meds_subsetter.config import ResampleConfig
        >>> tmp = tempfile.TemporaryDirectory()
        >>> family = build_resamples(
        ...     simple_static_MEDS, Path(tmp.name), ResampleConfig(n_resamples=2, size=3, salt="boot")
        ... )
        >>> family.pool_size, family.n_excluded, family.excluded_training_subset
        (4, 0, False)
        >>> [m.replicate for m in family.resamples]
        [0, 1]
        >>> tmp.cleanup()
    """

    config: ResampleConfig
    parent: str
    parent_subject_set_id: str
    pool_subject_set_id: str
    pool_size: int
    n_excluded: int
    excluded_training_subset: bool
    task_name: str | None
    resamples: tuple[ResampleManifest, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form written to ``family.json``.

        Returns:
            A dict of JSON-native values, with every replicate's manifest under ``"resamples"``.

        Examples:
            >>> from meds_subsetter.config import ResampleConfig
            >>> tmp = tempfile.TemporaryDirectory()
            >>> family = build_resamples(
            ...     simple_static_MEDS, Path(tmp.name), ResampleConfig(n_resamples=2, size=3, salt="boot")
            ... )
            >>> d = family.to_dict()
            >>> sorted(d)
            ['config', 'excluded_training_subset', 'n_excluded', 'parent', 'parent_subject_set_id',
             'pool_size', 'pool_subject_set_id', 'resamples', 'task_name']
            >>> [r["test_set_id"] for r in d["resamples"]]
            ['boot_000', 'boot_001']
            >>> json.loads(json.dumps(d)) == d
            True

            It is exactly what was written to disk:

            >>> json.loads((Path(tmp.name) / "resamples" / "family.json").read_text()) == d
            True
            >>> tmp.cleanup()
        """
        return {
            "config": self.config.to_dict(),
            "parent": self.parent,
            "parent_subject_set_id": self.parent_subject_set_id,
            "pool_subject_set_id": self.pool_subject_set_id,
            "pool_size": self.pool_size,
            "n_excluded": self.n_excluded,
            "excluded_training_subset": self.excluded_training_subset,
            "task_name": self.task_name,
            "resamples": [manifest.to_dict() for manifest in self.resamples],
        }


def build_resamples(
    parent: Path,
    out_root: Path,
    cfg: ResampleConfig,
    *,
    exclude_subjects: Iterable[int] | None = None,
    index_dir: Path | None = None,
    task_name: str | None = None,
) -> ResampleFamilyManifest:
    r"""Draw a family of resampled evaluation manifests from a parent split.

    Nothing about the parent is read except its ``metadata/subject_splits.parquet`` and, if given, its
    label shards; nothing about it is written. See the module docstring for why a resample is a manifest
    rather than a MEDS root.

    A build owns ``<out_root>/resamples/`` entirely: an earlier build's replicate directories and label
    shards that this one will not overwrite are removed first (see :func:`_prune_stale`), so what is on
    disk is always exactly the family ``family.json`` describes.

    Args:
        parent: The parent MEDS root.
        out_root: Where the family goes. Everything lands under ``<out_root>/resamples/``.
        cfg: The family config: how many replicates, how big, with or without replacement, and the salt
            that pins the draws.
        exclude_subjects: Subjects a replicate may not draw -- normally the training subset's, so the
            evaluation sets stay disjoint from what was trained on. Ignored when
            ``cfg.allow_overlap``. Consumed exactly once.
        index_dir: The directory holding the parent's task/index label shards. When given, each
            replicate gets its own ``labels/`` copy with rows repeated per draw; when omitted, only the
            drawn subject ids are written.
        task_name: The name of the task ``index_dir`` holds labels for, recorded in the manifests.
            Defaults to ``index_dir.name``.

    Returns:
        The family manifest, which is also written to ``<out_root>/resamples/family.json``.

    Raises:
        ValueError: If ``task_name`` is given without ``index_dir``; if the eligible pool is empty; or
            if the pool is smaller than ``cfg.size`` and ``cfg.replace`` is false.
        FileNotFoundError: If the parent has no ``metadata/subject_splits.parquet``, or if ``index_dir``
            holds no parquet shards.

    Examples:
        >>> from meds_subsetter.config import ResampleConfig
        >>> parent = simple_static_MEDS_dataset_with_task
        >>> index_dir = parent / "task_labels" / "boolean_value_task"
        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name)
        >>> cfg = ResampleConfig(n_resamples=3, size=4, salt="boot")
        >>> family = build_resamples(parent, out, cfg, index_dir=index_dir)

        The parent's ``train`` split has four subjects, and each replicate drew four of them with
        replacement, so replicates repeat subjects and cover fewer than four distinct ones:

        >>> family.pool_size
        4
        >>> [(m.test_set_id, m.n_draws, m.n_distinct) for m in family.resamples]
        [('boot_000', 4, 3), ('boot_001', 4, 3), ('boot_002', 4, 2)]

        Replicates really differ -- three distinct draws, three distinct ids:

        >>> draws = {
        ...     m.test_set_id: pl.read_parquet(
        ...         out / "resamples" / m.test_set_id / "subjects.parquet"
        ...     )["subject_id"].to_list()
        ...     for m in family.resamples
        ... }
        >>> draws["boot_000"]
        [814703, 68729, 68729, 1195293]
        >>> draws["boot_001"]
        [68729, 1195293, 1195293, 239684]
        >>> draws["boot_002"]
        [239684, 239684, 239684, 814703]
        >>> len({m.subject_multiset_id for m in family.resamples})
        3
        >>> len({m.index_id for m in family.resamples})
        3

        Replicate ``boot_002`` drew subject 239684 three times, so its labels carry three copies of
        every one of that subject's label rows, in draw order:

        >>> rep = out / "resamples" / "boot_002"
        >>> parent_rows = pl.read_parquet(index_dir / "labels_A.parquet.parquet").filter(
        ...     pl.col("subject_id") == 239684
        ... )
        >>> parent_rows.height
        3
        >>> drawn = pl.read_parquet(rep / "labels" / "labels_A.parquet.parquet")
        >>> drawn.height
        9
        >>> drawn.equals(pl.concat([parent_rows] * 3))
        True

        The fourth draw, 814703, lives in the other label shard, and the output is sharded exactly the
        way the input was -- doubled ``.parquet.parquet`` and all:

        >>> other = pl.read_parquet(rep / "labels" / "labels_B.parquet.parquet")
        >>> other.equals(
        ...     pl.read_parquet(index_dir / "labels_B.parquet.parquet").filter(
        ...         pl.col("subject_id") == 814703
        ...     )
        ... )
        True

        Multiplicity is in the ids, not just the files: the ``subject_set_id`` of a replicate ignores
        it, the ``subject_multiset_id`` does not:

        >>> boot_002 = family.resamples[2]
        >>> boot_002.subject_set_id == subject_set_digest([239684, 814703])
        True
        >>> boot_002.subject_multiset_id == digest(["239684", "239684", "239684", "814703"])
        True
        >>> boot_002.subject_set_id == boot_002.subject_multiset_id
        False

        Re-running is idempotent -- same manifest, byte-identical files:

        >>> before = (out / "resamples" / "family.json").read_bytes()
        >>> build_resamples(parent, out, cfg, index_dir=index_dir) == family
        True
        >>> (out / "resamples" / "family.json").read_bytes() == before
        True

        By default the draws exclude the training subset, and the manifest records both the pool and
        what was taken out of it:

        >>> held_out = build_resamples(
        ...     parent, out / "disjoint", cfg, index_dir=index_dir, exclude_subjects=[239684, 1195293]
        ... )
        >>> held_out.pool_size, held_out.n_excluded, held_out.excluded_training_subset
        (2, 2, True)
        >>> pl.read_parquet(
        ...     out / "disjoint" / "resamples" / "boot_000" / "subjects.parquet"
        ... )["subject_id"].to_list()
        [68729, 68729, 68729, 814703]

        ``allow_overlap`` puts them back, and says so:

        >>> overlapping = build_resamples(
        ...     parent,
        ...     out / "overlapping",
        ...     ResampleConfig(n_resamples=1, size=4, salt="boot", allow_overlap=True),
        ...     exclude_subjects=[239684, 1195293],
        ... )
        >>> overlapping.pool_size, overlapping.n_excluded, overlapping.excluded_training_subset
        (4, 0, False)

        Excluding everybody is refused rather than answered with an empty test set:

        >>> build_resamples(
        ...     parent, out / "empty", cfg, exclude_subjects=[68729, 239684, 814703, 1195293]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: The eligible pool is empty:
        every subject of the 'train' split of ... was excluded

        Without replacement, a replicate is a distinct subset of the pool, and no draw may exceed it:

        >>> distinct = build_resamples(
        ...     parent, out / "distinct", ResampleConfig(n_resamples=2, size=3, replace=False, salt="boot")
        ... )
        >>> [m.n_distinct for m in distinct.resamples]
        [3, 3]
        >>> build_resamples(
        ...     parent, out / "oops", ResampleConfig(n_resamples=1, size=5, replace=False)
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cannot draw 5 distinct subjects without replacement from a pool of 4
        ('train' split of ..., 0 excluded)

        Without an index directory, only the subject ids are written, and ``index_id`` is ``None``:

        >>> bare = build_resamples(parent, out / "bare", ResampleConfig(n_resamples=1, size=2))
        >>> print_directory(out / "bare")
        └── resamples
            ├── boot_000
            │   ├── manifest.json
            │   └── subjects.parquet
            └── family.json
        >>> bare.resamples[0].index_id is None, bare.task_name is None
        (True, True)

        A rebuild owns the whole family directory. Suppose the upstream task pipeline is re-run and
        emits the very same label rows as one shard rather than two:

        >>> one_shard = out / "one_shard"
        >>> one_shard.mkdir()
        >>> pl.concat(
        ...     [pl.read_parquet(p) for p in sorted(index_dir.glob("*.parquet"))]
        ... ).write_parquet(one_shard / "labels_A.parquet.parquet")
        >>> rebuilt = build_resamples(
        ...     parent, out, ResampleConfig(n_resamples=1, size=4, salt="boot"), index_dir=one_shard
        ... )

        The replicate the new config no longer names is gone rather than left beside a ``family.json``
        that does not list it, and ``labels/`` holds this run's shards and nothing else:

        >>> sorted(p.name for p in (out / "resamples").iterdir())
        ['boot_000', 'family.json']
        >>> sorted(p.name for p in (out / "resamples" / "boot_000" / "labels").iterdir())
        ['labels_A.parquet.parquet']

        That is what keeps the bootstrap's defining invariant true *on disk*: a subject drawn ``k``
        times contributes exactly ``k`` copies of each of its label rows, counted over the whole
        ``labels/`` directory rather than over the files this run happened to write. A surviving shard
        from the two-shard run would have silently doubled 814703:

        >>> drawn = pl.read_parquet(out / "resamples" / "boot_000" / "labels")
        >>> full = pl.read_parquet(one_shard / "labels_A.parquet.parquet")
        >>> ids = pl.read_parquet(
        ...     out / "resamples" / "boot_000" / "subjects.parquet"
        ... )["subject_id"].to_list()
        >>> drawn.height == sum(full.filter(pl.col("subject_id") == s).height for s in ids)
        True
        >>> rebuilt.resamples[0].index_id == _index_id(
        ...     _parquet_shard_paths(out / "resamples" / "boot_000" / "labels")
        ... )
        True

        Dropping the labels entirely drops the directory, so ``index_id: null`` never sits beside a
        previous run's label rows:

        >>> _ = build_resamples(parent, out, ResampleConfig(n_resamples=1, size=4, salt="boot"))
        >>> print_directory(out / "resamples" / "boot_000")
        ├── manifest.json
        └── subjects.parquet

        A ``task_name`` without labels to name would be silently dropped, so it is refused:

        >>> build_resamples(parent, out / "nope", cfg, task_name="boolean_value_task")
        Traceback (most recent call last):
            ...
        ValueError: task_name is meaningless without index_dir; got task_name='boolean_value_task'
        >>> build_resamples(parent, out / "nope", cfg, index_dir=parent / "no_such_task")
        Traceback (most recent call last):
            ...
        FileNotFoundError: No parquet label shards under ...
        >>> tmp.cleanup()
    """
    if index_dir is None and task_name is not None:
        raise ValueError(f"task_name is meaningless without index_dir; got task_name={task_name!r}")

    label_paths: list[Path] = []
    if index_dir is not None:
        label_paths = _parquet_shard_paths(index_dir)
        if not label_paths:
            raise FileNotFoundError(f"No parquet label shards under {index_dir}")
        if task_name is None:
            task_name = index_dir.name

    split_subjects = _source_split_subjects(parent, cfg.source_split)
    pool, n_excluded, excluded_training_subset = _eligible_pool(
        split_subjects, exclude_subjects, cfg.allow_overlap
    )
    if not pool:
        raise ValueError(
            f"The eligible pool is empty: every subject of the {cfg.source_split!r} split of {parent} "
            f"was excluded"
        )
    if not cfg.replace and cfg.size > len(pool):
        raise ValueError(
            f"Cannot draw {cfg.size} distinct subjects without replacement from a pool of {len(pool)} "
            f"({cfg.source_split!r} split of {parent}, {n_excluded} excluded)"
        )

    draw = draw_with_replacement if cfg.replace else draw_without_replacement
    family_dir = out_root / RESAMPLES_SUBDIR
    names = [cfg.resample_name(replicate) for replicate in range(cfg.n_resamples)]
    _prune_stale(family_dir, names)
    shared = {
        "config": cfg,
        "parent": str(parent),
        "parent_subject_set_id": subject_set_digest(split_subjects),
        "pool_subject_set_id": subject_set_digest(pool),
        "pool_size": len(pool),
        "n_excluded": n_excluded,
        "excluded_training_subset": excluded_training_subset,
        "task_name": task_name,
    }

    manifests = []
    for replicate in range(cfg.n_resamples):
        draws = draw(pool, cfg.size, cfg.salt, replicate)
        test_set_id = names[replicate]
        replicate_dir = family_dir / test_set_id
        atomic_write_parquet(_subjects_frame(draws), replicate_dir / SUBJECTS_FILENAME)

        index_id = None
        if label_paths:
            written = _write_labels(label_paths, index_dir, draws, replicate_dir / LABELS_SUBDIR)
            index_id = _index_id(written)

        manifest = ResampleManifest(
            test_set_id=test_set_id,
            replicate=replicate,
            n_draws=len(draws),
            n_distinct=len(set(draws)),
            # `digest` over the sorted decimal ids, not `subject_set_digest`, which deduplicates: the
            # multiplicity is the resample, so two draws of the same subjects in different proportions
            # must not share an id.
            subject_multiset_id=digest(str(subject_id) for subject_id in sorted(draws)),
            subject_set_id=subject_set_digest(draws),
            index_id=index_id,
            **shared,
        )
        atomic_write_json(manifest.to_dict(), replicate_dir / MANIFEST_FILENAME)
        manifests.append(manifest)

    family = ResampleFamilyManifest(resamples=tuple(manifests), **shared)
    atomic_write_json(family.to_dict(), family_dir / FAMILY_FILENAME)
    return family
