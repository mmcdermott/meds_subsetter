r"""Exact dataset size statistics, pre- and post-tokenization.

Two entry points, both exact and both cheap:

- :func:`size_meds` counts a raw MEDS root by streaming its shards.
- :func:`size_tensorized` counts a MEDS-Torch-Data (MTD) tensorized cohort by reading only the
  safetensors *headers* of its ``.nrt`` files plus the tokenization schema frames -- no tensor bytes are
  ever touched, so sizing a terabyte-scale cohort costs a few thousand ``read`` syscalls.

Both return a :class:`SizeReport`: one :class:`SplitSizes` row per split, a dataset-level total, a tidy
:meth:`SizeReport.to_frame` for the scaling-law parquet, and a JSON-ready :meth:`SizeReport.to_dict` for
the sidecar.

Counts reported here are **config-free corpus counts**. They are deliberately *not* a "number of tokens
the model will see": that depends on ``max_seq_len``, the batch mode, the static-inclusion mode, and the
subsequence-sampling strategy. :func:`project_trained_tokens` supplies that separately, as an explicitly
parameterized closed form.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import TYPE_CHECKING, Any

import meds
import polars as pl

from .digests import digest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

#: Split label used when a dataset carries no split assignment at all.
ALL_SPLIT = "__all__"

#: Split label used for subjects present in the data but absent from the split assignment.
UNKNOWN_SPLIT = "__unknown__"

#: Split label carried by :attr:`SizeReport.total`.
TOTAL_SPLIT = "__total__"

#: Where an MTD tensorized cohort keeps its per-shard subject schema frames.
TENSORIZED_SCHEMAS_SUBDIR = "tokenization/schemas"

#: Suffix of the nested-ragged-tensor files in an MTD tensorized cohort.
NRT_SUFFIX = ".nrt"

#: Valid ``batch_mode`` values for :func:`project_trained_tokens`.
BATCH_MODES = ("SM", "SEM")

#: Valid ``static_inclusion_mode`` values for :func:`project_trained_tokens`.
STATIC_INCLUSION_MODES = ("OMIT", "INCLUDE", "PREPEND")

_SUBJECT_ID = meds.DataSchema.subject_id_name
_TIME = meds.DataSchema.time_name
_CODE = meds.DataSchema.code_name
_SPLIT = meds.SubjectSplitSchema.split_name
_STATIC_CODE = "static_code"
_PATH_COL = "__path__"
_SPLIT_COL = "__split__"

#: The MEDS data columns this module reads, at their spec'd dtypes. Declaring them makes the scan a
#: function of the *data* rather than of which shard polars happened to infer a schema from: a MEDS
#: dataset may hold an empty or static-only shard whose `time` or `subject_id` is physically `Null`,
#: and letting the first file define the schema then makes a rename of a shard the difference between
#: a report and a `SchemaError`. Extra columns -- including a shard that carries the reserved
#: `__path__` / `__split__` names -- are dropped by the scan rather than colliding with ours.
_SCAN_SCHEMA: dict[str, pl.DataType] = {
    _SUBJECT_ID: pl.Int64,
    _TIME: pl.Datetime("us"),
    _CODE: pl.String,
}

#: Widenings tolerated while reading into :data:`_SCAN_SCHEMA`. MEDS pins `time` to microseconds and
#: `subject_id` to int64, but datasets written by other tooling carry millisecond or nanosecond
#: timestamps, narrower id columns, and dictionary-encoded codes; all of those are exactly
#: representable, so they are cast rather than refused.
with warnings.catch_warnings():  # `ScanCastOptions` is flagged unstable and warns on construction.
    warnings.simplefilter("ignore")
    _CAST_OPTIONS = pl.ScanCastOptions(
        integer_cast="upcast",
        datetime_cast=["downcast", "upcast"],
        categorical_to_string="allow",
    )


def _is_hidden(relative_path: Path) -> bool:
    """Return whether any component of ``relative_path`` is dot-prefixed.

    Args:
        relative_path: A path *relative to* the directory being scanned. Passing an absolute path would
            be wrong: an absolute path under, say, ``/home/.cache/x`` would be reported as hidden.

    Returns:
        ``True`` if any component starts with ``"."``.

    Examples:
        >>> _is_hidden(Path("train/0.parquet")), _is_hidden(Path(".logs/train/0.parquet"))
        (False, True)
        >>> _is_hidden(Path("train/.0.parquet"))
        True
    """
    return any(part.startswith(".") for part in relative_path.parts)


def _parquet_shard_paths(directory: Path) -> list[Path]:
    r"""List the non-hidden parquet files under ``directory``, sorted, refusing broken links.

    ``Path.is_file`` answers ``False`` for a dangling symlink exactly as it does for a directory named
    ``*.parquet``, so filtering on it alone would silently *shrink* a report when a root's links into a
    shard store cannot be followed. Subsets link into a shared store by default, which makes an
    unresolvable link a routine consequence of copying a subset without its store -- and a plausible,
    entirely wrong report the failure mode. A directory is skipped quietly; a broken link is an error.

    Args:
        directory: The directory to scan recursively. Need not exist.

    Returns:
        The parquet files under ``directory``, sorted, skipping any path with a dot-prefixed component.

    Raises:
        FileNotFoundError: If a non-hidden ``*.parquet`` entry is a symlink whose target is missing.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     data = Path(tmp) / "data"
        ...     (data / ".logs").mkdir(parents=True)
        ...     pl.DataFrame({"a": [1]}).write_parquet(data / "0.parquet")
        ...     pl.DataFrame({"a": [1]}).write_parquet(data / ".logs" / "0.parquet")
        ...     (data / "dir.parquet").mkdir()
        ...     [p.name for p in _parquet_shard_paths(data)]
        ['0.parquet']

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     data = Path(tmp) / "data"
        ...     data.mkdir()
        ...     (data / "0.parquet").symlink_to(Path(tmp) / "store" / "r0.parquet")
        ...     _parquet_shard_paths(data)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Dangling shard symlink: ...data/0.parquet -> .../store/r0.parquet
    """
    candidates = sorted(p for p in directory.rglob("*.parquet") if not _is_hidden(p.relative_to(directory)))
    for path in candidates:
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(f"Dangling shard symlink: {path} -> {path.readlink()}")
    return [p for p in candidates if p.is_file()]


def _shard_name(base: Path, path: Path) -> str:
    """Return ``path`` relative to ``base``, suffix stripped and POSIX-joined.

    Args:
        base: The directory shard names are relative to.
        path: The shard file.

    Returns:
        The shard name.

    Raises:
        ValueError: If ``path`` does not live under ``base``.

    Examples:
        >>> _shard_name(Path("/d"), Path("/d/deep/nest/1.parquet"))
        'deep/nest/1'
        >>> _shard_name(Path("/d"), Path("/e/1.parquet"))
        Traceback (most recent call last):
            ...
        ValueError: /e/1.parquet is not under /d
    """
    try:
        relative = path.relative_to(base)
    except ValueError as e:
        raise ValueError(f"{path} is not under {base}") from e
    return relative.with_suffix("").as_posix()


def _split_prefix(name: str) -> str:
    """Return the split a shard name implies: its first component, or ``"__all__"`` if it is flat.

    Args:
        name: A shard name, as returned by :func:`shard_name`.

    Returns:
        The leading path component, or :data:`ALL_SPLIT` for an unnested name.

    Examples:
        >>> _split_prefix("train/0"), _split_prefix("held_out/deep/nest/1"), _split_prefix("0")
        ('train', 'held_out', '__all__')
    """
    prefix, separator, _ = name.partition("/")
    return prefix if separator else ALL_SPLIT


def meds_shard_paths(root: Path) -> list[Path]:
    r"""List the data shard files of a MEDS root, in sorted order, skipping dotfiles.

    MEDS-Transforms writes ``.logs/``, ``.hydra/``, and ``.shards.json`` *inside* ``data/``, so a naive
    ``rglob("*.parquet")`` -- or a ``scan_parquet`` pointed at the directory -- picks up log shards and
    reports them as data. Every component of the path is checked, not just the file name.

    Args:
        root: The MEDS root (the directory *containing* ``data/``).

    Returns:
        The shard paths, sorted. Empty if ``<root>/data`` does not exist.

    Raises:
        FileNotFoundError: If a shard is a symlink whose target is missing. Subsets are linked into a
            shared shard store by default, so this is the shape a store that was not copied alongside
            its subsets takes; dropping the shard instead would silently report a smaller dataset.

    Examples:
        >>> files = {
        ...     "data/train/1.parquet": None,
        ...     "data/train/0.parquet": None,
        ...     "data/held_out/deep/nest/0.parquet": None,
        ...     "data/.logs/train/0.parquet": None,
        ...     "data/.shards.json": {"train/0": [1]},
        ...     "metadata/codes.parquet": None,
        ... }
        >>> with yaml_disk(files) as root:
        ...     [p.relative_to(root).as_posix() for p in meds_shard_paths(root)]
        ['data/held_out/deep/nest/0.parquet', 'data/train/0.parquet', 'data/train/1.parquet']

        The real static fixture has four shards:

        >>> [shard_name(simple_static_MEDS, p) for p in meds_shard_paths(simple_static_MEDS)]
        ['held_out/0', 'train/0', 'train/1', 'tuning/0']

        A root with no ``data/`` directory yields nothing rather than raising:

        >>> with yaml_disk({"metadata/codes.parquet": None}) as root:
        ...     meds_shard_paths(root)
        []

        A shard that is a broken symlink -- a subset root whose shard store did not come with it -- is
        an error, not a shard to skip:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     subset = Path(tmp) / "subset"
        ...     (subset / "data" / "train").mkdir(parents=True)
        ...     (subset / "data" / "train" / "0.parquet").symlink_to(Path(tmp) / "store" / "r0.parquet")
        ...     meds_shard_paths(subset)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Dangling shard symlink: ...train/0.parquet -> .../store/r0.parquet
    """
    return _parquet_shard_paths(root / meds.data_subdirectory)


def shard_name(root: Path, path: Path) -> str:
    """Return the MEDS shard name of ``path``: its path under ``<root>/data``, suffix stripped.

    Shard names may nest arbitrarily; the ``train/`` prefix is a convention, not part of the spec.

    Args:
        root: The MEDS root (the directory *containing* ``data/``).
        path: A shard file under ``<root>/data``.

    Returns:
        The POSIX-joined, suffix-stripped shard name.

    Raises:
        ValueError: If ``path`` does not live under ``<root>/data``.

    Examples:
        >>> shard_name(Path("/root"), Path("/root/data/train/0.parquet"))
        'train/0'
        >>> shard_name(Path("/root"), Path("/root/data/held_out/deep/nest/1.parquet"))
        'held_out/deep/nest/1'

        Only the final suffix is stripped, so the doubled ``.parquet.parquet`` that MEDS task label
        files carry keeps one:

        >>> shard_name(Path("/root"), Path("/root/data/0.parquet.parquet"))
        '0.parquet'

        >>> shard_name(Path("/root"), Path("/elsewhere/0.parquet"))
        Traceback (most recent call last):
            ...
        ValueError: /elsewhere/0.parquet is not under /root/data
    """
    return _shard_name(root / meds.data_subdirectory, path)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SplitSizes:
    r"""Exact size statistics for one split of one dataset.

    All counts are exact, not estimates, and every one of them is a property of the data alone -- none
    depends on a loader configuration.

    Attributes:
        split: The split label. ``"__all__"`` when the dataset carries no split assignment,
            ``"__unknown__"`` for subjects in the data but missing from the split assignment, and
            ``"__total__"`` on the row produced by :attr:`SizeReport.total`.
        n_subjects: Distinct ``subject_id`` values, ``|{subject_id}|``.
        n_events: Distinct *timed* ``(subject_id, time)`` pairs,
            ``|{(subject_id, time) : time is not null}|``. Rows carrying a null ``time`` (MEDS static
            measurements) contribute **no** event. This is the definition that matches the MTD tokenizer
            and the ``.nrt`` files on disk exactly, and unlike the alternative it is not a function of
            how many subjects happen to carry a static row.

            ``meds_summary_stats`` uses the other convention, in which each subject's static rows form
            one additional (null-timed) event. The two differ by exactly
            :attr:`n_subjects_with_static`, so that convention is recovered as
            ``n_events + n_subjects_with_static`` -- see the example below.
        n_measurements: Data rows, ``|rows|``. Always ``n_dynamic_measurements +
            n_static_measurements``.
        n_dynamic_measurements: Rows with a non-null ``time``.
        n_static_measurements: Rows with a null ``time``.
        n_subjects_with_static: Distinct subjects having at least one null-timed row.
        n_codes_observed: Distinct ``code`` values *observed in the data*, for a raw MEDS root. This is
            not the size of ``metadata/codes.parquet``, which the MEDS spec only requires to be a
            superset (the static test fixture ships 5 code metadata rows for 11 observed codes). A
            *tensorized* cohort is the one exception: its codes are vocabulary indices inside tensors
            that header-only sizing never opens, so there the field carries the global vocabulary --
            the distinct codes of ``metadata/codes.parquet`` -- and is the same on every split.
        observed_code_set_id: A ``"sha256:..."`` digest of those codes, sorted, from
            :func:`meds_subsetter.digests.digest`. Two splits share an id exactly when they observe
            the same set of codes, so a vocabulary can be compared -- across splits, subsets, and
            machines -- without shipping it.
        n_shards: Number of shard files contributing rows.
        n_bytes: On-disk size of those shard files, from ``stat().st_size``.

    Examples:
        >>> from meds_subsetter.digests import digest
        >>> sizes = SplitSizes(
        ...     split="train", n_subjects=4, n_events=20, n_measurements=44, n_dynamic_measurements=36,
        ...     n_static_measurements=8, n_subjects_with_static=4, n_codes_observed=2,
        ...     observed_code_set_id=digest(["ADMISSION", "DISCHARGE"]), n_shards=2, n_bytes=3238,
        ... )
        >>> sizes.n_dynamic_measurements + sizes.n_static_measurements == sizes.n_measurements
        True

        The ``meds_summary_stats`` event convention, recovered exactly:

        >>> sizes.n_events + sizes.n_subjects_with_static
        24

        >>> sizes.to_dict()
        {'split': 'train', 'n_subjects': 4, 'n_events': 20, 'n_measurements': 44,
         'n_dynamic_measurements': 36, 'n_static_measurements': 8, 'n_subjects_with_static': 4,
         'n_codes_observed': 2,
         'observed_code_set_id': 'sha256:d947ddc0a8063189a7ccd813c79456db80a8e947e5badd2a87493b5f69587b94',
         'n_shards': 2, 'n_bytes': 3238}

        Reports are frozen, so a consumer cannot quietly adjust a count it did not like:

        >>> sizes.n_subjects = 5
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'n_subjects'
    """

    split: str
    n_subjects: int
    n_events: int
    n_measurements: int
    n_dynamic_measurements: int
    n_static_measurements: int
    n_subjects_with_static: int
    n_codes_observed: int
    observed_code_set_id: str
    n_shards: int
    n_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        """Return the plain-dict form, in field order.

        Returns:
            A JSON-serializable dict of every field.
        """
        return dataclasses.asdict(self)


#: The additive fields of :class:`SplitSizes` -- those a dataset total is the plain sum of.
_ADDITIVE_FIELDS = (
    "n_subjects",
    "n_events",
    "n_measurements",
    "n_dynamic_measurements",
    "n_static_measurements",
    "n_subjects_with_static",
)

#: The :class:`SplitSizes` fields that are labels or ids rather than counts.
_STRING_FIELDS = ("split", "observed_code_set_id")

#: Column names and dtypes of :meth:`SizeReport.to_frame`, in order.
_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "kind": pl.String,
    **{f.name: (pl.String if f.name in _STRING_FIELDS else pl.Int64) for f in dataclasses.fields(SplitSizes)},
}


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SizeReport:
    r"""The sizes of every split of one dataset, plus its dataset-level totals.

    Six of the eleven :class:`SplitSizes` fields are additive across splits, because a subject belongs
    to exactly one split. Four are not, and are therefore carried on the report itself rather than
    recomputed by summing:

    - ``n_codes_observed`` / ``observed_code_set_id``: splits share codes, so the dataset value is the
      *union* of the per-split code sets -- generally smaller than their sum, and not a function of the
      per-split ids at all.
    - ``n_shards`` / ``n_bytes``: a shard is attributed to every split it holds rows for, so if a shard
      mixes splits (legal, and what flat-sharded datasets do) the per-split values sum to more than the
      dataset holds. The carried values count each file exactly once.

    Attributes:
        source: The dataset root, as a string.
        kind: ``"meds"`` for a raw MEDS root, ``"tensorized"`` for an MTD cohort.
        splits: The per-split sizes, sorted by split label.
        n_codes_observed: Distinct codes across the whole dataset (the union, not a sum).
        observed_code_set_id: The digest of that union, as on :class:`SplitSizes`.
        n_shards: Distinct shard files in the dataset, counted once each.
        n_bytes: On-disk size of those files, counted once each.

    Examples:
        >>> from meds_subsetter.digests import digest
        >>> train = SplitSizes(
        ...     split="train", n_subjects=4, n_events=20, n_measurements=44, n_dynamic_measurements=36,
        ...     n_static_measurements=8, n_subjects_with_static=4, n_codes_observed=11,
        ...     observed_code_set_id=digest([f"C{i}" for i in range(11)]), n_shards=2, n_bytes=3238,
        ... )
        >>> tuning = SplitSizes(
        ...     split="tuning", n_subjects=1, n_events=3, n_measurements=7, n_dynamic_measurements=5,
        ...     n_static_measurements=2, n_subjects_with_static=1, n_codes_observed=7,
        ...     observed_code_set_id=digest([f"C{i}" for i in range(7)]), n_shards=1, n_bytes=1473,
        ... )
        >>> report = SizeReport(
        ...     source="/data/cohort", kind="meds", splits=(train, tuning), n_codes_observed=11,
        ...     observed_code_set_id=train.observed_code_set_id, n_shards=3, n_bytes=4711,
        ... )

        The total sums the additive fields and carries the rest, so ``n_codes_observed`` is 11 (the
        union) rather than 18 (the sum), and the id is the union's id rather than a combination:

        >>> total = report.total
        >>> {k: v for k, v in total.to_dict().items() if k != "observed_code_set_id"}
        {'split': '__total__', 'n_subjects': 5, 'n_events': 23, 'n_measurements': 51,
         'n_dynamic_measurements': 41, 'n_static_measurements': 10, 'n_subjects_with_static': 5,
         'n_codes_observed': 11, 'n_shards': 3, 'n_bytes': 4711}
        >>> total.observed_code_set_id == train.observed_code_set_id != tuning.observed_code_set_id
        True

        :meth:`to_frame` is the tidy scaling-law surface -- one row per split, then the total:

        >>> report.to_frame().columns
        ['source', 'kind', 'split', 'n_subjects', 'n_events', 'n_measurements',
         'n_dynamic_measurements', 'n_static_measurements', 'n_subjects_with_static',
         'n_codes_observed', 'observed_code_set_id', 'n_shards', 'n_bytes']
        >>> report.to_frame().select("source", "kind", "split", "n_subjects", "n_events")
        shape: (3, 5)
        ┌──────────────┬──────┬───────────┬────────────┬──────────┐
        │ source       ┆ kind ┆ split     ┆ n_subjects ┆ n_events │
        │ ---          ┆ ---  ┆ ---       ┆ ---        ┆ ---      │
        │ str          ┆ str  ┆ str       ┆ i64        ┆ i64      │
        ╞══════════════╪══════╪═══════════╪════════════╪══════════╡
        │ /data/cohort ┆ meds ┆ train     ┆ 4          ┆ 20       │
        │ /data/cohort ┆ meds ┆ tuning    ┆ 1          ┆ 3        │
        │ /data/cohort ┆ meds ┆ __total__ ┆ 5          ┆ 23       │
        └──────────────┴──────┴───────────┴────────────┴──────────┘

        :meth:`to_dict` is the JSON sidecar, and includes the total so the file stands alone:

        >>> sorted(report.to_dict())
        ['kind', 'n_bytes', 'n_codes_observed', 'n_shards', 'observed_code_set_id', 'source', 'splits',
         'total']
        >>> print(json.dumps(report.to_dict()["total"]["n_subjects"]))
        5

        A report with no splits is well-defined: every additive total is zero.

        >>> SizeReport(source="/empty", kind="meds", splits=(), n_codes_observed=0,
        ...            observed_code_set_id=digest([]), n_shards=0, n_bytes=0).total.n_subjects
        0
    """

    source: str
    kind: str
    splits: tuple[SplitSizes, ...]
    n_codes_observed: int
    observed_code_set_id: str
    n_shards: int
    n_bytes: int

    @property
    def total(self) -> SplitSizes:
        """The dataset-level sizes, as a :class:`SplitSizes` labelled ``"__total__"``.

        Returns:
            The additive fields summed over :attr:`splits`; ``n_codes_observed``,
            ``observed_code_set_id``, ``n_shards``, and ``n_bytes`` taken from the report's own
            carried, deduplicated values.
        """
        return SplitSizes(
            split=TOTAL_SPLIT,
            n_codes_observed=self.n_codes_observed,
            observed_code_set_id=self.observed_code_set_id,
            n_shards=self.n_shards,
            n_bytes=self.n_bytes,
            **{f: sum(getattr(s, f) for s in self.splits) for f in _ADDITIVE_FIELDS},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of the report.

        Returns:
            A dict with the scalar fields, a ``"splits"`` list of per-split dicts, and a ``"total"``
            dict, so the sidecar is readable without recomputing anything.
        """
        return {
            "source": self.source,
            "kind": self.kind,
            "n_codes_observed": self.n_codes_observed,
            "observed_code_set_id": self.observed_code_set_id,
            "n_shards": self.n_shards,
            "n_bytes": self.n_bytes,
            "splits": [s.to_dict() for s in self.splits],
            "total": self.total.to_dict(),
        }

    def to_frame(self) -> pl.DataFrame:
        """Return the tidy frame: one row per split, then the total row.

        Column names are plain (``n_subjects``, not ``dataset_size__n_subjects``); renaming for a
        particular scaling-law consumer is a one-liner on the caller's side.

        Returns:
            A :class:`polars.DataFrame` with ``source``, ``kind``, ``split``, the nine count columns,
            and ``observed_code_set_id``.
        """
        rows = [{"source": self.source, "kind": self.kind, **s.to_dict()} for s in (*self.splits, self.total)]
        return pl.DataFrame(rows, schema=_FRAME_SCHEMA)


def _subject_split_frame(root: Path, splits: Mapping[int, str] | None) -> pl.DataFrame | None:
    """Resolve a ``subject_id -> split`` assignment for a MEDS root, if one exists.

    Args:
        root: The MEDS root.
        splits: An explicit mapping, which wins over anything on disk.

    Returns:
        A two-column frame (``subject_id``, ``__split__``), or ``None`` when neither an explicit
        mapping nor ``metadata/subject_splits.parquet`` is available -- in which case the caller falls
        back to shard-name prefixes.

    Examples:
        >>> _subject_split_frame(Path("/nonexistent"), {2: "b", 1: "a"})
        shape: (2, 2)
        ┌────────────┬───────────┐
        │ subject_id ┆ __split__ │
        │ ---        ┆ ---       │
        │ i64        ┆ str       │
        ╞════════════╪═══════════╡
        │ 2          ┆ b         │
        │ 1          ┆ a         │
        └────────────┴───────────┘

        >>> _subject_split_frame(simple_static_MEDS, None).head(2)
        shape: (2, 2)
        ┌────────────┬───────────┐
        │ subject_id ┆ __split__ │
        │ ---        ┆ ---       │
        │ i64        ┆ str       │
        ╞════════════╪═══════════╡
        │ 239684     ┆ train     │
        │ 1195293    ┆ train     │
        └────────────┴───────────┘

        >>> with yaml_disk({"data/0.parquet": None}) as root:
        ...     _subject_split_frame(root, None) is None
        True
    """
    if splits is not None:
        return pl.DataFrame(
            {_SUBJECT_ID: list(splits), _SPLIT_COL: list(splits.values())},
            schema={_SUBJECT_ID: pl.Int64, _SPLIT_COL: pl.String},
        )
    path = root / meds.subject_splits_filepath
    if not path.is_file():
        return None
    return pl.read_parquet(path, glob=False).select(
        pl.col(_SUBJECT_ID).cast(pl.Int64), pl.col(_SPLIT).alias(_SPLIT_COL)
    )


def size_meds(root: Path, *, splits: Mapping[int, str] | None = None) -> SizeReport:
    r"""Count a raw MEDS root exactly, in one streaming pass.

    Split membership is resolved in this order: an explicit ``splits`` mapping; else
    ``metadata/subject_splits.parquet`` if it exists; else the first component of each shard name (the
    ``train/0`` convention); else a single ``"__all__"`` split. Subjects present in the data but absent
    from the resolved mapping land in ``"__unknown__"`` rather than being silently dropped, so the
    totals always account for every row on disk.

    The scan is given an explicit, sorted, dotfile-filtered list of files (never the ``data/``
    directory, which would recurse into ``.logs/``, and never as a glob, which would re-expand a
    bracketed path), at the pinned :data:`_SCAN_SCHEMA` rather than whichever dtypes the
    first-sorting shard happens to carry. The per-split aggregation and the dataset-wide code union are
    collected together under the streaming engine so they share one pass over the shards. Peak memory
    is dominated by the exact ``n_events`` count, which holds one entry per distinct
    ``(subject_id, time)`` pair so that the count stays right even when a split spans many shards, plus
    one copy of each distinct code per split.

    Args:
        root: The MEDS root (the directory containing ``data/`` and ``metadata/``).
        splits: An optional explicit ``subject_id -> split`` mapping, which overrides
            ``metadata/subject_splits.parquet``.

    Returns:
        A :class:`SizeReport` with ``kind="meds"``, splits sorted by label.

    Raises:
        FileNotFoundError: If ``<root>/data`` holds no non-hidden parquet shards, or if one of them is
            a symlink whose target is missing (see :func:`meds_shard_paths`).

    Examples:
        >>> report = size_meds(simple_static_MEDS)
        >>> report.kind, report.source == str(simple_static_MEDS)
        ('meds', True)
        >>> report.to_frame().select(
        ...     "split", "n_subjects", "n_events", "n_measurements", "n_dynamic_measurements"
        ... )
        shape: (4, 5)
        ┌───────────┬────────────┬──────────┬────────────────┬────────────────────────┐
        │ split     ┆ n_subjects ┆ n_events ┆ n_measurements ┆ n_dynamic_measurements │
        │ ---       ┆ ---        ┆ ---      ┆ ---            ┆ ---                    │
        │ str       ┆ i64        ┆ i64      ┆ i64            ┆ i64                    │
        ╞═══════════╪════════════╪══════════╪════════════════╪════════════════════════╡
        │ held_out  ┆ 1          ┆ 5        ┆ 11             ┆ 9                      │
        │ train     ┆ 4          ┆ 20       ┆ 44             ┆ 36                     │
        │ tuning    ┆ 1          ┆ 3        ┆ 7              ┆ 5                      │
        │ __total__ ┆ 6          ┆ 28       ┆ 62             ┆ 50                     │
        └───────────┴────────────┴──────────┴────────────────┴────────────────────────┘

        The fixture has 62 rows, 12 of them static, spread over 6 subjects and 4 shards; ``train``'s 20
        events span *both* of its shards, which a per-shard count would double-count only if an event
        straddled a shard, and undercount not at all:

        >>> total = report.total
        >>> total.n_measurements, total.n_static_measurements, total.n_subjects_with_static
        (62, 12, 6)
        >>> total.n_shards, total.n_bytes > 0
        (4, True)

        ``n_codes_observed`` is the union over splits (11), not their sum (25), and it is emphatically
        not the height of ``metadata/codes.parquet``, which ships only 5 of those 11 codes:

        >>> total.n_codes_observed, sum(s.n_codes_observed for s in report.splits)
        (11, 25)
        >>> pl.read_parquet(simple_static_MEDS / "metadata" / "codes.parquet").height
        5

        ``observed_code_set_id`` digests that same set of codes, sorted, so two datasets can be
        compared for vocabulary equality without shipping a vocabulary. ``train`` happens to observe
        all 11 codes and so carries the dataset's own id; the two evaluation splits do not:

        >>> total.observed_code_set_id
        'sha256:4e3cf86133348078e139dd03eb0880907d847bb2991bbcf5194e188099fb4447'
        >>> {s.split: s.observed_code_set_id == total.observed_code_set_id for s in report.splits}
        {'held_out': False, 'train': True, 'tuning': False}

        Under the ``meds_summary_stats`` event convention the same dataset has 34 events:

        >>> total.n_events + total.n_subjects_with_static
        34

        An explicit ``splits`` mapping overrides the file. A partial mapping puts the rest in
        ``"__unknown__"``, and because subjects 239684 and 1195293 share the shard ``train/0``, that one
        file is attributed to two different splits -- which is why the report carries the deduplicated
        dataset ``n_shards`` (4) rather than summing the column (5):

        >>> partial = size_meds(simple_static_MEDS, splits={239684: "a", 1195293: "b"})
        >>> partial.to_frame().select("split", "n_subjects", "n_measurements", "n_shards")
        shape: (4, 4)
        ┌─────────────┬────────────┬────────────────┬──────────┐
        │ split       ┆ n_subjects ┆ n_measurements ┆ n_shards │
        │ ---         ┆ ---        ┆ ---            ┆ ---      │
        │ str         ┆ i64        ┆ i64            ┆ i64      │
        ╞═════════════╪════════════╪════════════════╪══════════╡
        │ __unknown__ ┆ 4          ┆ 32             ┆ 3        │
        │ a           ┆ 1          ┆ 13             ┆ 1        │
        │ b           ┆ 1          ┆ 17             ┆ 1        │
        │ __total__   ┆ 6          ┆ 62             ┆ 4        │
        └─────────────┴────────────┴────────────────┴──────────┘

        With neither a mapping nor a splits file, the shard prefix is used; with neither of those, one
        ``"__all__"`` split covers everything:

        >>> when = datetime(2020, 1, 1)
        >>> rows = {"subject_id": [1, 1, 2], "time": [None, when, None], "code": ["A", "B", "A"]}
        >>> with yaml_disk({"data/0.parquet": rows}) as root:
        ...     flat = size_meds(root)
        >>> [(s.split, s.n_subjects, s.n_events, s.n_subjects_with_static) for s in flat.splits]
        [('__all__', 2, 1, 2)]

        Shards need not agree on their columns: MEDS allows extra ones and does not backfill the
        optional ones, so exactly the three MEDS columns are read, at their spec'd dtypes, from every
        shard. That the first shard here is empty (all-null dtypes on disk) and the second static-only
        does not change what the rest of them mean, and neither does a shard that carries the reserved
        ``__path__``/``__split__`` names as extra columns. An empty shard counts toward the dataset's
        ``n_shards`` while belonging to no split, so the per-split column can sum to less than the
        dataset's total as well as more:

        >>> files = {
        ...     "data/train/0.parquet": {"subject_id": [], "time": [], "code": []},
        ...     "data/train/1.parquet": {"subject_id": [2], "time": [None], "code": ["C"]},
        ...     "data/train/2.parquet": {"subject_id": [1, 1], "time": [None, when],
        ...                              "code": ["A", "B"], "numeric_value": [None, 1.0],
        ...                              "__path__": ["x", "y"], "__split__": ["y", "z"]},
        ... }
        >>> with yaml_disk(files) as root:
        ...     mixed = size_meds(root)
        >>> [(s.split, s.n_subjects, s.n_measurements, s.n_shards) for s in mixed.splits]
        [('train', 2, 3, 2)]
        >>> mixed.n_shards
        3

        The shard list is taken from the filesystem and handed to the scan as literal paths, so a root
        whose name holds a glob metacharacter is read, not re-expanded as a pattern:

        >>> with yaml_disk({"cohort[1]/data/train/0.parquet": rows}) as tmp:
        ...     size_meds(tmp / "cohort[1]").total.n_measurements
        3

        A root with no shards is an error, not an empty report -- it is nearly always a wrong path:

        >>> with yaml_disk({"metadata/codes.parquet": None}) as root:
        ...     size_meds(root)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No MEDS shards found under ...data
    """
    paths = meds_shard_paths(root)
    if not paths:
        raise FileNotFoundError(f"No MEDS shards found under {root / meds.data_subdirectory}")

    by_subject = _subject_split_frame(root, splits)
    # An explicit list of files, never the directory: a directory source would recurse into `.logs/`.
    # `glob=False` because those files are already resolved -- a root or shard whose name holds `*`,
    # `?` or `[` would otherwise be re-expanded as a pattern, quietly scanning a sibling's shards or
    # the same shard twice. MEDS allows extra columns and does not backfill optional ones, so shards
    # legitimately differ in width; `_SCAN_SCHEMA` pins the three that are read.
    lf = pl.scan_parquet(
        paths,
        schema=_SCAN_SCHEMA,
        cast_options=_CAST_OPTIONS,
        glob=False,
        include_file_paths=_PATH_COL,
        missing_columns="insert",
        extra_columns="ignore",
    )
    if by_subject is None:
        by_path = {str(p): _split_prefix(shard_name(root, p)) for p in paths}
        lf = lf.with_columns(
            pl.col(_PATH_COL).replace_strict(by_path, return_dtype=pl.String).alias(_SPLIT_COL)
        )
    else:
        lf = lf.join(by_subject.lazy(), on=_SUBJECT_ID, how="left").with_columns(
            pl.col(_SPLIT_COL).fill_null(UNKNOWN_SPLIT)
        )

    timed = pl.col(_TIME).is_not_null()
    per_split = lf.group_by(_SPLIT_COL).agg(
        pl.col(_SUBJECT_ID).n_unique().alias("n_subjects"),
        pl.struct(_SUBJECT_ID, _TIME).filter(timed).n_unique().alias("n_events"),
        pl.len().alias("n_measurements"),
        timed.sum().alias("n_dynamic_measurements"),
        (~timed).sum().alias("n_static_measurements"),
        pl.col(_SUBJECT_ID).filter(~timed).n_unique().alias("n_subjects_with_static"),
        pl.col(_CODE).drop_nulls().unique().alias("codes"),
        pl.col(_PATH_COL).unique().alias("shards"),
    )
    overall = lf.select(pl.col(_CODE).drop_nulls().unique().alias("codes"))
    # Collected together so the two queries share one streaming pass over the shards.
    per_split_df, overall_df = pl.collect_all([per_split, overall], engine="streaming")

    n_bytes = {str(p): p.stat().st_size for p in paths}
    split_sizes = []
    for row in per_split_df.sort(_SPLIT_COL).iter_rows(named=True):
        shards = row.pop("shards")
        codes = sorted(row.pop("codes"))
        row["split"] = row.pop(_SPLIT_COL)
        split_sizes.append(
            SplitSizes(
                **row,
                n_codes_observed=len(codes),
                observed_code_set_id=digest(codes),
                n_shards=len(shards),
                n_bytes=sum(n_bytes[s] for s in shards),
            )
        )
    all_codes = sorted(overall_df.to_series().to_list())
    return SizeReport(
        source=str(root),
        kind="meds",
        splits=tuple(split_sizes),
        n_codes_observed=len(all_codes),
        observed_code_set_id=digest(all_codes),
        n_shards=len(paths),
        n_bytes=sum(n_bytes.values()),
    )


def nrt_header(path: Path) -> dict[str, dict]:
    r"""Read only the safetensors header of a nested-ragged-tensor (``.nrt``) file.

    The format is a little-endian ``u64`` header length, then exactly that many bytes of JSON mapping
    each tensor name to its ``dtype``, ``shape``, and ``data_offsets``. Every count MTD sizing needs is
    a ``shape``, so the tensor bytes -- which are the entire dataset -- are never read.

    Args:
        path: The ``.nrt`` file.

    Returns:
        The header, with the optional ``__metadata__`` key removed so that every remaining key is a
        tensor name.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is shorter than the 8-byte length prefix, is truncated relative to the
            length it declares (checked against the file size *before* anything is read, so a file that
            is not safetensors at all is reported rather than allocated for), or does not hold a JSON
            object.

    Examples:
        >>> def write_nrt(path, header):
        ...     blob = json.dumps(header).encode()
        ...     path.parent.mkdir(parents=True, exist_ok=True)
        ...     path.write_bytes(len(blob).to_bytes(8, "little") + blob + b"\x00" * 32)
        >>> entry = {"dtype": "I64", "shape": [3], "data_offsets": [0, 24]}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     nrt = Path(tmp) / "0.nrt"
        ...     write_nrt(nrt, {"dim1/bounds": entry, "__metadata__": {"format": "pt"}})
        ...     header = nrt_header(nrt)
        >>> header
        {'dim1/bounds': {'dtype': 'I64', 'shape': [3], 'data_offsets': [0, 24]}}

        An all-empty shard tokenizes to a *keyless* ``.nrt``. That is valid, and must read as an empty
        header rather than an error:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     nrt = Path(tmp) / "empty.nrt"
        ...     write_nrt(nrt, {})
        ...     nrt_header(nrt)
        {}

        A file too short to hold the length prefix:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     nrt = Path(tmp) / "short.nrt"
        ...     _ = nrt.write_bytes(b"\x01\x02")
        ...     nrt_header(nrt)
        Traceback (most recent call last):
            ...
        ValueError: ...short.nrt is not a valid safetensors file: expected an 8-byte header length,
        got 2 bytes

        A file whose declared header runs past the end of the file:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     nrt = Path(tmp) / "truncated.nrt"
        ...     _ = nrt.write_bytes((999).to_bytes(8, "little") + b'{"a": 1}')
        ...     nrt_header(nrt)
        Traceback (most recent call last):
            ...
        ValueError: ...truncated.nrt is truncated: the header claims 999 bytes but only 8 follow the
        length prefix

        The declared length is checked *before* the read, so a file that is not a ``.nrt`` at all --
        whose first eight bytes are then an arbitrary 64-bit number -- is reported rather than
        attempted: reading it would ask the allocator for exabytes and die with a bare ``MemoryError``.

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     nrt = Path(tmp) / "huge.nrt"
        ...     _ = nrt.write_bytes((2**62).to_bytes(8, "little") + b"{}")
        ...     nrt_header(nrt)
        Traceback (most recent call last):
            ...
        ValueError: ...huge.nrt is truncated: the header claims 4611686018427387904 bytes but only 2
        follow the length prefix

        And a header that is not JSON, or is JSON but not an object:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     nrt = Path(tmp) / "garbage.nrt"
        ...     _ = nrt.write_bytes((5).to_bytes(8, "little") + b"{not!")
        ...     nrt_header(nrt)
        Traceback (most recent call last):
            ...
        ValueError: ...garbage.nrt has an unparseable safetensors header: ...
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     nrt = Path(tmp) / "list.nrt"
        ...     write_nrt(nrt, [1, 2])
        ...     nrt_header(nrt)
        Traceback (most recent call last):
            ...
        ValueError: ...list.nrt has an invalid safetensors header: expected a JSON object, got list
    """
    n_available = max(path.stat().st_size - 8, 0)
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(
                f"{path} is not a valid safetensors file: expected an 8-byte header length, got "
                f"{len(prefix)} bytes"
            )
        # Checked against the file before the read, never after: the declared length is 64 unvalidated
        # bits out of an untrusted file, and reading it first turns a mistyped path into a MemoryError.
        n_header_bytes = int.from_bytes(prefix, "little")
        if n_header_bytes > n_available:
            raise ValueError(
                f"{path} is truncated: the header claims {n_header_bytes} bytes but only "
                f"{n_available} follow the length prefix"
            )
        raw = f.read(n_header_bytes)
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} has an unparseable safetensors header: {e}") from e
    if not isinstance(header, dict):
        raise ValueError(
            f"{path} has an invalid safetensors header: expected a JSON object, got {type(header).__name__}"
        )
    return {k: v for k, v in header.items() if k != "__metadata__"}


def _nrt_length(header: Mapping[str, dict], key: str) -> int:
    """Return the length of the leading axis of one ``.nrt`` tensor, or 0 if it is absent.

    Args:
        header: A parsed header, as returned by :func:`nrt_header`.
        key: The tensor name.

    Returns:
        ``shape[0]``, or 0 if the key is missing or the tensor is a scalar. A keyless header is what an
        all-empty shard produces, so a missing key is a zero, not an error.

    Examples:
        >>> header = {"dim2/code": {"shape": [17]}, "scalar": {"shape": []}}
        >>> _nrt_length(header, "dim2/code"), _nrt_length(header, "scalar"), _nrt_length(header, "nope")
        (17, 0, 0)
    """
    entry = header.get(key)
    if not entry:
        return 0
    shape = entry.get("shape") or []
    return int(shape[0]) if shape else 0


def size_tensorized(root: Path) -> SizeReport:
    r"""Count an MTD tensorized cohort exactly, from schema frames and ``.nrt`` headers only.

    Shards are enumerated from ``<root>/tokenization/schemas`` (sorted, dotfile-skipping) and each is
    paired with ``<root>/data/<shard>.nrt``. Per shard:

    - ``n_subjects`` is the **height of the schemas frame**, never ``len(dim1/bounds)``: subjects with
      only static measurements are absent from the ``.nrt`` entirely, so ``dim1/bounds`` undercounts
      them.
    - ``n_events`` is ``len(dim1/time_delta_days)`` and ``n_dynamic_measurements`` is
      ``len(dim2/code)``, both read from the header.
    - statics come from the schemas frame's ``static_code`` list lengths, which are null for a subject
      with no statics.

    A tensorized cohort has no ``subject_splits.parquet``, so splits come from the shard-name prefix
    (``<split>/<n>``), or ``"__all__"`` if the shards are flat. Its ``metadata/codes.parquet`` is a
    single global vocabulary and is required, not optional: the codes in a tensorized cohort are
    vocabulary *indices* living inside tensors that header-only sizing never opens, so the observed set
    cannot be recovered here. ``n_codes_observed`` and ``observed_code_set_id`` therefore describe that
    vocabulary and repeat on every split rather than partitioning -- the one place in this module where
    "observed" means the metadata file rather than the data.

    Args:
        root: The tensorized cohort root.

    Returns:
        A :class:`SizeReport` with ``kind="tensorized"``, splits sorted by label.

    Raises:
        FileNotFoundError: If no schema shards exist, if a schema shard has no matching ``.nrt``, if
            ``metadata/codes.parquet`` is absent, or if a shard is a dangling symlink.
        ValueError: If ``metadata/codes.parquet`` has no ``code`` column, or a schema shard has no
            list-typed ``static_code`` column.

    Examples:
        >>> def write_nrt(path, lengths):
        ...     header = {k: {"dtype": "I64", "shape": [n]} for k, n in lengths.items()}
        ...     blob = json.dumps(header).encode()
        ...     path.parent.mkdir(parents=True, exist_ok=True)
        ...     path.write_bytes(len(blob).to_bytes(8, "little") + blob)

        Shard ``train/0`` holds three subjects, but only two of them have timed events, so
        ``dim1/bounds`` is 2 while ``n_subjects`` is 3. Shard ``train/1`` is all-empty and so has a
        keyless ``.nrt``:

        >>> cohort = {
        ...     "tokenization/schemas/train/0.parquet": {
        ...         "subject_id": [1, 2, 3], "static_code": [[10, 11], [12], None],
        ...     },
        ...     "tokenization/schemas/train/1.parquet": {"subject_id": [4], "static_code": [[]]},
        ...     "metadata/codes.parquet": {
        ...         "code": ["A", "B", "C", "D", "E"], "vocab_index": [10, 11, 12, 13, 14],
        ...     },
        ... }
        >>> with yaml_disk(cohort) as root:
        ...     write_nrt(root / "data/train/0.nrt",
        ...               {"dim1/bounds": 2, "dim1/time_delta_days": 7, "dim2/code": 20})
        ...     write_nrt(root / "data/train/1.nrt", {})
        ...     report = size_tensorized(root)
        >>> report.kind
        'tensorized'
        >>> report.to_frame().select("split", "n_subjects", "n_events", "n_dynamic_measurements")
        shape: (2, 4)
        ┌───────────┬────────────┬──────────┬────────────────────────┐
        │ split     ┆ n_subjects ┆ n_events ┆ n_dynamic_measurements │
        │ ---       ┆ ---        ┆ ---      ┆ ---                    │
        │ str       ┆ i64        ┆ i64      ┆ i64                    │
        ╞═══════════╪════════════╪══════════╪════════════════════════╡
        │ train     ┆ 4          ┆ 7        ┆ 20                     │
        │ __total__ ┆ 4          ┆ 7        ┆ 20                     │
        └───────────┴────────────┴──────────┴────────────────────────┘

        The statics come from the schema frame's ``static_code`` lists: two codes for subject 1, one for
        subject 2, a null for subject 3, and an empty list for subject 4.

        >>> report.to_frame().select("split", "n_static_measurements", "n_subjects_with_static")
        shape: (2, 3)
        ┌───────────┬───────────────────────┬────────────────────────┐
        │ split     ┆ n_static_measurements ┆ n_subjects_with_static │
        │ ---       ┆ ---                   ┆ ---                    │
        │ str       ┆ i64                   ┆ i64                    │
        ╞═══════════╪═══════════════════════╪════════════════════════╡
        │ train     ┆ 3                     ┆ 2                      │
        │ __total__ ┆ 3                     ┆ 2                      │
        └───────────┴───────────────────────┴────────────────────────┘

        ``n_measurements`` is dynamic plus static, and both shard files count toward ``n_shards``.
        ``n_codes_observed`` is the *global vocabulary* -- five here, even though the cohort's statics
        reference only three of those five entries and its dynamic codes cannot be seen at all from
        headers -- so on this path alone the name is a size of ``metadata/codes.parquet``:

        >>> report.total.n_measurements, report.total.n_codes_observed, report.total.n_shards
        (23, 5, 2)

        Every parquet path is handed to the reader literally, so the same cohort under a directory
        whose name holds a glob metacharacter reads the same rather than expanding as a pattern:

        >>> with yaml_disk({f"c[1]/{k}": v for k, v in cohort.items()}) as tmp:
        ...     write_nrt(tmp / "c[1]/data/train/0.nrt",
        ...               {"dim1/bounds": 2, "dim1/time_delta_days": 7, "dim2/code": 20})
        ...     write_nrt(tmp / "c[1]/data/train/1.nrt", {})
        ...     size_tensorized(tmp / "c[1]").total.to_dict() == report.total.to_dict()
        True

        A schemas shard with no ``.nrt`` names the file it wanted:

        >>> shard = {"tokenization/schemas/0.parquet": {"static_code": [[1]]},
        ...          "metadata/codes.parquet": {"code": ["A"]}}
        >>> with yaml_disk(shard) as root:
        ...     size_tensorized(root)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No tensorized shard at ...data/0.nrt for schema shard ...schemas/0.parquet

        As does a cohort with no schemas at all:

        >>> with yaml_disk({"metadata/codes.parquet": {"code": ["A"]}}) as root:
        ...     size_tensorized(root)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No tensorized schema shards found under ...tokenization/schemas

        A cohort with no ``metadata/codes.parquet`` is an error too, rather than a cohort with a
        vocabulary of zero: the file is one of the three a tensorized cohort is made of, and a silent
        zero would join into a scaling-law table as if it were a count:

        >>> with yaml_disk({"tokenization/schemas/0.parquet": {"static_code": [[1]]}}) as root:
        ...     size_tensorized(root)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No code metadata at ...metadata/codes.parquet

        As is one whose code metadata is not code metadata:

        >>> with yaml_disk({"tokenization/schemas/0.parquet": {"static_code": [[1]]},
        ...                 "metadata/codes.parquet": {"vocab_index": [1]}}) as root:
        ...     size_tensorized(root)
        Traceback (most recent call last):
            ...
        ValueError: ...codes.parquet is not code metadata: expected a 'code' column, got
        ['vocab_index']

        And a frame that is not a tokenization schema at all is rejected before it is miscounted:

        >>> with yaml_disk({"tokenization/schemas/0.parquet": {"subject_id": [1]},
        ...                 "metadata/codes.parquet": {"code": ["A"]}}) as root:
        ...     write_nrt(root / "data/0.nrt", {})
        ...     size_tensorized(root)
        Traceback (most recent call last):
            ...
        ValueError: ...0.parquet is not a tokenization schema shard: expected a list-typed 'static_code'
        column, got None
    """
    schemas_dir = root / TENSORIZED_SCHEMAS_SUBDIR
    paths = _parquet_shard_paths(schemas_dir)
    if not paths:
        raise FileNotFoundError(f"No tensorized schema shards found under {schemas_dir}")

    codes_path = root / meds.code_metadata_filepath
    if not codes_path.is_file():
        raise FileNotFoundError(f"No code metadata at {codes_path}")
    codes_lf = pl.scan_parquet(codes_path, glob=False)
    columns = codes_lf.collect_schema().names()
    if _CODE not in columns:
        raise ValueError(f"{codes_path} is not code metadata: expected a {_CODE!r} column, got {columns}")
    # Cast for the same reason `_SCAN_SCHEMA` pins its columns: MEDS says `code` is a string, and the
    # id must not depend on whether a file stored it as one or as a dictionary-encoded column.
    codes = codes_lf.select(pl.col(_CODE).cast(pl.String).drop_nulls().unique()).collect()
    vocabulary = sorted(codes.to_series())
    n_codes = len(vocabulary)
    code_set_id = digest(vocabulary)

    per_split: dict[str, dict[str, int]] = {}
    n_bytes = 0
    for path in paths:
        name = _shard_name(schemas_dir, path)
        nrt_path = root / meds.data_subdirectory / f"{name}{NRT_SUFFIX}"
        if not nrt_path.is_file():
            raise FileNotFoundError(f"No tensorized shard at {nrt_path} for schema shard {path}")
        header = nrt_header(nrt_path)

        schema_lf = pl.scan_parquet(path, glob=False)
        dtype = schema_lf.collect_schema().get(_STATIC_CODE)
        if not isinstance(dtype, pl.List):
            raise ValueError(
                f"{path} is not a tokenization schema shard: expected a list-typed {_STATIC_CODE!r} "
                f"column, got {dtype}"
            )
        n_statics = pl.col(_STATIC_CODE).list.len().fill_null(0)
        counts = schema_lf.select(
            pl.len().alias("n_subjects"),
            n_statics.sum().alias("n_static_measurements"),
            (n_statics > 0).sum().alias("n_subjects_with_static"),
        ).collect()

        shard = dict(counts.row(0, named=True))
        # `dim1/bounds` counts only subjects with a timed event, so `n_subjects` comes from the schema.
        shard["n_events"] = _nrt_length(header, "dim1/time_delta_days")
        shard["n_dynamic_measurements"] = _nrt_length(header, "dim2/code")
        shard["n_measurements"] = shard["n_dynamic_measurements"] + shard["n_static_measurements"]
        shard["n_shards"] = 1
        shard["n_bytes"] = path.stat().st_size + nrt_path.stat().st_size
        n_bytes += shard["n_bytes"]

        totals = per_split.setdefault(_split_prefix(name), dict.fromkeys(shard, 0))
        for key, value in shard.items():
            totals[key] += value

    splits = tuple(
        SplitSizes(split=split, n_codes_observed=n_codes, observed_code_set_id=code_set_id, **totals)
        for split, totals in sorted(per_split.items())
    )
    return SizeReport(
        source=str(root),
        kind="tensorized",
        splits=splits,
        n_codes_observed=n_codes,
        observed_code_set_id=code_set_id,
        n_shards=len(paths),
        n_bytes=n_bytes,
    )


def _n_windows(length: int, max_seq_len: int, stride: int | None) -> int:
    """Return how many windows one sequence is cut into.

    This mirrors MTD's ``STEP_THROUGH`` geometry, in which window *ends* run ``max_seq_len``,
    ``max_seq_len + stride``, ... and a final end is appended at the sequence's own end whenever the
    stride did not already land there. Every window is then ``max_seq_len`` wide -- the last one is
    anchored to the end rather than allowed to run short -- so the count is
    ``1 + ceil((length - max_seq_len) / stride)``.

    Args:
        length: The sequence length, in whatever unit ``max_seq_len`` counts.
        max_seq_len: The window size.
        stride: The step between window starts, or ``None`` for truncation (a single window).

    Returns:
        1 when truncating or when the sequence fits, else the number of strided windows needed to reach
        the end of the sequence.

    Examples:
        >>> _n_windows(100, 40, None), _n_windows(30, 40, 20), _n_windows(100, 40, 20)
        (1, 1, 4)
        >>> _n_windows(41, 40, 20), _n_windows(0, 40, 20)
        (2, 1)

        A stride equal to the window still needs a second window for one extra element, because that
        window is full width and backs onto the end of the sequence:

        >>> _n_windows(80, 40, 40), _n_windows(50, 40, 40)
        (2, 2)
    """
    if stride is None or length <= max_seq_len:
        return 1
    return 1 + (length - max_seq_len + stride - 1) // stride


def project_trained_tokens(
    *,
    events_per_subject: Sequence[int],
    measurements_per_subject: Sequence[int],
    statics_per_subject: Sequence[int],
    max_seq_len: int,
    batch_mode: str = "SM",
    static_inclusion_mode: str = "INCLUDE",
    step_through_stride: int | None = None,
) -> dict[str, int]:
    r"""Project how many tokens one epoch actually trains on, given a loader configuration.

    **A corpus token count is not a trained-on token count.** Truncation to ``max_seq_len`` *loses*
    tokens -- a subject longer than the window contributes only a window's worth per epoch, forever --
    while stepping through a long subject in overlapping windows *gains* them, by revisiting the same
    measurements in several windows (measured at 1.67x the corpus count on a real cohort). The two
    effects run in opposite directions and neither is small, which is why the size reports in this
    module stay config-free and this projection is separate and explicitly parameterized.

    Under ``RANDOM`` subsequence sampling the realized count is a *random variable*: which window a
    subject contributes is redrawn every epoch. The number here is then the expectation, and it is
    exact only in the sense that every draw has the same length. For the deterministic strategies
    (``FROM_START``, ``TO_END``) it is exact; for step-through it is an upper bound on distinct tokens
    seen, since overlapping windows re-present the same measurements.

    The two batch modes count ``max_seq_len`` in different units, which is the single most consequential
    thing to get right:

    - ``"SM"`` (subject-measurement): the window is ``max_seq_len`` *measurements*. Everything here is
      exact integer arithmetic.
    - ``"SEM"`` (subject-event-measurement): the window is ``max_seq_len`` *events*, and the token count
      then depends on how measurements are distributed over the events inside the window. With only
      per-subject totals to work from, this function assumes a uniform ``measurements / events`` density
      per subject; it is exact when every event of a subject has the same number of measurements and an
      approximation otherwise.

    Step-through windows follow MTD's geometry, which is *not* a partition of the sequence: window ends
    run ``max_seq_len``, ``max_seq_len + stride``, ..., and a final end is appended at the end of the
    sequence. Every window is therefore full width, the last one backing onto the sequence end, and a
    subject of ``length`` units yields ``1 + ceil((length - max_seq_len) / stride)`` of them. Nothing is
    dropped and the overshoot is reported as ``n_repeated_tokens`` -- even when ``stride ==
    max_seq_len``, where the final window still overlaps its predecessor unless the stride divides the
    sequence exactly.

    Static handling follows MTD's three modes:

    - ``"OMIT"``: statics are not delivered at all.
    - ``"INCLUDE"``: statics ride alongside every window in their own tensors, so they cost no window
      budget and are re-delivered once per window.
    - ``"PREPEND"``: statics are prepended into the sequence itself, so they compete for the budget. In
      ``"SM"`` they cost one token each; in ``"SEM"`` they cost a single *event* slot no matter how many
      static codes there are.

    Args:
        events_per_subject: Per-subject timed event counts.
        measurements_per_subject: Per-subject dynamic measurement counts, in the same subject order.
        statics_per_subject: Per-subject static measurement counts, in the same subject order.
        max_seq_len: The window size: measurements in ``"SM"``, events in ``"SEM"``.
        batch_mode: One of :data:`BATCH_MODES`.
        static_inclusion_mode: One of :data:`STATIC_INCLUSION_MODES`.
        step_through_stride: Window stride for MTD's ``STEP_THROUGH`` iteration, in the same unit as
            ``max_seq_len``. ``None`` (the default) means each subject yields exactly one window per
            epoch, which is what every truncating sampling strategy does.

    Returns:
        A dict of ``n_subjects``, ``n_windows`` (items per epoch), ``n_corpus_tokens`` (what the data
        holds, under this static mode), ``n_trained_tokens`` (what the model sees per epoch),
        ``n_static_tokens`` (the static share of that), ``n_dropped_tokens`` (per-subject corpus tokens
        never reached), and ``n_repeated_tokens`` (per-subject tokens seen more than once).

    Raises:
        ValueError: If the three per-subject sequences differ in length, if ``max_seq_len`` or
            ``step_through_stride`` is below 1, or if a mode is not recognized.

    Examples:
        Two subjects with 10 and 20 dynamic measurements and 2 statics each. With a window big enough
        for everyone and statics delivered separately, an epoch trains on the whole corpus:

        >>> events, measurements, statics = [3, 3], [10, 20], [2, 2]
        >>> project_trained_tokens(events_per_subject=events, measurements_per_subject=measurements,
        ...                        statics_per_subject=statics, max_seq_len=100)
        {'n_subjects': 2, 'n_windows': 2, 'n_corpus_tokens': 34, 'n_trained_tokens': 34,
         'n_static_tokens': 4, 'n_dropped_tokens': 0, 'n_repeated_tokens': 0}

        Shrink the window to 15 and the second subject is truncated -- 5 of its measurements are never
        trained on, in any epoch:

        >>> project_trained_tokens(events_per_subject=events, measurements_per_subject=measurements,
        ...                        statics_per_subject=statics, max_seq_len=15)
        {'n_subjects': 2, 'n_windows': 2, 'n_corpus_tokens': 34, 'n_trained_tokens': 29,
         'n_static_tokens': 4, 'n_dropped_tokens': 5, 'n_repeated_tokens': 0}

        Prepending the statics instead makes them compete for the same budget, so the second subject
        loses 7 rather than 5:

        >>> project_trained_tokens(events_per_subject=events, measurements_per_subject=measurements,
        ...                        statics_per_subject=statics, max_seq_len=15,
        ...                        static_inclusion_mode="PREPEND")
        {'n_subjects': 2, 'n_windows': 2, 'n_corpus_tokens': 34, 'n_trained_tokens': 27,
         'n_static_tokens': 4, 'n_dropped_tokens': 7, 'n_repeated_tokens': 0}

        Omitting them shrinks the corpus itself:

        >>> project_trained_tokens(events_per_subject=events, measurements_per_subject=measurements,
        ...                        statics_per_subject=statics, max_seq_len=15,
        ...                        static_inclusion_mode="OMIT")
        {'n_subjects': 2, 'n_windows': 2, 'n_corpus_tokens': 30, 'n_trained_tokens': 25,
         'n_static_tokens': 0, 'n_dropped_tokens': 5, 'n_repeated_tokens': 0}

        Stepping through runs the other way: one 100-measurement subject, a 40-token window and a stride
        of 20 gives four windows per epoch and 1.6x the corpus, none of it dropped:

        >>> stepped = project_trained_tokens(
        ...     events_per_subject=[10], measurements_per_subject=[100], statics_per_subject=[0],
        ...     max_seq_len=40, static_inclusion_mode="OMIT", step_through_stride=20,
        ... )
        >>> stepped["n_windows"], stepped["n_trained_tokens"], stepped["n_repeated_tokens"]
        (4, 160, 60)
        >>> stepped["n_trained_tokens"] / stepped["n_corpus_tokens"]
        1.6

        Every window is full width, including the last: MTD anchors the final window to the end of the
        sequence rather than letting it run short. So a stride equal to the window does *not* partition
        a sequence whose length is not a multiple of it -- 50 measurements in windows of 40 give
        ``[0, 40)`` and ``[10, 50)``, which re-present 30 measurements and drop none:

        >>> project_trained_tokens(events_per_subject=[5], measurements_per_subject=[50],
        ...                        statics_per_subject=[0], max_seq_len=40,
        ...                        static_inclusion_mode="OMIT", step_through_stride=40)
        {'n_subjects': 1, 'n_windows': 2, 'n_corpus_tokens': 50, 'n_trained_tokens': 80,
         'n_static_tokens': 0, 'n_dropped_tokens': 0, 'n_repeated_tokens': 30}

        The same holds in ``"SEM"``, where the unit is events: 5 events in windows of 4 are covered by
        ``[0, 4)`` and ``[1, 5)``, so 8 of the 5 subject-events are presented and the 50 measurements
        are re-presented in proportion:

        >>> project_trained_tokens(events_per_subject=[5], measurements_per_subject=[50],
        ...                        statics_per_subject=[0], max_seq_len=4, batch_mode="SEM",
        ...                        static_inclusion_mode="OMIT", step_through_stride=4)
        {'n_subjects': 1, 'n_windows': 2, 'n_corpus_tokens': 50, 'n_trained_tokens': 80,
         'n_static_tokens': 0, 'n_dropped_tokens': 0, 'n_repeated_tokens': 30}

        In ``"SEM"`` the window counts events, so a 4-event subject with 20 measurements and a 2-event
        window trains on about half its measurements:

        >>> project_trained_tokens(events_per_subject=[4], measurements_per_subject=[20],
        ...                        statics_per_subject=[3], max_seq_len=2, batch_mode="SEM",
        ...                        static_inclusion_mode="OMIT")
        {'n_subjects': 1, 'n_windows': 1, 'n_corpus_tokens': 20, 'n_trained_tokens': 10,
         'n_static_tokens': 0, 'n_dropped_tokens': 10, 'n_repeated_tokens': 0}

        Prepending in ``"SEM"`` spends one of those two event slots on the statics, so only one event's
        worth of measurements survives -- but all three static codes do, since they share a slot:

        >>> project_trained_tokens(events_per_subject=[4], measurements_per_subject=[20],
        ...                        statics_per_subject=[3], max_seq_len=2, batch_mode="SEM",
        ...                        static_inclusion_mode="PREPEND")
        {'n_subjects': 1, 'n_windows': 1, 'n_corpus_tokens': 23, 'n_trained_tokens': 8,
         'n_static_tokens': 3, 'n_dropped_tokens': 15, 'n_repeated_tokens': 0}

        A subject with no timed events at all (statics only) contributes no dynamic tokens and does not
        divide by zero:

        >>> project_trained_tokens(events_per_subject=[0], measurements_per_subject=[0],
        ...                        statics_per_subject=[5], max_seq_len=8, batch_mode="SEM")
        {'n_subjects': 1, 'n_windows': 1, 'n_corpus_tokens': 5, 'n_trained_tokens': 5,
         'n_static_tokens': 5, 'n_dropped_tokens': 0, 'n_repeated_tokens': 0}

        Unknown modes are named, not guessed:

        >>> project_trained_tokens(events_per_subject=[1], measurements_per_subject=[1],
        ...                        statics_per_subject=[1], max_seq_len=4, batch_mode="sm")
        Traceback (most recent call last):
            ...
        ValueError: Unknown batch_mode 'sm'; must be one of ('SM', 'SEM')
        >>> project_trained_tokens(events_per_subject=[1], measurements_per_subject=[1],
        ...                        statics_per_subject=[1], max_seq_len=4,
        ...                        static_inclusion_mode="drop")
        Traceback (most recent call last):
            ...
        ValueError: Unknown static_inclusion_mode 'drop'; must be one of ('OMIT', 'INCLUDE', 'PREPEND')

        As are mismatched inputs and nonsensical window geometry:

        >>> project_trained_tokens(events_per_subject=[1, 2], measurements_per_subject=[1],
        ...                        statics_per_subject=[1], max_seq_len=4)
        Traceback (most recent call last):
            ...
        ValueError: events_per_subject, measurements_per_subject, and statics_per_subject must be the
        same length; got 2, 1, and 1
        >>> project_trained_tokens(events_per_subject=[1], measurements_per_subject=[1],
        ...                        statics_per_subject=[1], max_seq_len=0)
        Traceback (most recent call last):
            ...
        ValueError: max_seq_len must be >= 1; got 0
        >>> project_trained_tokens(events_per_subject=[1], measurements_per_subject=[1],
        ...                        statics_per_subject=[1], max_seq_len=4, step_through_stride=0)
        Traceback (most recent call last):
            ...
        ValueError: step_through_stride must be >= 1 when given; got 0
    """
    n_subjects = len(events_per_subject)
    if len(measurements_per_subject) != n_subjects or len(statics_per_subject) != n_subjects:
        raise ValueError(
            "events_per_subject, measurements_per_subject, and statics_per_subject must be the same "
            f"length; got {n_subjects}, {len(measurements_per_subject)}, and {len(statics_per_subject)}"
        )
    if batch_mode not in BATCH_MODES:
        raise ValueError(f"Unknown batch_mode {batch_mode!r}; must be one of {BATCH_MODES}")
    if static_inclusion_mode not in STATIC_INCLUSION_MODES:
        raise ValueError(
            f"Unknown static_inclusion_mode {static_inclusion_mode!r}; must be one of "
            f"{STATIC_INCLUSION_MODES}"
        )
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1; got {max_seq_len}")
    if step_through_stride is not None and step_through_stride < 1:
        raise ValueError(f"step_through_stride must be >= 1 when given; got {step_through_stride}")

    projection = {
        "n_subjects": n_subjects,
        "n_windows": 0,
        "n_corpus_tokens": 0,
        "n_trained_tokens": 0,
        "n_static_tokens": 0,
        "n_dropped_tokens": 0,
        "n_repeated_tokens": 0,
    }
    for events, measurements, statics in zip(
        events_per_subject, measurements_per_subject, statics_per_subject, strict=True
    ):
        corpus = measurements + (0 if static_inclusion_mode == "OMIT" else statics)
        prepended = static_inclusion_mode == "PREPEND"
        if batch_mode == "SM":
            # Units are measurements, so the arithmetic is exact.
            length = measurements + (statics if prepended else 0)
            window = min(length, max_seq_len)
            n_windows = _n_windows(length, max_seq_len, step_through_stride)
            trained = n_windows * window
            if static_inclusion_mode == "INCLUDE":
                # Statics ride alongside every window, outside the window's budget.
                static_tokens = n_windows * statics
                trained += static_tokens
            else:
                # Prepended statics sit at the head of the sequence, so only the first window holds them.
                static_tokens = min(statics, window) if prepended else 0
        else:
            # Units are events; a prepended static block costs one event slot however many codes it has.
            length = events + (1 if prepended else 0)
            window = min(length, max_seq_len)
            n_windows = _n_windows(length, max_seq_len, step_through_stride)
            # Only the first window can contain the leading static slot.
            event_units = n_windows * window - (1 if prepended and length else 0)
            # Uniform measurements-per-event density: exact for even subjects, an estimate otherwise.
            dynamic = (event_units * measurements + events // 2) // events if events else 0
            if static_inclusion_mode == "INCLUDE":
                static_tokens = n_windows * statics
            elif prepended:
                static_tokens = statics
            else:
                static_tokens = 0
            trained = dynamic + static_tokens

        projection["n_windows"] += n_windows
        projection["n_corpus_tokens"] += corpus
        projection["n_trained_tokens"] += trained
        projection["n_static_tokens"] += static_tokens
        projection["n_dropped_tokens"] += max(corpus - trained, 0)
        projection["n_repeated_tokens"] += max(trained - corpus, 0)
    return projection
