r"""Build a nested family of MEDS subject subsets that share their shards on disk.

This is the orchestrator: it turns one parent MEDS root plus a :class:`~meds_subsetter.config.SubsetConfig`
into a directory of complete, valid MEDS roots -- one per requested size -- that between them cost about as
much disk as their largest member.

Three decisions from the design carry the whole module, and each one is visible in the code below:

* **Nesting** (D1). Subjects are ordered by :func:`~meds_subsetter.selection.rank_key`, a salted SHA-256
  digest with the subject id appended to make the order total. The ``N``-subject member is the first ``N``
  of that order, so a smaller member's subjects are always a literal prefix of a larger one's.
* **Rank-bucketed sharding** (D2). A subject's output shard is ``rank // n_subjects_per_shard`` -- a
  function of the subject and the salt alone, never of how many subjects were selected. Every *full* shard
  is therefore byte-identical across every member, so it is written once into
  ``<out_root>/.shard_store`` and linked into each member that needs it. Only each member's final, partial
  shard is its own.
* **Fixed evaluation splits** (D3). Only the train split is subsetted. Under the default
  ``non_train_policy=passthrough`` the parent's other splits are shared byte-for-byte across the family,
  which is both what a scaling sweep wants and what makes them shareable at all.

Data movement follows addendum A1/A2. With ``workers == 1`` the shards are produced by a single streaming
``sink_parquet`` through :class:`polars.PartitionBy`: one pass over the parent's train shards, memory-flat,
every missing output shard written at once. With ``workers > 1`` the same work is split into two phases --
fragment each input shard by output shard, then concatenate each output shard's fragments -- with each
output guarded by ``MEDS_transforms.mapreduce.rwlock.rwlock_wrap`` so independent processes cooperate.
Both paths run the same scan, the same sort and the same writer, and produce byte-identical shards; the
doctest on :func:`build_family` checks that against the real fixture.

Every digest and every scan renders against one pinned, dataset-wide column set (:func:`_pinned_schema`),
so a legal three-column MEDS shard cannot digest differently from the same rows in a four-column one, and
no id is a function of how the parent happened to be sharded.

Everything is idempotent. The shard store is a skip-if-exists cache, per-subject digests are cached against
the parent's shard fingerprint, and every write is atomic, so re-running a build is a cheap no-op that
returns an equal manifest.

Examples:
    >>> from meds_subsetter.config import SubsetConfig
    >>> tmp = tempfile.TemporaryDirectory()
    >>> out = Path(tmp.name) / "sweep"
    >>> family = build_family(simple_static_MEDS, out, SubsetConfig(n_subjects=(2, 4)))
    >>> [(m.name, m.n_subjects) for m in family.members]
    [('N0000002', 2), ('N0000004', 4)]
    >>> tmp.cleanup()
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import meds
import polars as pl
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap

from . import __package_name__, __version__
from .config import CodeMetadataPolicy, LinkMode, NonTrainPolicy, SubsetConfig
from .digests import (
    combine_subject_digests,
    digest,
    frame_digest,
    short_id,
    subject_digests,
    subject_set_digest,
)
from .materialize import (
    ShardStore,
    atomic_write_json,
    atomic_write_parquet,
    dir_size,
    place,
    unique_inode_size,
)
from .selection import assign_shards, rank_subjects
from .sizes import _parquet_shard_paths, meds_shard_paths, shard_name, size_meds

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Directory under the output root holding one copy of every distinct shard the family uses.
SHARD_STORE_DIRNAME = ".shard_store"

#: Directory under the output root holding derived state that can always be recomputed.
CACHE_DIRNAME = ".cache"

#: File inside :data:`CACHE_DIRNAME` recording which parent the derived state describes.
CACHE_STAMP_NAME = "parent.json"

#: Directory inside each member root holding that member's provenance manifest.
MEMBER_MANIFEST_DIRNAME = ".meds_subsetter"

#: Name of the family manifest, written at the top of the output root.
FAMILY_MANIFEST_NAME = "family.json"

#: Name of the per-member manifest, inside :data:`MEMBER_MANIFEST_DIRNAME`.
MEMBER_MANIFEST_NAME = "manifest.json"

#: Directory inside each member root holding re-sharded task/index label frames.
TASKS_DIRNAME = "tasks"

#: The extension key added to a member's ``metadata/dataset.json``. Legal: ``DatasetMetadataSchema``
#: declares ``additionalProperties: true`` and has no required fields.
SUBSET_OF_KEY = "subset_of"

#: Store bucket holding the parent's non-train shards, shared verbatim by every member.
PASSTHROUGH_KIND = "passthrough"

#: Store bucket holding the shared ``metadata/codes.parquet`` variants.
METADATA_KIND = "metadata"

#: Parent files outside ``data/`` that a member copies verbatim rather than rewriting, and that
#: therefore have to be fingerprinted alongside the shards: a stale one would otherwise be served out
#: of the shard store forever. ``dataset.json`` and ``subject_splits.parquet`` are re-read on every
#: build, so they are not here.
_FINGERPRINTED_METADATA = (meds.code_metadata_filepath,)

#: Temporary column naming a row's output shard while it is being routed by ``PartitionBy``.
_SHARD_COL = "__shard__"

#: Temporary column naming the input file a row came from.
_PATH_COL = "__shard_path__"

#: Temporary column numbering a row within its input scan, so tie order never depends on the join.
_ROW_COL = "__row__"

#: Seconds a worker waits before re-checking outputs another worker has claimed.
_POLL_SECONDS = 0.05

#: How many times a tree measurement is retried past a file a cooperating worker removed mid-walk.
_MEASURE_ATTEMPTS = 8

#: Rows per parquet row group in every shard this module writes. Pinning it is what makes the
#: single-pass and two-phase data paths byte-identical rather than merely equal: left to its default the
#: streaming writer flushes a row group whenever it runs out of buffered input, and a sink fed one whole
#: split at once runs out at different points from one fed a single output shard's fragments. The value
#: is polars' own default, so the shards are laid out exactly as an ordinary ``write_parquet`` would.
_ROW_GROUP_SIZE = 512 * 512

_SUBJECT_ID = meds.DataSchema.subject_id_name
_TIME = meds.DataSchema.time_name
_CODE = meds.DataSchema.code_name
_SPLIT = meds.SubjectSplitSchema.split_name


class NotAMEDSDatasetError(ValueError):
    """Raised when the directory handed in as a parent is not usable as a MEDS root.

    It is a :class:`ValueError` so a CLI that already renders ``error: {e}`` for bad input needs no extra
    handling.

    Examples:
        >>> raise NotAMEDSDatasetError("/tmp/nope has no data directory")
        Traceback (most recent call last):
            ...
        meds_subsetter.subset.NotAMEDSDatasetError: /tmp/nope has no data directory
        >>> issubclass(NotAMEDSDatasetError, ValueError)
        True
    """


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SubsetManifest:
    """Everything one member of a family is: its subjects, its ids, its shards, and its sizes.

    Attributes:
        name: The member's directory name, from
            :meth:`~meds_subsetter.config.SubsetConfig.subset_name`.
        root: Absolute path of the member's MEDS root.
        n_subjects: The requested train-split size, in subjects. Also the member's position on a scaling
            sweep's x-axis.
        subject_set_id: Digest of the member's subject ids, across every split it carries. Answers "are
            these the same subjects?" without reading a data row.
        train_set_id: The train split's ``content_id``; carried separately because it is the id a scaling
            experiment is actually indexed by.
        content_ids: Split label to that split's reshard-invariant content digest, from
            :func:`~meds_subsetter.digests.combine_subject_digests`.
        dataset_id: Digest over every split's ``content_id`` plus the member's code metadata, i.e. the id
            of the whole emitted dataset.
        codes_id: Digest of the member's ``metadata/codes.parquet``, or of nothing when the parent has
            none.
        observed_code_set_id: Digest of the codes actually observed in the member's data.
        index_id: Digest of the member's slice of the task/index frame, or ``None`` when no ``index_dir``
            was given.
        n_index_rows: Rows in that slice, or ``None``.
        task_name: The task directory the index was written under, or ``None``.
        shards: Member shard name (e.g. ``"train/0"``) to the shard's path inside the store, relative to
            the store root. Two members sharing an entry here share the bytes on disk.
        sizes: :meth:`~meds_subsetter.sizes.SizeReport.to_dict` for the finished member.

    Examples:
        >>> m = SubsetManifest(
        ...     name="N0000010", root="/out/N0000010", n_subjects=10,
        ...     subject_set_id=digest(["1"]), train_set_id=digest(["2"]),
        ...     content_ids={"train": digest(["2"])}, dataset_id=digest(["3"]),
        ...     codes_id=digest([]), observed_code_set_id=digest(["A"]),
        ...     index_id=None, n_index_rows=None, task_name=None,
        ...     shards={"train/0": "train/6b0e7e5d/r0000000000-0000000010.parquet"}, sizes={},
        ... )
        >>> m.name, m.n_subjects, sorted(m.shards)
        ('N0000010', 10, ['train/0'])
        >>> sorted(m.to_dict())
        ['codes_id', 'content_ids', 'dataset_id', 'index_id', 'n_index_rows', 'n_subjects', 'name',
         'observed_code_set_id', 'root', 'shards', 'sizes', 'subject_set_id', 'task_name',
         'train_set_id']

        The manifest is frozen and JSON-native, so it can be written beside the data it describes and
        compared across runs:

        >>> m.n_subjects = 20
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'n_subjects'
        >>> json.loads(json.dumps(m.to_dict())) == m.to_dict()
        True
    """

    name: str
    root: str
    n_subjects: int
    subject_set_id: str
    train_set_id: str
    content_ids: dict[str, str]
    dataset_id: str
    codes_id: str
    observed_code_set_id: str
    index_id: str | None
    n_index_rows: int | None
    task_name: str | None
    shards: dict[str, str]
    sizes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of the manifest.

        Returns:
            A plain dict of every field, with sorted keys.
        """
        return {f.name: getattr(self, f.name) for f in sorted(dataclasses.fields(self), key=lambda f: f.name)}


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FamilyManifest:
    """The record of one whole family: the parent it was cut from, the config, and every member.

    The parent ids are the family's safety belt. A family's shard store is addressed by rank range under
    one salt, so pointing a *different* parent at the same output root would silently hand it the first
    parent's shards; :func:`build_family` refuses to extend a family whose recorded parent
    ``subject_set_id`` or ``fingerprint`` no longer matches -- the first says these are other subjects,
    the second says these are other bytes for the same ones (see :func:`_check_family`).

    Attributes:
        out_root: Absolute path of the family's output root, where ``family.json`` lives.
        parent: Absolute path of the parent MEDS root.
        parent_ids: The parent's ``subject_set_id``, ``train_set_id``, and shard ``fingerprint``.
        config: :meth:`~meds_subsetter.config.SubsetConfig.to_dict` of the config that produced it.
        members: One :class:`SubsetManifest` per member, in ascending ``n_subjects``.
        disk: What the family actually cost, and what it would have cost as independent copies.
        tool: The package that wrote it.
        version: That package's version.

    Examples:
        >>> fam = FamilyManifest(
        ...     out_root="/out", parent="/parent",
        ...     parent_ids={"subject_set_id": digest(["1"])},
        ...     config=SubsetConfig(n_subjects=(10,)).to_dict(), members=(), disk={"n_bytes_on_disk": 0},
        ... )
        >>> fam.tool, fam.members
        ('meds_subsetter', ())
        >>> sorted(fam.to_dict())
        ['config', 'disk', 'members', 'out_root', 'parent', 'tool', 'version']

        The parent's path and its ids are nested together, which is the shape
        :func:`build_family` reads back when it checks whether a family may be extended:

        >>> fam.to_dict()["parent"] == {"path": "/parent", "subject_set_id": digest(["1"])}
        True
        >>> json.loads(json.dumps(fam.to_dict())) == fam.to_dict()
        True
    """

    out_root: str
    parent: str
    parent_ids: dict[str, str]
    config: dict[str, Any]
    members: tuple[SubsetManifest, ...]
    disk: dict[str, int]
    tool: str = __package_name__
    version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of the manifest.

        Returns:
            A plain dict; the parent's path and ids are nested under ``"parent"`` and each member is
            rendered by :meth:`SubsetManifest.to_dict`.
        """
        return {
            "tool": self.tool,
            "version": self.version,
            "out_root": self.out_root,
            "parent": {"path": self.parent, **self.parent_ids},
            "config": self.config,
            "disk": self.disk,
            "members": [m.to_dict() for m in self.members],
        }


def _scan(paths: Sequence[Path], schema: Mapping[str, pl.DataType]) -> pl.LazyFrame:
    """Scan MEDS shards into exactly ``schema``'s columns, in ``schema``'s order.

    Two things make this necessary rather than decorative. A MEDS shard may legally omit
    ``numeric_value``, so a scan of one shard and a scan of the whole dataset can disagree on columns;
    and the single-pass and multi-worker data paths must write the *same bytes*, which they only do if
    they render the same columns in the same order at the same dtypes.

    Args:
        paths: The parquet files to scan, in the order they should be concatenated.
        schema: The dataset-wide pinned column set. Columns it names that the files lack are inserted as
            typed nulls; columns the files carry that it does not name are dropped.

    Returns:
        A lazy frame whose ``collect_schema()`` is ``schema``.

    Examples:
        A narrow shard is widened to the pinned set rather than raising:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> narrow = Path(tmp.name) / "narrow.parquet"
        >>> pl.DataFrame({"subject_id": [1], "code": ["A"]}).write_parquet(narrow)
        >>> pinned = pl.Schema(
        ...     {"subject_id": pl.Int64, "code": pl.String, "numeric_value": pl.Float32}
        ... )
        >>> _scan([narrow], pinned).collect()
        shape: (1, 3)
        ┌────────────┬──────┬───────────────┐
        │ subject_id ┆ code ┆ numeric_value │
        │ ---        ┆ ---  ┆ ---           │
        │ i64        ┆ str  ┆ f32           │
        ╞════════════╪══════╪═══════════════╡
        │ 1          ┆ A    ┆ null          │
        └────────────┴──────┴───────────────┘

        Columns outside the pinned set are dropped, and the pinned order wins over the file's:

        >>> wide = Path(tmp.name) / "wide.parquet"
        >>> pl.DataFrame(
        ...     {"code": ["A"], "extra": [1], "subject_id": [2]}
        ... ).write_parquet(wide)
        >>> _scan([wide], pinned).collect_schema().names()
        ['subject_id', 'code', 'numeric_value']

        The pinned schema is pushed into the *scan*, per file, rather than applied to whatever schema
        the scan resolved on its own. That is the difference between reading a mixed-width dataset and
        emptying half of it: resolving the schema from the first file and then filling in the rest
        would turn every value the later, wider files hold in a column the first one lacks into a null.

        >>> mixed = Path(tmp.name) / "mixed"
        >>> mixed.mkdir()
        >>> pl.DataFrame({"subject_id": [1], "code": ["A"]}).write_parquet(mixed / "0.parquet")
        >>> pl.DataFrame(
        ...     {"subject_id": [2], "code": ["B"], "numeric_value": pl.Series([9.5], dtype=pl.Float32)}
        ... ).write_parquet(mixed / "1.parquet")
        >>> _scan(sorted(mixed.glob("*.parquet")), pinned).collect()["numeric_value"].to_list()
        [None, 9.5]

        A dataset whose shards disagree on a column's *dtype* is refused rather than quietly narrowed,
        since narrowing is exactly the kind of silent edit a subsetter must not make:

        >>> pl.DataFrame(
        ...     {"subject_id": [3], "code": ["C"], "numeric_value": pl.Series([1.0], dtype=pl.Float64)}
        ... ).write_parquet(mixed / "2.parquet")
        >>> _scan(sorted(mixed.glob("*.parquet")), pinned).collect()
        Traceback (most recent call last):
            ...
        polars.exceptions.SchemaError: data type mismatch for column numeric_value: ...
        >>> tmp.cleanup()
    """
    return pl.scan_parquet(
        list(paths),
        schema=pl.Schema(schema),
        missing_columns="insert",
        extra_columns="ignore",
        glob=False,
    )


def _empty_shard(schema: Mapping[str, pl.DataType], path: Path) -> None:
    """Write a row-less but schema-correct parquet file at ``path``.

    A shard whose every subject happens to carry no data rows still has to exist: a MEDS root with a
    missing shard is a different dataset, and a downstream loader that globs would silently see fewer
    subjects. The same helper is used by both data paths, so an empty shard is byte-identical between
    them too.

    Args:
        schema: The pinned column set to write.
        path: Where to write. Parent directories are created.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> path = Path(tmp.name) / "nested" / "0.parquet"
        >>> _empty_shard({"subject_id": pl.Int64, "code": pl.String}, path)
        >>> pl.read_parquet(path)
        shape: (0, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ code │
        │ ---        ┆ ---  │
        │ i64        ┆ str  │
        ╞════════════╪══════╡
        └────────────┴──────┘
        >>> tmp.cleanup()
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(schema=dict(schema)).write_parquet(path)


def _sink_partitioned(
    lf: pl.LazyFrame, base_dir: Path, path_for: Callable[[str], Path], *, include_key: bool
) -> None:
    r"""Sink ``lf`` to one parquet file per distinct value of :data:`_SHARD_COL`.

    This is addendum A1's mechanism, and it is the *only* place this module writes shard bytes -- the
    single-pass path, the multi-worker fragment phase, and the multi-worker reduce phase all funnel
    through it, which is what makes their outputs byte-identical.

    ``approximate_bytes_per_file`` is disabled explicitly. Its default (~4 GiB) would silently split a
    large partition across two files, breaking the MEDS invariant that a subject lives in exactly one
    file; the guard in the provider then makes the impossible case loud rather than silent.

    ``row_group_size`` is pinned for a different reason: see :data:`_ROW_GROUP_SIZE`. Without it the two
    data paths write the same *rows in the same order* but not the same *bytes*, because the streaming
    writer's row-group boundaries follow how the input happened to arrive.

    Args:
        lf: The frame to sink. Must carry :data:`_SHARD_COL`.
        base_dir: Base path handed to :class:`polars.PartitionBy`. Created if absent.
        path_for: Maps a shard key to the file it is written to. Its parent is created.
        include_key: Whether :data:`_SHARD_COL` is kept in the written files.

    Raises:
        RuntimeError: If a partition would spill into a second file.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "out"
        >>> lf = pl.LazyFrame({"subject_id": [1, 2, 1], "__shard__": ["a", "b", "a"]})
        >>> _sink_partitioned(lf, out, lambda key: out / f"{key}.parquet", include_key=False)
        >>> sorted(p.name for p in out.iterdir())
        ['a.parquet', 'b.parquet']
        >>> pl.read_parquet(out / "a.parquet")["subject_id"].to_list()
        [1, 1]

        The key is dropped unless asked for, which is how a routed frame becomes a MEDS shard again:

        >>> pl.read_parquet(out / "a.parquet").columns
        ['subject_id']
        >>> kept = Path(tmp.name) / "kept"
        >>> _sink_partitioned(lf, kept, lambda key: kept / f"{key}.parquet", include_key=True)
        >>> pl.read_parquet(kept / "a.parquet").columns
        ['subject_id', '__shard__']
        >>> tmp.cleanup()
    """

    def provider(args: Any) -> Path:
        key = args.partition_keys.item(0, _SHARD_COL)
        if args.index_in_partition != 0:
            raise RuntimeError(f"Output shard {key!r} spilled into more than one file; refusing to write.")
        out = path_for(key)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    with warnings.catch_warnings():  # `PartitionBy` is flagged unstable and warns on construction.
        warnings.simplefilter("ignore")
        partition = pl.PartitionBy(
            str(base_dir),
            key=_SHARD_COL,
            include_key=include_key,
            file_path_provider=provider,
            approximate_bytes_per_file=None,
        )
    lf.sink_parquet(partition, mkdir=True, engine="streaming", row_group_size=_ROW_GROUP_SIZE)


def _publish(build: Callable[[Path], None], path: Path) -> None:
    """Put a file into place through a staging name that no other process shares.

    Cooperating workers all build every member: they divide the *shard* production between them, but
    each one writes the member metadata, places the links, and ends holding a complete manifest. Every
    one of those outputs is a deterministic function of the parent and the config, so two workers
    writing one at the same moment agree on its bytes. What they must not share is the *staging* path:
    the foundation's atomic writers and :func:`~meds_subsetter.materialize.place` both stage at a fixed
    ``<path>.tmp``, and two processes on one staging file can have the first promote the half-written
    output of the second. Naming the staging file after this process fixes that, and the final
    :func:`os.replace` still makes the result appear whole or not at all.

    ``os.replace`` is a documented no-op when both names are links to one inode, which is exactly the
    case when two workers hardlink the same shard, so the staging file is removed either way.

    Args:
        build: Writes the finished file to the path it is handed.
        path: Where the file belongs. Parent directories are created.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> path = Path(tmp.name) / ".cache" / "value.parquet"
        >>> _publish(lambda p: atomic_write_parquet(pl.DataFrame({"a": [1]}), p), path)
        >>> pl.read_parquet(path)["a"].to_list()
        [1]

        Nothing is left behind, on success or on failure:

        >>> sorted(p.name for p in path.parent.iterdir())
        ['value.parquet']
        >>> _publish(lambda p: (_ for _ in ()).throw(RuntimeError("interrupted")), path)
        Traceback (most recent call last):
            ...
        RuntimeError: interrupted
        >>> sorted(p.name for p in path.parent.iterdir())
        ['value.parquet']
        >>> tmp.cleanup()
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.{os.getpid()}.staging")
    try:
        build(staging)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _fingerprint(root: Path, paths: Sequence[Path]) -> str:
    """Digest what a parent looks like: every copied file's relative path, size, and modification time.

    This is what keys the derived caches under ``<out_root>/.cache`` and what binds a family to its
    parent in :func:`_check_family`, so its one job is to *move whenever the parent does*. Size alone
    does not do that: a rewrite that keeps a shard's size -- an ETL bug fixed, a float corrected, a
    column re-encoded -- is exactly the routine edit a re-run makes, and it would leave every cached
    digest and every stored shard describing bytes that are no longer there. The modification time
    catches it for two more ``stat`` fields and still no bytes read, which is what keeps an unchanged
    rebuild a metadata-only no-op instead of a full pass over the dataset.

    It covers the shards *and* the parent metadata a member copies rather than rewrites -- which is
    ``metadata/codes.parquet`` alone; ``dataset.json`` and ``subject_splits.parquet`` are re-read on
    every build and so cannot go stale in the store.

    The trade is deliberate and one-directional: a copy that resets modification times reads as a
    changed parent (loud, and ``do_overwrite`` clears it), while a rewrite that forges the old
    ``(size, mtime)`` reads as an unchanged one. Only the first failure mode is silent, and this is a
    change detector, not a proof of identity.

    Args:
        root: The MEDS root the paths are relative to.
        paths: The shard files, in sorted order.

    Returns:
        A ``"sha256:<hex>"`` digest.

    Examples:
        >>> from meds_subsetter.sizes import meds_shard_paths
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name) / "parent"
        >>> _ = shutil.copytree(simple_static_MEDS, root)
        >>> paths = meds_shard_paths(root)
        >>> fp = _fingerprint(root, paths)
        >>> fp == _fingerprint(root, paths)
        True
        >>> _fingerprint(root, paths[:2]) == fp
        False

        A rewrite that leaves a shard's *size* alone still moves it:

        >>> raw = bytearray(paths[0].read_bytes())
        >>> raw[60] ^= 0xFF
        >>> _ = paths[0].write_bytes(bytes(raw))
        >>> paths[0].stat().st_size == len(raw), _fingerprint(root, paths) == fp
        (True, False)

        So does a correction to the code metadata the members copy, which lives outside ``data/``:

        >>> fresh = Path(tmp.name) / "codes"
        >>> _ = shutil.copytree(simple_static_MEDS, fresh)
        >>> paths = meds_shard_paths(fresh)
        >>> before = _fingerprint(fresh, paths)
        >>> codes = fresh / "metadata" / "codes.parquet"
        >>> pl.read_parquet(codes).with_columns(
        ...     pl.lit("corrected").alias("description")
        ... ).write_parquet(codes)
        >>> _fingerprint(fresh, paths) == before
        False
        >>> tmp.cleanup()
    """
    files = [*paths, *(root / rel for rel in _FINGERPRINTED_METADATA if (root / rel).is_file())]
    items = []
    for path in files:
        stat = path.stat()
        items.append(f"{path.relative_to(root).as_posix()}\x1f{stat.st_size}\x1f{stat.st_mtime_ns}")
    return digest(items)


@dataclasses.dataclass(slots=True, eq=False)
class _Parent:
    """The parent dataset, resolved once and shared by every member of a family.

    Attributes:
        root: The parent MEDS root.
        shard_paths: Every data shard, sorted, dotfiles excluded.
        schema: The dataset-wide pinned column set. Every digest and every scan renders against it, so a
            legal three-column shard cannot digest differently from the same rows in a four-column one.
        splits: Split label to that split's subject ids, sorted.
        fingerprint: :func:`_fingerprint` of the shards, keying the derived caches.
        subject_set_id: Digest of every subject the parent assigns to a split.
        cache_dir: ``<out_root>/.cache``, where the derived per-subject digests live.
        _subject_shards: Memoized ``subject_id -> shard path`` frame; see :meth:`subject_shards`.
    """

    root: Path
    shard_paths: tuple[Path, ...]
    schema: pl.Schema
    splits: dict[str, tuple[int, ...]]
    fingerprint: str
    subject_set_id: str
    cache_dir: Path
    _subject_shards: pl.DataFrame | None = None

    def subject_shards(self) -> pl.DataFrame:
        """Return the ``subject_id -> shard path`` map, reading or filling the on-disk cache.

        MEDS guarantees a subject lives in exactly one data file, which is what lets the shard producer
        scan only the input files that actually feed the outputs it is missing, and what lets a
        passed-through shard be checked for train subjects before it is shared. Building the map costs
        one pass over the ``subject_id`` column of every shard, so it is cached against
        :attr:`fingerprint` and never rebuilt for an unchanged parent.

        Returns:
            A two-column frame of ``subject_id`` and :data:`_PATH_COL`, sorted.
        """
        if self._subject_shards is None:
            cached = self.cache_dir / "subject_shards.parquet"
            if cached.is_file():
                self._subject_shards = pl.read_parquet(cached, glob=False)
            else:
                frame = (
                    pl.scan_parquet(
                        list(self.shard_paths),
                        schema=pl.Schema(self.schema),
                        missing_columns="insert",
                        extra_columns="ignore",
                        include_file_paths=_PATH_COL,
                        glob=False,
                    )
                    .select(_SUBJECT_ID, _PATH_COL)
                    .unique()
                    .sort(_PATH_COL, _SUBJECT_ID)
                    .collect()
                )
                _publish(lambda p: atomic_write_parquet(frame, p), cached)
                self._subject_shards = frame
        return self._subject_shards

    def forget(self) -> None:
        """Drop the memoized subject-to-shard map, so the next caller rebuilds it from the shards."""
        self._subject_shards = None

    def shards_for(self, subject_ids: Iterable[int]) -> list[Path]:
        """Return the shard files holding any of ``subject_ids``, sorted.

        Args:
            subject_ids: The subjects whose input files are wanted.

        Returns:
            The subset of :attr:`shard_paths` that holds them.
        """
        wanted = pl.Series(_SUBJECT_ID, sorted(set(subject_ids)), dtype=pl.Int64)
        held = set(self.subject_shards().filter(pl.col(_SUBJECT_ID).is_in(wanted.implode()))[_PATH_COL])
        return [p for p in self.shard_paths if str(p) in held]

    def subjects_in(self, path: Path) -> list[int]:
        """Return the subjects held by one shard file, sorted.

        Args:
            path: A shard file of this parent.

        Returns:
            Its subject ids.
        """
        rows = self.subject_shards().filter(pl.col(_PATH_COL) == str(path))
        return sorted(rows[_SUBJECT_ID].to_list())


def _pinned_schema(paths: Sequence[Path]) -> pl.Schema:
    r"""Return the *union* of the column sets of a dataset's shards.

    This is the schema every digest and every scan in the module renders against, so it has to be the
    dataset's columns and not one shard's. Reading it off a multi-file
    ``scan_parquet(..., missing_columns="insert", extra_columns="ignore")`` -- the obvious way, and the
    one the sibling modules' docstrings suggest -- does **not** give that: polars resolves the schema
    from the *first* file and then inserts or ignores to match it. MEDS makes ``numeric_value``
    optional, so a dataset whose first shard by sorted path happens to be three columns wide would have
    every numeric value silently dropped out of both its subsets and its ids. The union is therefore
    built from each shard's own footer, first occurrence fixing a column's position and dtype.

    Args:
        paths: Every shard of the dataset, sorted. Only parquet footers are read.

    Returns:
        The union schema, in first-seen order.

    Examples:
        >>> from meds_subsetter.sizes import meds_shard_paths
        >>> _pinned_schema(meds_shard_paths(simple_static_MEDS))
        Schema({'subject_id': Int64, 'time': Datetime(time_unit='us', time_zone=None),
                'code': String, 'numeric_value': Float32})

        The hazard, exhibited. A three-column shard is legal MEDS, and sorting puts it first here:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> pl.DataFrame({"subject_id": [2], "code": ["B"]}).write_parquet(root / "a.parquet")
        >>> pl.DataFrame(
        ...     {"subject_id": [1], "code": ["A"], "numeric_value": pl.Series([1.0], dtype=pl.Float32)}
        ... ).write_parquet(root / "b.parquet")
        >>> paths = sorted(root.glob("*.parquet"))
        >>> pl.scan_parquet(paths, missing_columns="insert", extra_columns="ignore").collect_schema()
        Schema({'subject_id': Int64, 'code': String})
        >>> _pinned_schema(paths)
        Schema({'subject_id': Int64, 'code': String, 'numeric_value': Float32})
        >>> tmp.cleanup()
    """
    merged: dict[str, pl.DataType] = {}
    for path in paths:
        for name, dtype in pl.scan_parquet(path, glob=False).collect_schema().items():
            merged.setdefault(name, dtype)
    return pl.Schema(merged)


def _splits_from_file(path: Path) -> dict[str, tuple[int, ...]]:
    """Read ``metadata/subject_splits.parquet`` into a split label to subject ids mapping.

    Args:
        path: The splits file.

    Returns:
        Split label to sorted subject ids.

    Examples:
        >>> splits = _splits_from_file(simple_static_MEDS / "metadata" / "subject_splits.parquet")
        >>> {k: len(v) for k, v in sorted(splits.items())}
        {'held_out': 1, 'train': 4, 'tuning': 1}
        >>> splits["train"]
        (68729, 239684, 814703, 1195293)
    """
    df = pl.read_parquet(path, glob=False).select(
        pl.col(_SUBJECT_ID).cast(pl.Int64), pl.col(_SPLIT).cast(pl.String)
    )
    grouped = df.group_by(_SPLIT).agg(pl.col(_SUBJECT_ID).unique().sort()).sort(_SPLIT)
    return {row[0]: tuple(row[1]) for row in grouped.iter_rows()}


def _splits_from_shard_names(parent: _Parent) -> dict[str, tuple[int, ...]]:
    """Derive splits from shard-name prefixes, for a parent with no splits file.

    Args:
        parent: The partially-built parent; only :attr:`_Parent.root` and
            :attr:`_Parent.shard_paths` are read.

    Returns:
        Split label to sorted subject ids.

    Raises:
        NotAMEDSDatasetError: If any shard name is flat, i.e. carries no split prefix. Subsetting "the
            train split" is meaningless without an assignment, and guessing one would silently build a
            family against the wrong subjects.
    """
    names = {path: shard_name(parent.root, path) for path in parent.shard_paths}
    flat = sorted(name for name in names.values() if "/" not in name)
    if flat:
        raise NotAMEDSDatasetError(
            f"{parent.root} has no {meds.subject_splits_filepath} and its shard names carry no split "
            f"prefix (e.g. {flat[0]!r}), so its splits cannot be determined; a subset of 'the train "
            f"split' is undefined without one."
        )
    out: dict[str, list[int]] = {}
    for path, name in names.items():
        out.setdefault(name.split("/", 1)[0], []).extend(parent.subjects_in(path))
    return {split: tuple(sorted(set(ids))) for split, ids in sorted(out.items())}


def _write_stamp(root: Path, fingerprint: str, cache_dir: Path) -> None:
    """Record which parent the derived caches under ``cache_dir`` describe.

    The stamp is written *before* any cache entry, so a cache directory that has no stamp has nothing
    derived in it either -- which is what lets :func:`_open_parent` distinguish "not built yet" from
    "built for a different parent" without clearing a peer's fresh work.

    Args:
        root: The parent MEDS root.
        fingerprint: :func:`_fingerprint` of its shards.
        cache_dir: ``<out_root>/.cache``.
    """
    _publish(
        lambda staged: atomic_write_json({"path": str(root), "fingerprint": fingerprint}, staged),
        cache_dir / CACHE_STAMP_NAME,
    )


def _open_parent(root: Path, cache_dir: Path) -> _Parent:
    """Validate a parent MEDS root and resolve everything a family needs from it.

    Args:
        root: The candidate MEDS root.
        cache_dir: ``<out_root>/.cache``. Reset whenever the parent's fingerprint has moved, so a changed
            parent can never be described by a stale cached digest.

    Returns:
        The resolved parent.

    Raises:
        NotAMEDSDatasetError: If ``root`` has no ``data/`` directory, if that directory holds no shards,
            or if the dataset's splits cannot be determined.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> cache = Path(tmp.name) / ".cache"
        >>> parent = _open_parent(simple_static_MEDS, cache)
        >>> sorted(parent.splits)
        ['held_out', 'train', 'tuning']
        >>> parent.schema.names()
        ['subject_id', 'time', 'code', 'numeric_value']

        The fingerprint stamp is written beside the caches it keys:

        >>> json.loads((cache / "parent.json").read_text())["fingerprint"] == parent.fingerprint
        True

        A directory that is not a MEDS root says exactly what is missing:

        >>> _open_parent(Path(tmp.name) / "nope", cache)
        Traceback (most recent call last):
            ...
        meds_subsetter.subset.NotAMEDSDatasetError: ...nope is not a MEDS dataset: no data directory
        at .../nope/data
        >>> with yaml_disk({"data/.logs/0.parquet": None}) as empty:
        ...     _open_parent(empty, cache)
        Traceback (most recent call last):
            ...
        meds_subsetter.subset.NotAMEDSDatasetError: ... is not a MEDS dataset: .../data holds no
        parquet shards

        A root whose splits cannot be determined is refused rather than guessed at:

        >>> flat = Path(tmp.name) / "flat"
        >>> (flat / "data").mkdir(parents=True)
        >>> pl.DataFrame({"subject_id": [1], "code": ["A"]}).write_parquet(flat / "data" / "0.parquet")
        >>> _open_parent(flat, cache)
        Traceback (most recent call last):
            ...
        meds_subsetter.subset.NotAMEDSDatasetError: ...flat has no metadata/subject_splits.parquet and
        its shard names carry no split prefix (e.g. '0'), so its splits cannot be determined; a subset
        of 'the train split' is undefined without one.
        >>> tmp.cleanup()
    """
    data_dir = root / meds.data_subdirectory
    if not data_dir.is_dir():
        raise NotAMEDSDatasetError(f"{root} is not a MEDS dataset: no data directory at {data_dir}")
    shard_paths = meds_shard_paths(root)
    if not shard_paths:
        raise NotAMEDSDatasetError(f"{root} is not a MEDS dataset: {data_dir} holds no parquet shards")

    fingerprint = _fingerprint(root, shard_paths)
    stamp = cache_dir / CACHE_STAMP_NAME
    recorded = json.loads(stamp.read_text(encoding="utf-8")).get("fingerprint") if stamp.is_file() else None
    if recorded != fingerprint:
        # Only an actual mismatch clears the cache. An *absent* stamp means nothing derived has been
        # cached yet -- the stamp is written before any cache entry is -- so there is nothing stale to
        # remove, and clearing anyway would let one starting worker delete a peer's fresh entries.
        if recorded is not None:
            shutil.rmtree(cache_dir, ignore_errors=True)
        _write_stamp(root, fingerprint, cache_dir)

    parent = _Parent(
        root=root,
        shard_paths=tuple(shard_paths),
        schema=_pinned_schema(shard_paths),
        splits={},
        fingerprint=fingerprint,
        subject_set_id="",
        cache_dir=cache_dir,
    )
    splits_path = root / meds.subject_splits_filepath
    parent.splits = (
        _splits_from_file(splits_path) if splits_path.is_file() else _splits_from_shard_names(parent)
    )
    parent.subject_set_id = subject_set_digest(s for ids in parent.splits.values() for s in ids)
    return parent


def _cached_subject_digests(parent: _Parent, split: str) -> pl.DataFrame:
    """Return the parent's per-subject digests for one split, computing them at most once.

    Per-subject digests are reshard-invariant, so every member's ``content_id`` for a split is a cheap
    :func:`~meds_subsetter.digests.combine_digests` over a *row subset* of this frame -- no member ever
    re-reads the data. The cache is keyed by :attr:`_Parent.fingerprint`, so a changed parent invalidates
    it (see :func:`_open_parent`).

    Args:
        parent: The resolved parent.
        split: The split to digest.

    Returns:
        A frame of ``subject_id`` and ``digest``, sorted by ``subject_id``, covering exactly the split's
        subjects that have data rows.
    """
    cached = parent.cache_dir / "subject_digests" / f"{split}.parquet"
    if cached.is_file():
        return pl.read_parquet(cached, glob=False)

    ids = pl.Series(_SUBJECT_ID, parent.splits.get(split, ()), dtype=pl.Int64)
    per_shard = [
        subject_digests(pl.scan_parquet(path, glob=False), schema=parent.schema)
        for path in parent.shards_for(parent.splits.get(split, ()))
    ]
    empty = pl.DataFrame(schema={_SUBJECT_ID: pl.Int64, "digest": pl.String})
    frame = (
        pl.concat([empty, *per_shard])
        .filter(pl.col(_SUBJECT_ID).is_in(ids.implode()))
        .unique()
        .sort(_SUBJECT_ID)
    )
    _publish(lambda p: atomic_write_parquet(frame, p), cached)
    return frame


@dataclasses.dataclass(frozen=True, slots=True)
class _Placement:
    """One shard of one member: where it lands, and where in the store it comes from.

    Attributes:
        shard: The member-relative MEDS shard name, e.g. ``"train/0"``.
        kind: The store bucket (a split label for computed shards, :data:`PASSTHROUGH_KIND` otherwise).
        key: The store key: a rank range for computed shards, the parent's shard name otherwise.
        subdir: The store subdirectory: the salt's short id for computed shards, empty otherwise.
        subject_ids: The subjects the shard holds. Empty for a passed-through shard, whose subjects are
            whatever the parent put there; :func:`_shard_subjects` resolves those on demand.
        source: The parent shard a passed-through shard is copied from, or ``None`` for a computed one.
    """

    shard: str
    kind: str
    key: str
    subdir: str
    subject_ids: tuple[int, ...] = ()
    source: Path | None = None

    @property
    def stored_name(self) -> str:
        """The shard's path inside the store, relative to the store root, as recorded in a manifest.

        Returns:
            A POSIX path such as ``"train/6b0e7e5d/r0000000000-0000000002.parquet"``.
        """
        return "/".join(part for part in (self.kind, self.subdir, f"{self.key}.parquet") if part)


def _salt_id(cfg: SubsetConfig) -> str:
    """Return the short id of a config's salt, used as the store subdirectory for computed shards.

    Args:
        cfg: The family config.

    Returns:
        Sixteen hex characters' worth of ``sha256(salt)``, truncated to eight.

    Examples:
        >>> _salt_id(SubsetConfig(n_subjects=(10,)))
        '6b0e7e5d'
        >>> _salt_id(SubsetConfig(n_subjects=(10,), salt="other")) != _salt_id(
        ...     SubsetConfig(n_subjects=(10,))
        ... )
        True
    """
    return short_id(digest([cfg.salt]), 8)


def _non_train_splits(parent: _Parent, cfg: SubsetConfig) -> list[str]:
    """Return the parent's splits other than the one being subsetted, sorted.

    Args:
        parent: The resolved parent.
        cfg: The family config.

    Returns:
        The other split labels.
    """
    return sorted(split for split in parent.splits if split != cfg.train_split)


def _passthrough_shards(parent: _Parent, cfg: SubsetConfig) -> list[Path]:
    """Return the parent shards that can be shared verbatim: those holding no train subject.

    Args:
        parent: The resolved parent.
        cfg: The family config.

    Returns:
        The shareable shard paths, sorted.

    Raises:
        ValueError: If a shard mixes train and non-train subjects. Such a shard cannot be passed through
            verbatim without leaking the parent's full train split into every member, and dropping it
            would lose real evaluation subjects, so the caller is told to rewrite the non-train splits
            instead.
    """
    train = set(parent.splits.get(cfg.train_split, ()))
    non_train = {s for split, ids in parent.splits.items() if split != cfg.train_split for s in ids}
    out = []
    for path in parent.shard_paths:
        subjects = set(parent.subjects_in(path))
        if not subjects & non_train:
            continue
        if subjects & train:
            raise ValueError(
                f"Shard {shard_name(parent.root, path)!r} of {parent.root} holds both "
                f"{cfg.train_split!r} and non-{cfg.train_split} subjects, so it cannot be passed "
                f"through verbatim; rebuild with non_train_policy='subset' (or 'drop')."
            )
        out.append(path)
    return out


def _member_splits(
    parent: _Parent, cfg: SubsetConfig, n: int, ranked: Mapping[str, Sequence[int]]
) -> dict[str, tuple[int, ...]]:
    """Return the subjects each split of the ``n``-subject member carries.

    Args:
        parent: The resolved parent.
        cfg: The family config.
        n: The member's train size.
        ranked: Split label to that split's rank order. Must cover the train split, and every non-train
            split under :attr:`~meds_subsetter.config.NonTrainPolicy.SUBSET`.

    Returns:
        Split label to that split's subject ids, in *selection* (rank) order for a split this member
        selects from, and in the parent's own order for one it passes through. Everything downstream
        either shards by rank or sorts, so the distinction only matters to :func:`_placements`.

    Raises:
        ValueError: If ``n`` exceeds the train pool.
    """
    pool = ranked[cfg.train_split]
    if n > len(pool):
        raise ValueError(
            f"Cannot build a {n}-subject subset: the parent's {cfg.train_split!r} split has only "
            f"{len(pool)} subjects."
        )
    out = {cfg.train_split: tuple(pool[:n])}
    match cfg.non_train_policy:
        case NonTrainPolicy.DROP:
            pass
        case NonTrainPolicy.PASSTHROUGH:
            for split in _non_train_splits(parent, cfg):
                out[split] = parent.splits[split]
        case NonTrainPolicy.SUBSET:
            # A split smaller than `n` is taken whole rather than refused: the family still shrinks
            # monotonically with `N` and still nests, which is what SUBSET is for.
            for split in _non_train_splits(parent, cfg):
                out[split] = tuple(ranked[split][: min(n, len(ranked[split]))])
    return {split: out[split] for split in sorted(out)}


def _placements(
    parent: _Parent, cfg: SubsetConfig, n: int, ranked: Mapping[str, Sequence[int]]
) -> list[_Placement]:
    """Plan every shard of the ``n``-subject member: where it lands and what produces it.

    Args:
        parent: The resolved parent.
        cfg: The family config.
        n: The member's train size.
        ranked: Split label to that split's rank order, as for :func:`_member_splits`.

    Returns:
        One :class:`_Placement` per member shard, ordered by where it lives in the store. Shard keys are
        zero-padded rank ranges, so that order is rank order -- unlike ordering by member shard name,
        under which ``train/10`` would sort between ``train/1`` and ``train/2``.

    Raises:
        ValueError: If a computed shard and a passed-through parent shard claim the same member shard
            name. The ``train/`` prefix is convention rather than spec -- split membership is
            ``subject_splits.parquet``'s to declare -- so a parent may legally hold a *tuning* shard at
            ``data/train/2``, and the rank buckets are named ``train/0, train/1, ...`` whatever the
            parent's layout. Two placements at one path would silently overwrite one another and lose a
            whole evaluation split.

    Examples:
        A parent whose tuning shard is named ``train/2``: legal MEDS, and what a re-split done at shard
        granularity produces. While the member's rank buckets do not reach that name, it passes
        through beside them:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name) / "parent"
        >>> _ = shutil.copytree(simple_static_MEDS, root)
        >>> _ = shutil.move(root / "data" / "tuning" / "0.parquet", root / "data" / "train" / "2.parquet")
        >>> parent = _open_parent(root, Path(tmp.name) / ".cache")
        >>> cfg = SubsetConfig(n_subjects=(2, 4), n_subjects_per_shard=1)
        >>> ranked = _rank_orders(parent, cfg)
        >>> [p.shard for p in _placements(parent, cfg, 2, ranked)]
        ['held_out/0', 'train/2', 'train/0', 'train/1']

        At four subjects the fourth rank bucket *is* ``train/2``, and the collision is refused rather
        than resolved by whichever placement is written last:

        >>> _placements(parent, cfg, 4, ranked)
        Traceback (most recent call last):
            ...
        ValueError: Member shard 'train/2' is claimed twice: by the rank bucket
        'r0000000002-0000000003' of the 'train' split and by the parent shard .../data/train/2.parquet,
        which holds no 'train' subjects. Rebuild with non_train_policy='subset' (or 'drop').
        >>> tmp.cleanup()
    """
    subdir = _salt_id(cfg)
    out: list[_Placement] = []
    for split, ids in _member_splits(parent, cfg, n, ranked).items():
        if cfg.non_train_policy is NonTrainPolicy.PASSTHROUGH and split != cfg.train_split:
            continue
        for spec in assign_shards(ids, cfg.n_subjects_per_shard):
            out.append(
                _Placement(
                    shard=f"{split}/{spec.index}",
                    kind=split,
                    key=spec.key,
                    subdir=subdir,
                    subject_ids=spec.subject_ids,
                )
            )
    computed = {placement.shard: placement for placement in out}
    if cfg.non_train_policy is NonTrainPolicy.PASSTHROUGH:
        for path in _passthrough_shards(parent, cfg):
            name = shard_name(parent.root, path)
            clash = computed.get(name)
            if clash is not None:
                raise ValueError(
                    f"Member shard {name!r} is claimed twice: by the rank bucket {clash.key!r} of the "
                    f"{clash.kind!r} split and by the parent shard {path}, which holds no "
                    f"{cfg.train_split!r} subjects. Rebuild with non_train_policy='subset' (or 'drop')."
                )
            out.append(_Placement(shard=name, kind=PASSTHROUGH_KIND, key=name, subdir="", source=path))
    return sorted(out, key=lambda p: p.stored_name)


def _shard_subjects(parent: _Parent, placement: _Placement) -> list[int]:
    """Return the subjects one placed shard holds.

    Args:
        parent: The resolved parent.
        placement: The placement to resolve.

    Returns:
        Sorted subject ids: the placement's own for a computed shard, the parent shard's for a passed
        through one.
    """
    if placement.source is None:
        return sorted(placement.subject_ids)
    return parent.subjects_in(placement.source)


def _is_placed(stored: Path, dst: Path, mode: LinkMode) -> bool:
    """Return whether ``dst`` already *is* the placement of ``stored`` under ``mode``.

    Rebuilding a family re-places every shard, and re-placing something already in place is not quite
    free: ``place`` stages the new entry at ``<dst>.tmp`` and promotes it with :func:`os.replace`, and
    POSIX makes that rename a **no-op when both names are links to one inode** -- so under
    ``hardlink`` the staging file is not consumed and a ``.parquet.tmp`` accumulates beside every
    shard. Checking first keeps a rebuild a true no-op, and keeps the member roots clean.

    Args:
        stored: The store entry.
        dst: Where the member expects it.
        mode: The family's link mode.

    Returns:
        ``True`` when placing would not change anything. Under ``copy`` that is judged by size and
        mtime, which :func:`shutil.copy2` preserves; a false negative only costs one redundant copy.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> stored = root / "store" / "r0.parquet"
        >>> stored.parent.mkdir(parents=True)
        >>> _ = stored.write_text("shard")
        >>> dst = root / "member" / "0.parquet"
        >>> [_is_placed(stored, dst, m) for m in LinkMode]
        [False, False, False]

        Every mode recognizes its own placement, and only its own:

        >>> for mode in LinkMode:
        ...     place(stored, dst, mode)
        ...     print(mode.value, [_is_placed(stored, dst, m) for m in LinkMode])
        symlink [True, False, False]
        hardlink [False, True, True]
        copy [False, False, True]

        A placement of something *else* is not a placement of this:

        >>> other = root / "store" / "r1.parquet"
        >>> _ = other.write_text("different")
        >>> place(other, dst, "copy")
        >>> _is_placed(stored, dst, LinkMode.COPY)
        False
        >>> tmp.cleanup()
    """
    try:
        if mode is LinkMode.SYMLINK:
            return dst.is_symlink() and dst.resolve() == stored.resolve()
        if dst.is_symlink() or not dst.is_file():
            return False
        here, there = dst.stat(), stored.stat()
        if mode is LinkMode.HARDLINK:
            return (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino)
        return (here.st_size, here.st_mtime_ns) == (there.st_size, there.st_mtime_ns)
    except OSError:  # a broken link, or a store entry that has gone away
        return False


def _link_into(store: ShardStore, stored: Path, dst: Path, mode: LinkMode) -> None:
    """Place a stored shard into a member root, unless it is already placed there.

    Args:
        store: The family's shard store.
        stored: The store entry.
        dst: Where the member expects it.
        mode: The family's link mode.
    """
    if not _is_placed(stored, dst, mode):
        _publish(lambda staged: store.link_into(stored, staged, mode), dst)


def _prune_shards(root: Path, keep: Iterable[str]) -> list[str]:
    """Remove the parquet files under ``root`` that this build does not name, and return what went.

    Placing shards is not enough to make a member root correct: the plan can *shrink* between two runs
    -- a larger ``n_subjects_per_shard``, a ``non_train_policy`` that no longer passes the evaluation
    splits through -- and the previous run's shards would then sit beside the new ones, putting a
    subject in two files at once. That breaks the single MEDS invariant this whole module leans on, and
    it breaks it silently: every count downstream doubles and nothing says why. So the tree is
    reconciled with the plan rather than merely written into.

    Only ``*.parquet`` is touched, and never inside a dot-directory: ``.logs/`` and friends legitimately
    live under ``data/``, and they are none of this function's business.

    Args:
        root: The tree to reconcile, e.g. ``<member>/data``. Need not exist.
        keep: The shard names this build placed, e.g. ``{"train/0", "tuning/0"}``.

    Returns:
        The names of the shards removed, sorted.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> data = Path(tmp.name) / "data"
        >>> for name in ("train/0", "train/1", "tuning/0"):
        ...     path = data / f"{name}.parquet"
        ...     path.parent.mkdir(parents=True, exist_ok=True)
        ...     pl.DataFrame({"subject_id": [1]}).write_parquet(path)
        >>> (data / ".logs").mkdir()
        >>> pl.DataFrame({"subject_id": [1]}).write_parquet(data / ".logs" / "0.parquet")
        >>> _prune_shards(data, {"train/0"})
        ['train/1', 'tuning/0']

        The dot-directory is left exactly as it was, and the directory the removal emptied is gone, so
        a member that no longer carries a split does not keep an empty one:

        >>> sorted(p.name for p in data.iterdir())
        ['.logs', 'train']
        >>> sorted(p.relative_to(data).as_posix() for p in data.rglob("*.parquet"))
        ['.logs/0.parquet', 'train/0.parquet']

        A dangling link -- a shard whose store entry has been discarded -- is removed like any other,
        and reconciling an already-reconciled tree does nothing:

        >>> (data / "train" / "9.parquet").symlink_to(Path(tmp.name) / "gone.parquet")
        >>> _prune_shards(data, {"train/0"})
        ['train/9']
        >>> _prune_shards(data, {"train/0"})
        []
        >>> _prune_shards(Path(tmp.name) / "absent", {"train/0"})
        []
        >>> tmp.cleanup()
    """
    if not root.is_dir():
        return []
    wanted = set(keep)
    removed = []
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_symlink() and not path.is_file():  # a directory that happens to end in .parquet
            continue
        name = relative.with_suffix("").as_posix()
        if name in wanted:
            continue
        path.unlink(missing_ok=True)
        removed.append(name)
    directories = [p for p in root.rglob("*") if p.is_dir() and not p.is_symlink()]
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        if any(part.startswith(".") for part in directory.relative_to(root).parts):
            continue
        # Not empty (the common case), or a cooperating worker got there first.
        with contextlib.suppress(OSError):
            directory.rmdir()
    return sorted(removed)


def _assignment(specs: Mapping[str, Sequence[int]]) -> pl.LazyFrame:
    """Build the ``subject_id -> output shard`` frame the data movement joins against.

    The mapping is deliberately one-to-*many*: with a large ``n_subjects_per_shard`` every member has its
    own partial shard, so one subject legitimately belongs to several output shards at once and its rows
    are duplicated into each by the join.

    Args:
        specs: Output shard key to the subjects it holds.

    Returns:
        A lazy two-column frame of ``subject_id`` and :data:`_SHARD_COL`, sorted.

    Examples:
        >>> _assignment({"r0-4": [7, 3], "r0-2": [3]}).collect()
        shape: (3, 2)
        ┌────────────┬───────────┐
        │ subject_id ┆ __shard__ │
        │ ---        ┆ ---       │
        │ i64        ┆ str       │
        ╞════════════╪═══════════╡
        │ 3          ┆ r0-2      │
        │ 3          ┆ r0-4      │
        │ 7          ┆ r0-4      │
        └────────────┴───────────┘
    """
    rows = sorted((subject, key) for key, ids in specs.items() for subject in ids)
    return pl.DataFrame(
        {_SUBJECT_ID: [r[0] for r in rows], _SHARD_COL: [r[1] for r in rows]},
        schema={_SUBJECT_ID: pl.Int64, _SHARD_COL: pl.String},
    ).lazy()


def _indexed(parent: _Parent, paths: Sequence[Path]) -> pl.LazyFrame:
    """Scan MEDS shards and number their rows, so that tie order is fixed by the *files*.

    MEDS routinely carries several measurements at one ``(subject_id, time)``, so a sort on those two
    columns alone leaves ties to be broken by whatever order the preceding join emitted -- and a hash
    join promises no order at all. Numbering the rows *before* the join adds a tiebreaker that is a
    function of the input files, which is exactly what makes the single-pass and two-phase paths write
    the same bytes: a subject lives in one input file, so its rows carry increasing indices in both.

    Args:
        parent: The resolved parent, for its pinned schema.
        paths: The input shards to read, in the order they are concatenated.

    Returns:
        A lazy frame carrying the pinned columns plus :data:`_ROW_COL`.
    """
    return _scan(paths, parent.schema).with_row_index(_ROW_COL)


def _routed(parent: _Parent, specs: Mapping[str, Sequence[int]], paths: Sequence[Path]) -> pl.LazyFrame:
    """Scan ``paths``, keep only the rows the output shards want, and tag each with its shard.

    Args:
        parent: The resolved parent, for its pinned schema.
        specs: Output shard key to the subjects it holds.
        paths: The input shards to read.

    Returns:
        A lazy frame carrying the pinned columns plus :data:`_ROW_COL` and :data:`_SHARD_COL`, unsorted.
    """
    return _indexed(parent, paths).join(_assignment(specs), on=_SUBJECT_ID, how="inner")


def _sorted_for_output(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Put a routed frame into MEDS order: by subject, then by time, ties in input-file order.

    The sort key ``(subject_id, time, __row__)`` is total -- a row index is unique within its input
    file and a subject lives in exactly one -- so the result does not depend on the sort's stability or
    on the engine's chunking. ``maintain_order=True`` is kept anyway, so that a parent which violates
    the one-subject-one-file invariant degrades to "reproducible" rather than to "arbitrary".

    Args:
        lf: A routed frame, from :func:`_routed` or read back from fragments.

    Returns:
        The sorted frame, with :data:`_ROW_COL` dropped.
    """
    return lf.sort(_SUBJECT_ID, _TIME, _ROW_COL, maintain_order=True).drop(_ROW_COL)


def _produce_single_pass(
    parent: _Parent,
    store: ShardStore,
    kind: str,
    subdir: str,
    specs: Mapping[str, Sequence[int]],
    scratch: Path,
) -> None:
    """Write every missing shard of one split in one streaming pass (addendum A1).

    Args:
        parent: The resolved parent.
        store: The family's shard store.
        kind: The store bucket, i.e. the split label.
        subdir: The store subdirectory, i.e. the salt's short id.
        specs: Output shard key to the subjects it holds.
        scratch: A directory the sink may write into. Removed afterwards.
    """
    missing = {key: ids for key, ids in sorted(specs.items()) if not store.has(kind, key, subdir=subdir)}
    if not missing:
        return
    inputs = parent.shards_for(s for ids in missing.values() for s in ids)
    staging = scratch / f"staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if inputs:
            lf = _sorted_for_output(_routed(parent, missing, inputs))
            _sink_partitioned(lf, staging, lambda key: staging / f"{key}.parquet", include_key=False)
        for key in missing:
            produced = staging / f"{key}.parquet"
            if not produced.is_file():
                _empty_shard(parent.schema, produced)
            store.put(kind, key, lambda tmp, src=produced: os.replace(src, tmp), subdir=subdir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _run_phase(
    units: Sequence[Any],
    run_one: Callable[[Any], None],
    is_done: Callable[[Any], bool],
    *,
    worker: int,
    obsolete: Callable[[], bool] | None = None,
) -> None:
    """Run every unit of one phase exactly once across cooperating workers, and return when all are done.

    Each worker walks the unit list from its own offset, so workers spread out rather than queueing on
    the same output; ``rwlock_wrap`` inside ``run_one`` makes a unit another worker has claimed a no-op.
    A worker that finds units still outstanding waits and re-checks, which is what turns "I have done my
    share" into the barrier the next phase needs.

    Two workers can also disagree about whether a phase is needed at all -- a worker that starts late
    may find every output of the *next* phase already written by a peer, in which case its own units are
    pointless work over scratch the peer is entitled to delete. ``obsolete`` lets it notice, once per
    sweep rather than once per unit.

    A worker killed while holding a lock leaves it behind, and the survivors then wait on a unit nobody
    is doing; the wait is logged. That is the same failure mode every ``rwlock_wrap`` stage in the
    ecosystem has, and the remedy is the same: remove the stale ``.lock``.

    Args:
        units: The work items, in a fixed order every worker agrees on.
        run_one: Attempts one unit. Must be a no-op when the unit is claimed or already done.
        is_done: Whether a unit's output exists.
        worker: This worker's index, used only to rotate the walk order.
        obsolete: Checked once per sweep; when it returns ``True`` the phase returns immediately,
            however many units are outstanding.
    """
    if not units:
        return
    order = [units[(i + worker) % len(units)] for i in range(len(units))]
    warned = False
    while True:
        if obsolete is not None and obsolete():
            return
        for unit in order:
            if not is_done(unit):
                run_one(unit)
        pending = [unit for unit in order if not is_done(unit)]
        if not pending:
            return
        if not warned:
            logger.info("Waiting on %d output(s) claimed by other workers.", len(pending))
            warned = True
        order = pending
        time.sleep(_POLL_SECONDS)


def _store_shard(
    store: ShardStore,
    kind: str,
    key: str,
    write: Callable[[Path], None],
    *,
    subdir: str = "",
    guard: bool,
) -> Path:
    """Put one shard into the store, optionally serialized against cooperating workers.

    :meth:`~meds_subsetter.materialize.ShardStore.put` is a skip-if-exists cache, not a lock: two
    processes putting one key share a staging path, and the first can promote the half-written file of
    the second. When workers may be cooperating, the write therefore goes through ``rwlock_wrap`` --
    the same per-output ``FileLock`` the rest of the ecosystem uses -- and then *waits*, because a
    worker that loses the race still has to link the result into its member root.

    Args:
        store: The family's shard store.
        kind: The store bucket.
        key: The store key.
        write: Writes a complete parquet file to the path it is handed.
        subdir: The store subdirectory.
        guard: Whether other processes may be writing this store. ``False`` skips the lock entirely,
            which keeps the single-worker default free of lock files and of any wait at all.

    Returns:
        The path of the stored shard.
    """
    if not guard:
        return store.put(kind, key, write, subdir=subdir)
    stored = store.path(kind, key, subdir=subdir)

    def claim(_unit: str) -> None:
        rwlock_wrap(
            stored,
            stored,
            lambda p: p,
            lambda _p, _out: store.put(kind, key, write, subdir=subdir),
            lambda p: p,
        )

    _run_phase([key], claim, lambda _unit: store.has(kind, key, subdir=subdir), worker=0)
    return stored


def _produce_two_phase(
    parent: _Parent,
    store: ShardStore,
    kind: str,
    subdir: str,
    specs: Mapping[str, Sequence[int]],
    scratch: Path,
    *,
    worker: int,
) -> None:
    """Write every missing shard of one split with cooperating workers (addendum A2).

    Phase A is parallel over *input* shards: each is read once, joined to the assignment, and fragmented
    into one file per output shard it feeds. Phase B is parallel over *output* shards: each concatenates
    its own fragments, sorts, and writes the final shard. Total I/O is twice the bytes whatever the
    worker count, unlike giving every worker its own full-input scan.

    Args:
        parent: The resolved parent.
        store: The family's shard store.
        kind: The store bucket, i.e. the split label.
        subdir: The store subdirectory.
        specs: Output shard key to the subjects it holds.
        scratch: Directory the fragments live in. This worker's own fragment tree is removed once every
            shard of the split is stored; a peer's tree is removed by that peer, so only a *crashed*
            worker leaves reclaimable scratch behind.
        worker: This process's index, which only rotates the order units are attempted in.
    """
    missing = {key: ids for key, ids in sorted(specs.items()) if not store.has(kind, key, subdir=subdir)}
    # The fragment tree is addressed by the exact set of shards it is being built for, because a
    # phase-A marker means "this input has been fragmented for *these* outputs" and nothing weaker.
    # Two workers that start a moment apart can see different shards already stored, and sharing one
    # tree between them would let one worker's marker suppress the fragments the other still needs --
    # which silently produces a short shard. Workers that agree on the work share a tree and split it;
    # workers that do not each build their own, and both are right.
    fragments = scratch / "fragments" / kind / subdir / short_id(digest(sorted(missing)), 16)
    if missing:
        inputs = parent.shards_for(s for ids in missing.values() for s in ids)
        markers = fragments / ".markers"

        def marker_of(path: Path) -> Path:
            # Not a parquet: `rwlock_wrap`'s default checker *reads* a `.parquet` output to see whether
            # it is complete, and a peer that has finished the split is entitled to delete this whole
            # tree underneath us. A plain file is checked with `is_file()`, which cannot raise.
            return markers / f"{shard_name(parent.root, path)}.done"

        def all_stored() -> bool:
            return all(store.has(kind, key, subdir=subdir) for key in missing)

        def fragment_one(path: Path) -> None:
            index = parent.shard_paths.index(path)

            def write_fn(lf: pl.LazyFrame, out_fp: Path) -> None:
                _sink_partitioned(
                    lf, fragments, lambda key: fragments / key / f"{index:010d}.parquet", include_key=True
                )
                out_fp.parent.mkdir(parents=True, exist_ok=True)
                out_fp.write_text(shard_name(parent.root, path), encoding="utf-8")

            rwlock_wrap(
                path,
                marker_of(path),
                lambda p: _indexed(parent, [p]),
                write_fn,
                lambda lf: lf.join(_assignment(missing), on=_SUBJECT_ID, how="inner"),
            )

        _run_phase(inputs, fragment_one, lambda p: marker_of(p).is_file(), worker=worker, obsolete=all_stored)

        def reduce_one(key: str) -> None:
            def write_fn(paths: list[Path], out_fp: Path) -> None:
                def write(tmp: Path) -> None:
                    if not paths:
                        _empty_shard(parent.schema, tmp)
                        return

                    def one_file(produced: str) -> Path:
                        # A key's fragment directory holds only that key's rows, so the sink sees
                        # exactly one partition; checking says so rather than silently letting one
                        # partition overwrite another.
                        if produced != key:
                            raise RuntimeError(f"Fragments of {key!r} carry rows routed to {produced!r}.")
                        return tmp

                    _sink_partitioned(
                        _sorted_for_output(pl.scan_parquet(paths, glob=False)),
                        tmp.parent,
                        one_file,
                        include_key=False,
                    )

                store.put(kind, key, write, subdir=subdir)

            rwlock_wrap(
                fragments / key,
                store.path(kind, key, subdir=subdir),
                lambda d: sorted(d.glob("*.parquet")) if d.is_dir() else [],
                write_fn,
                lambda paths: paths,
            )

        _run_phase(sorted(missing), reduce_one, lambda k: store.has(kind, k, subdir=subdir), worker=worker)
    if all(store.has(kind, key, subdir=subdir) for key in specs):
        # Safe to drop only once *every* shard of the split is stored: at that point no worker can
        # still need a fragment, since phase B skips a key the store already has.
        shutil.rmtree(fragments, ignore_errors=True)


def _produce_shards(
    parent: _Parent,
    cfg: SubsetConfig,
    store: ShardStore,
    placements: Sequence[_Placement],
    scratch: Path,
    *,
    workers: int,
    worker: int,
) -> None:
    """Ensure every shard the given placements reference exists in the store.

    Shards are produced once for the whole family: two members that need the same rank range resolve to
    the same store entry, and the second one to ask does no work at all.

    A passed-through shard is *copied* into the store, whatever the family's ``link_mode``. The link
    mode governs the store-to-member hop and nothing else: a store entry that is itself a link into the
    parent gives every member a path that resolves straight past the store into the parent MEDS root,
    and both of :func:`~meds_subsetter.materialize.place`'s promises then fail. An in-place rewrite of
    a member's evaluation shard would land on the *parent's* file rather than visibly in the store
    (under ``hardlink`` it is the parent's inode outright), and moving the family would break every
    non-train shard in it. One copy of the fixed evaluation split per family is what decision D3
    budgets for, and it is what makes an output root self-contained.

    Args:
        parent: The resolved parent.
        cfg: The family config.
        store: The family's shard store.
        placements: Every placement of every member.
        scratch: ``<out_root>/.cache``, for staging and fragments.
        workers: How many processes are cooperating.
        worker: This process's index.
    """
    for placement in sorted(placements, key=lambda p: p.stored_name):
        if placement.source is not None and not store.has(placement.kind, placement.key):
            _store_shard(
                store,
                placement.kind,
                placement.key,
                lambda tmp, src=placement.source: place(src, tmp, LinkMode.COPY),
                guard=workers > 1,
            )

    computed: dict[tuple[str, str], dict[str, tuple[int, ...]]] = {}
    for placement in placements:
        if placement.source is None:
            computed.setdefault((placement.kind, placement.subdir), {})[placement.key] = placement.subject_ids
    for (kind, subdir), specs in sorted(computed.items()):
        if workers > 1:
            _produce_two_phase(parent, store, kind, subdir, specs, scratch, worker=worker)
        else:
            _produce_single_pass(parent, store, kind, subdir, specs, scratch)


def _write_subject_splits(splits: Mapping[str, Sequence[int]], path: Path) -> None:
    """Write a member's ``metadata/subject_splits.parquet``.

    The physical column order is ``(subject_id, split)`` and that is load-bearing:
    MEDS-Transforms' ``make_new_shards_fn`` unpacks the frame positionally
    (``for pt_id, sp in df.iter_rows()``) and would silently swap the two fields otherwise.

    Args:
        splits: Split label to subject ids.
        path: Where to write.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> path = Path(tmp.name) / "metadata" / "subject_splits.parquet"
        >>> _write_subject_splits({"train": [3, 1], "tuning": [2]}, path)
        >>> pl.read_parquet(path)
        shape: (3, 2)
        ┌────────────┬────────┐
        │ subject_id ┆ split  │
        │ ---        ┆ ---    │
        │ i64        ┆ str    │
        ╞════════════╪════════╡
        │ 1          ┆ train  │
        │ 3          ┆ train  │
        │ 2          ┆ tuning │
        └────────────┴────────┘
        >>> pl.read_parquet(path).columns  # positionally unpacked downstream; order is the contract
        ['subject_id', 'split']
        >>> tmp.cleanup()
    """
    rows = [(subject, split) for split in sorted(splits) for subject in sorted(splits[split])]
    frame = pl.DataFrame(
        {_SUBJECT_ID: [r[0] for r in rows], _SPLIT: [r[1] for r in rows]},
        schema={_SUBJECT_ID: pl.Int64, _SPLIT: pl.String},
    )
    _publish(lambda staged: atomic_write_parquet(frame, staged), path)


def _warn_if_pruning(cfg: SubsetConfig) -> None:
    """Warn that pruned code metadata makes a multi-member family's vocabulary a function of ``N``.

    Args:
        cfg: The family config.

    Examples:
        >>> import sys
        >>> handler = logging.StreamHandler(sys.stdout)
        >>> logger.addHandler(handler)
        >>> _warn_if_pruning(SubsetConfig(n_subjects=(10, 100)))
        This family prunes metadata/codes.parquet, so its 2 members have different vocabularies: a
        downstream MTD_preprocess refits fit_vocabulary_indices from each member's own codes.parquet
        and permutes the indices, worst at the small-N end. Use code_metadata='copy' to share the
        parent's vocabulary across the family.

        A single-member family has nothing to be inconsistent with, and ``copy`` is the fix:

        >>> _warn_if_pruning(SubsetConfig(n_subjects=(10,)))
        >>> _warn_if_pruning(SubsetConfig(n_subjects=(10, 100), code_metadata="copy"))
        >>> logger.removeHandler(handler)
    """
    if cfg.code_metadata is CodeMetadataPolicy.PRUNE and len(cfg.n_subjects) > 1:
        logger.warning(
            "This family prunes %s, so its %d members have different vocabularies: a downstream "
            "MTD_preprocess refits fit_vocabulary_indices from each member's own codes.parquet and "
            "permutes the indices, worst at the small-N end. Use code_metadata='copy' to share the "
            "parent's vocabulary across the family.",
            meds.code_metadata_filepath,
            len(cfg.n_subjects),
        )


def _observed_codes(paths: Sequence[Path]) -> list[str]:
    """Return the distinct, sorted codes observed in a set of shards.

    Args:
        paths: The member's data shards.

    Returns:
        The observed code strings, sorted. Nulls -- which a MEDS ``code`` may not carry, but a
        row-less shard's column can be typed as -- are dropped.
    """
    if not paths:
        return []
    codes = (
        _scan(paths, _pinned_schema(paths))
        .select(pl.col(_CODE).cast(pl.String))
        .unique()
        .drop_nulls()
        .collect()[_CODE]
    )
    return sorted(codes.to_list())


def _write_code_metadata(
    parent: _Parent,
    cfg: SubsetConfig,
    store: ShardStore,
    member_root: Path,
    data_paths: Sequence[Path],
    observed_code_set_id: str,
    *,
    guard: bool,
) -> str:
    """Place the member's ``metadata/codes.parquet`` and return its digest.

    Under :attr:`~meds_subsetter.config.CodeMetadataPolicy.COPY` the parent's file is shared verbatim by
    the whole family, which is the only policy under which a nested family has one vocabulary. Under
    ``PRUNE`` the file is filtered to the codes this member observes and stored under *that* code set's
    id, so two members observing the same codes share one file and neither rebuilds it.

    ``observed_code_set_id`` is :meth:`~meds_subsetter.sizes.SizeReport.observed_code_set_id`, which is
    the digest of exactly the sorted code list ``PRUNE`` would filter by. Keying on it means the member's
    codes are re-read only when the pruned file is genuinely missing from the store.

    Args:
        parent: The resolved parent.
        cfg: The family config.
        store: The family's shard store.
        member_root: The member's MEDS root.
        data_paths: The member's data shards, read only when a pruned file must be built.
        observed_code_set_id: Digest of the codes observed in the member's data.
        guard: Whether other processes may be writing this store; see :func:`_store_shard`.

    Returns:
        The digest of the emitted file, or of nothing when the parent carries no code metadata.
    """
    source = parent.root / meds.code_metadata_filepath
    if not source.is_file():
        logger.warning("Parent %s has no %s; emitting none.", parent.root, meds.code_metadata_filepath)
        return digest([])

    if cfg.code_metadata is CodeMetadataPolicy.COPY:
        key = "codes/parent"

        def write(tmp: Path) -> None:
            # Copied, not linked, for the reason `_produce_shards` gives: a store entry that links
            # into the parent hands every member a path that resolves past the store to the parent's
            # own file, so an in-place rewrite of a member's codes.parquet would edit the parent's.
            place(source, tmp, LinkMode.COPY)
    else:
        key = f"codes/{short_id(observed_code_set_id)}"

        def write(tmp: Path) -> None:
            wanted = pl.Series(_CODE, _observed_codes(data_paths), dtype=pl.String)
            pl.read_parquet(source, glob=False).filter(pl.col(_CODE).is_in(wanted.implode())).write_parquet(
                tmp
            )

    stored = _store_shard(store, METADATA_KIND, key, write, guard=guard)
    _link_into(store, stored, member_root / meds.code_metadata_filepath, cfg.link_mode)
    return frame_digest(pl.scan_parquet(stored, glob=False))


def _write_dataset_metadata(parent: _Parent, member_root: Path, subset_of: Mapping[str, Any]) -> None:
    """Write the member's ``metadata/dataset.json``: the parent's, plus one ``subset_of`` key.

    Args:
        parent: The resolved parent.
        member_root: The member's MEDS root.
        subset_of: The provenance block to add.

    Raises:
        ValueError: If the parent's ``dataset.json`` is not a JSON object, which the schema requires.
    """
    source = parent.root / meds.dataset_metadata_filepath
    raw: Any = {}
    if source.is_file():
        raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must hold a JSON object; got {type(raw).__name__}")
    merged = {**raw, SUBSET_OF_KEY: dict(subset_of)}
    _publish(
        lambda staged: atomic_write_json(merged, staged),
        member_root / meds.dataset_metadata_filepath,
    )


def _write_index(
    parent: _Parent,
    member_root: Path,
    index_dir: Path,
    task_name: str,
    placements: Sequence[_Placement],
    subjects: Sequence[int],
) -> tuple[str, int]:
    """Re-shard a task/index label frame to match the member's data shards exactly.

    Matching the *data* sharding satisfies both conventions in the ecosystem at once: MEDS-Tab derives a
    label file's path positionally from its data shard's path, while MEDS-Torch-Data and others glob the
    task directory. One label file per data shard is the only layout both accept, so a shard with no
    labels gets an empty, schema-correct file rather than no file.

    The label columns are taken as they are found. ``meds.LabelSchema.align`` is deliberately *not*
    called: real ACES output leaves ``integer_value`` / ``float_value`` / ``categorical_value`` entirely
    null while the closed schema declares them non-nullable, so aligning raises on genuine data.

    Args:
        parent: The resolved parent, for resolving a passed-through shard's subjects.
        member_root: The member's MEDS root.
        index_dir: Directory of label parquet files, read recursively, dotfiles skipped.
        task_name: The directory under ``tasks/`` to write into.
        placements: The member's shards, which fix the output sharding.
        subjects: Every subject the member carries.

    Returns:
        The digest of the member's label rows, and how many there are.

    Raises:
        FileNotFoundError: If ``index_dir`` holds no parquet files.
    """
    paths = _parquet_shard_paths(index_dir)
    if not paths:
        raise FileNotFoundError(f"No parquet label files under {index_dir}")
    # Label files differ in width just as data shards do -- ACES emits only the value column a task
    # actually uses -- so the rendered set is pinned to their union, exactly as for the data.
    schema = _pinned_schema(paths)
    lf = _scan(paths, schema)
    wanted = pl.Series(_SUBJECT_ID, sorted(subjects), dtype=pl.Int64)
    labels = lf.filter(pl.col(_SUBJECT_ID).is_in(wanted.implode())).collect()

    tasks_root = member_root / TASKS_DIRNAME / task_name
    for placement in placements:
        held = pl.Series(_SUBJECT_ID, _shard_subjects(parent, placement), dtype=pl.Int64)
        slice_ = labels.filter(pl.col(_SUBJECT_ID).is_in(held.implode()))
        _publish(
            lambda staged, frame=slice_: atomic_write_parquet(frame, staged),
            tasks_root / f"{placement.shard}.parquet",
        )
    # The labels shadow the data sharding, so they go stale in exactly the same way and are
    # reconciled in exactly the same way; see :func:`_prune_shards`.
    for stale in _prune_shards(tasks_root, {p.shard for p in placements}):
        logger.info("Removed %s from %s: this build does not name it.", stale, tasks_root)
    return frame_digest(labels.lazy(), schema=schema), labels.height


def _dataset_id(content_ids: Mapping[str, str], codes_id: str) -> str:
    """Combine a member's per-split content ids and its code metadata id into one dataset id.

    Args:
        content_ids: Split label to that split's content id.
        codes_id: The digest of the member's code metadata.

    Returns:
        A ``"sha256:<hex>"`` digest, independent of the order the splits arrive in.

    Examples:
        >>> a, b = digest(["train"]), digest(["tuning"])
        >>> _dataset_id({"train": a, "tuning": b}, digest([])) == _dataset_id(
        ...     {"tuning": b, "train": a}, digest([])
        ... )
        True
        >>> _dataset_id({"train": a}, digest([])) == _dataset_id({"train": a, "tuning": b}, digest([]))
        False
        >>> _dataset_id({"train": a}, digest([])) == _dataset_id({"train": a}, digest(["A"]))
        False
    """
    items = [f"{meds.data_subdirectory}/{split}={cid}" for split, cid in content_ids.items()]
    items.append(f"{meds.code_metadata_filepath}={codes_id}")
    return digest(sorted(items))


def _build_member(
    parent: _Parent,
    cfg: SubsetConfig,
    out_root: Path,
    store: ShardStore,
    n: int,
    ranked: Mapping[str, Sequence[int]],
    *,
    index_dir: Path | None,
    task_name: str | None,
    workers: int,
    worker: int,
) -> SubsetManifest:
    """Build one member of a family, assuming the parent has already been resolved.

    Args:
        parent: The resolved parent.
        cfg: The family config.
        out_root: The family's output root.
        store: The family's shard store.
        n: The member's train size.
        ranked: Split label to that split's rank order.
        index_dir: Optional directory of task/index label parquet files.
        task_name: Name of the task directory; defaults to ``index_dir.name``.
        workers: How many processes are cooperating.
        worker: This process's index.

    Returns:
        The member's manifest.
    """
    name = cfg.subset_name(n)
    member_root = out_root / name
    splits = _member_splits(parent, cfg, n, ranked)
    placements = _placements(parent, cfg, n, ranked)
    _produce_shards(parent, cfg, store, placements, out_root / CACHE_DIRNAME, workers=workers, worker=worker)

    data_root = member_root / meds.data_subdirectory
    for placement in placements:
        stored = store.path(placement.kind, placement.key, subdir=placement.subdir)
        _link_into(store, stored, data_root / f"{placement.shard}.parquet", cfg.link_mode)
    # Placing is not enough: a rebuild whose plan is *smaller* than the last one's -- a larger
    # `n_subjects_per_shard`, a `non_train_policy` that no longer passes the evaluation splits
    # through -- would otherwise leave the previous run's shards beside the new ones, and a subject
    # would live in two files at once.
    for stale in _prune_shards(data_root, {p.shard for p in placements}):
        logger.info("Removed %s from %s: this build does not name it.", stale, data_root)

    _write_subject_splits(splits, member_root / meds.subject_splits_filepath)
    # Sized before the metadata is written, so the pruned code file can be keyed by the report's own
    # `observed_code_set_id` rather than by a second scan of the member's data.
    report = size_meds(member_root)
    if report.n_shards != len(placements):
        # The plan and the tree have to agree exactly, since every id and every count below is
        # computed from the plan while a reader sees the tree. This cannot fire after the
        # reconciliation above; it is here so that if it ever does, it does so loudly.
        raise RuntimeError(
            f"{member_root} holds {report.n_shards} data shard(s) but this build planned "
            f"{len(placements)}; refusing to report ids for a tree that is not the one planned."
        )
    codes_id = _write_code_metadata(
        parent,
        cfg,
        store,
        member_root,
        [data_root / f"{p.shard}.parquet" for p in placements],
        report.observed_code_set_id,
        guard=workers > 1,
    )

    content_ids = {}
    for split, ids in splits.items():
        cached = _cached_subject_digests(parent, split)
        wanted = pl.Series(_SUBJECT_ID, list(ids), dtype=pl.Int64)
        content_ids[split] = combine_subject_digests(
            cached.filter(pl.col(_SUBJECT_ID).is_in(wanted.implode()))
        )
    subject_ids = sorted({s for ids in splits.values() for s in ids})
    subject_set_id = subject_set_digest(subject_ids)
    dataset_id = _dataset_id(content_ids, codes_id)

    _write_dataset_metadata(
        parent,
        member_root,
        {
            "parent": str(parent.root),
            "parent_subject_set_id": parent.subject_set_id,
            "parent_shard_fingerprint": parent.fingerprint,
            "subset_id": dataset_id,
            "subject_set_id": subject_set_id,
            "n_subjects": n,
            "tool": __package_name__,
            "version": __version__,
        },
    )

    index_id: str | None = None
    n_index_rows: int | None = None
    resolved_task: str | None = None
    if index_dir is not None:
        resolved_task = task_name if task_name is not None else index_dir.name
        index_id, n_index_rows = _write_index(
            parent, member_root, index_dir, resolved_task, placements, subject_ids
        )

    manifest = SubsetManifest(
        name=name,
        root=str(member_root),
        n_subjects=n,
        subject_set_id=subject_set_id,
        train_set_id=content_ids[cfg.train_split],
        content_ids=content_ids,
        dataset_id=dataset_id,
        codes_id=codes_id,
        observed_code_set_id=report.observed_code_set_id,
        index_id=index_id,
        n_index_rows=n_index_rows,
        task_name=resolved_task,
        shards={p.shard: p.stored_name for p in placements},
        sizes=report.to_dict(),
    )
    provenance = {
        "tool": __package_name__,
        "version": __version__,
        "parent": {
            "path": str(parent.root),
            "subject_set_id": parent.subject_set_id,
            "fingerprint": parent.fingerprint,
        },
        "config": cfg.to_dict(),
        "member": manifest.to_dict(),
    }
    _publish(
        lambda staged: atomic_write_json(provenance, staged),
        member_root / MEMBER_MANIFEST_DIRNAME / MEMBER_MANIFEST_NAME,
    )
    return manifest


def _measure(measure: Callable[[Path], int], path: Path) -> int:
    """Measure a file tree, retrying past a file a cooperating worker removed mid-walk.

    Walking a tree that other workers are still writing into is inherently racy: a staging file can be
    listed and then promoted before it is ``lstat``-ed, and the foundation's measures do not swallow
    that. Staging files live for milliseconds, so a short retry is enough; a failure that survives every
    attempt is something else and is raised.

    Args:
        measure: :func:`~meds_subsetter.materialize.dir_size` or
            :func:`~meds_subsetter.materialize.unique_inode_size`, or a partial of one.
        path: The tree to measure.

    Returns:
        The measured bytes.

    Raises:
        OSError: If every attempt fails.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> _ = (Path(tmp.name) / "a.txt").write_bytes(b"x" * 10)
        >>> _measure(dir_size, Path(tmp.name))
        10
        >>> _measure(dir_size, Path(tmp.name) / "nope")
        Traceback (most recent call last):
            ...
        FileNotFoundError: No such file or directory: /...nope
        >>> tmp.cleanup()
    """
    for attempt in range(_MEASURE_ATTEMPTS):
        try:
            return measure(path)
        except OSError:
            if attempt + 1 == _MEASURE_ATTEMPTS:
                raise
            time.sleep(_POLL_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover


def _disk_report(out_root: Path, store: ShardStore, members: Sequence[SubsetManifest]) -> dict[str, int]:
    """Measure what the family actually cost, and what it would have cost as independent copies.

    ``n_bytes_on_disk`` counts the shard store and every member root, charging each inode once, so a
    hardlinked or symlinked shard is paid for exactly as often as it really is.
    ``n_bytes_independent_copies`` is what the same members would have cost as standalone MEDS roots,
    with every link resolved to the bytes it points at. The difference between the two is the saving the
    store exists to produce. The reclaimable ``.cache`` and the family manifest itself are excluded from
    the first figure and reported separately, so a rebuild measures the same tree the first run did.

    Args:
        out_root: The family's output root.
        store: The family's shard store.
        members: The members built.

    Returns:
        Byte counts and the store's file count. Under cooperating workers these are a snapshot: a peer
        that is still building will not be finished being counted.
    """
    manifest = out_root / FAMILY_MANIFEST_NAME
    cache_dir = out_root / CACHE_DIRNAME
    # The manifest describes the tree, so counting it would make the figures depend on whether an
    # earlier run's manifest happened to be lying there; the cache is a rebuild accelerator that can be
    # deleted at any moment, so charging the family for it would understate the sharing. Both are
    # distinct inodes shared with nothing else, so subtracting them is exact -- and the cache is
    # reported on its own line rather than hidden.
    own = manifest.stat().st_size if manifest.is_file() else 0
    cache = _measure(unique_inode_size, cache_dir) if cache_dir.is_dir() else 0
    independent = sum(_measure(lambda p: dir_size(p, follow_links=True), Path(m.root)) for m in members)
    on_disk = _measure(unique_inode_size, out_root) - own - cache
    stat = store.stat()
    return {
        "n_bytes_on_disk": on_disk,
        "n_bytes_apparent": _measure(dir_size, out_root) - own - cache,
        "n_bytes_independent_copies": independent,
        "n_bytes_saved": independent - on_disk,
        "n_bytes_cache": cache,
        "n_store_files": stat["n_files"],
        "n_store_bytes": stat["n_bytes"],
    }


def _check_family(out_root: Path, parent: _Parent, store: ShardStore, *, do_overwrite: bool) -> None:
    """Refuse to extend a family whose recorded parent is not the one being handed in.

    The store is addressed by rank range under one salt, not by content, so a second parent pointed at
    the same output root would silently be handed the first one's shards. Both recorded parent ids are
    therefore checked, and they answer different questions:

    * ``subject_set_id`` says whether these are the same *subjects*. A different cohort's rank ranges
      name different people, so its stored shards are not this parent's at any size.
    * ``fingerprint`` says whether they are the same *bytes* (:func:`_fingerprint`). It is the one that
      catches the ordinary case -- an ETL re-run, a corrected value, a fixed ``codes.parquet`` -- where
      the subject set is untouched and only the data moved. Without it the derived caches are rebuilt
      against the new parent while the store keeps serving the old one's shards, and the member ends up
      carrying ids that do not describe its own data.

    A mismatch is refused rather than resolved, because both possible resolutions are destructive: this
    parent's shards would have to be rebuilt (discarding the recorded family) or the recorded family's
    kept (mislabelling this one). ``do_overwrite`` picks the first, explicitly.

    Args:
        out_root: The family's output root.
        parent: The resolved parent.
        store: The family's shard store, discarded when a mismatch is overridden.
        do_overwrite: Whether to discard the recorded family and rebuild.

    Raises:
        ValueError: On a mismatch, unless ``do_overwrite``.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name) / "parent"
        >>> _ = shutil.copytree(simple_static_MEDS, root)
        >>> out = Path(tmp.name) / "family"
        >>> cfg = SubsetConfig(n_subjects=(2,), n_subjects_per_shard=2, code_metadata="copy")
        >>> before = build_family(root, out, cfg).members[0]
        >>> unchanged = build_family(root, out, cfg).members[0]
        >>> unchanged == before
        True

        Now the parent's data changes while its subject set does not -- one value corrected, which is
        what an ETL re-run looks like. The stored shards are the *old* parent's, so the build is
        refused instead of quietly emitting a member whose ``train_set_id`` describes bytes that are
        not in its ``data/``:

        >>> shard = root / "data" / "train" / "0.parquet"
        >>> pl.read_parquet(shard).with_columns(
        ...     numeric_value=pl.col("numeric_value") + 1
        ... ).write_parquet(shard)
        >>> build_family(root, out, cfg)
        Traceback (most recent call last):
            ...
        ValueError: ...family.json records a family built from '...' with fingerprint sha256:...,
        but ...parent has sha256:...; its shards are not interchangeable. Pass do_overwrite=True to
        discard and rebuild.

        ``do_overwrite`` discards the store and rebuilds, and the id the manifest then reports is a
        digest of the bytes actually on disk -- which is the whole point of the binding:

        >>> after = build_family(root, out, cfg, do_overwrite=True).members[0]
        >>> after.train_set_id == before.train_set_id
        False
        >>> emitted = sorted((out / "N0000002" / "data" / "train").glob("*.parquet"))
        >>> schema = _pinned_schema(emitted)
        >>> combine_subject_digests(
        ...     subject_digests(_scan(emitted, schema), schema=schema)
        ... ) == after.train_set_id
        True

        The parent's ``metadata/codes.parquet`` is bound too. A member copies it, so a corrected one
        that went unnoticed would be served out of the store for the life of the family:

        >>> codes = root / "metadata" / "codes.parquet"
        >>> pl.read_parquet(codes).with_columns(
        ...     description=pl.lit("corrected")
        ... ).write_parquet(codes)
        >>> build_family(root, out, cfg)
        Traceback (most recent call last):
            ...
        ValueError: ...with fingerprint sha256:..., but ...parent has sha256:...
        >>> _ = build_family(root, out, cfg, do_overwrite=True)
        >>> sorted(
        ...     pl.read_parquet(out / "N0000002" / "metadata" / "codes.parquet")["description"].unique()
        ... )
        ['corrected']
        >>> tmp.cleanup()
    """
    manifest = out_root / FAMILY_MANIFEST_NAME
    if not manifest.is_file():
        return
    recorded = json.loads(manifest.read_text(encoding="utf-8")).get("parent", {})
    moved = [
        (field, recorded.get(field), current)
        for field, current in (
            ("subject_set_id", parent.subject_set_id),
            ("fingerprint", parent.fingerprint),
        )
        if recorded.get(field) != current
    ]
    if not moved:
        return
    if not do_overwrite:
        field, was, now = moved[0]
        raise ValueError(
            f"{manifest} records a family built from {recorded.get('path')!r} with {field} "
            f"{was}, but {parent.root} has {now}; its shards are not interchangeable. Pass "
            f"do_overwrite=True to discard and rebuild."
        )
    logger.warning("Parent of %s changed; discarding the existing shard store and caches.", out_root)
    shutil.rmtree(store.root, ignore_errors=True)
    # The caches are keyed by the parent's shard fingerprint -- relative paths and sizes -- which two
    # different datasets can share (a synthetic cohort regenerated under a new seed, say). The subject
    # set says they are different, so the derived state goes too, memo included.
    shutil.rmtree(parent.cache_dir, ignore_errors=True)
    parent.forget()
    _write_stamp(parent.root, parent.fingerprint, parent.cache_dir)


def _check_workers(workers: int, worker: int) -> None:
    """Validate the worker arguments.

    Args:
        workers: How many processes are cooperating.
        worker: This process's index.

    Raises:
        ValueError: If ``workers`` is below 1 or ``worker`` is outside ``[0, workers)``.

    Examples:
        >>> _check_workers(1, 0)
        >>> _check_workers(0, 0)
        Traceback (most recent call last):
            ...
        ValueError: workers must be >= 1; got 0
        >>> _check_workers(4, 4)
        Traceback (most recent call last):
            ...
        ValueError: worker must be in [0, 4); got 4
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1; got {workers}")
    if not 0 <= worker < workers:
        raise ValueError(f"worker must be in [0, {workers}); got {worker}")


def _rank_orders(parent: _Parent, cfg: SubsetConfig) -> dict[str, list[int]]:
    """Rank every split that will actually be selected from.

    Only the train split is ranked under the default policy: ranking is a SHA-256 per subject, so a
    cohort's evaluation splits are not hashed unless :attr:`~meds_subsetter.config.NonTrainPolicy.SUBSET`
    is going to select from them.

    Args:
        parent: The resolved parent.
        cfg: The family config.

    Returns:
        Split label to that split's subjects in rank order.
    """
    wanted = [cfg.train_split]
    if cfg.non_train_policy is NonTrainPolicy.SUBSET:
        wanted.extend(_non_train_splits(parent, cfg))
    return {split: rank_subjects(parent.splits.get(split, ()), cfg.salt) for split in wanted}


def build_subset(
    parent: Path,
    out_root: Path,
    cfg: SubsetConfig,
    n: int,
    *,
    index_dir: Path | None = None,
    task_name: str | None = None,
    workers: int = 1,
    worker: int = 0,
    do_overwrite: bool = False,
) -> SubsetManifest:
    """Build a single member of a family, into a family layout that :func:`build_family` can extend.

    Everything a member needs is written: its data shards (through the shared store), its
    ``metadata/{codes.parquet,subject_splits.parquet,dataset.json}``, its provenance manifest, and -- if
    an ``index_dir`` is given -- its task labels, resharded to match its data. No ``family.json`` is
    written, so the parent binding that guards a shared store is *not* established; prefer
    :func:`build_family` unless you are building one size on purpose.

    Args:
        parent: The parent MEDS root.
        out_root: Where the family lives. Created if absent.
        cfg: The family config. ``n`` need not be one of its ``n_subjects``.
        n: The member's train size, in subjects.
        index_dir: Optional directory of task/index label parquet files, read recursively.
        task_name: Name of the ``tasks/`` subdirectory; defaults to ``index_dir.name``.
        workers: How many processes are cooperating on this output root.
        worker: This process's index in ``[0, workers)``.
        do_overwrite: Whether to discard a shard store recorded against a different parent.

    Returns:
        The member's manifest.

    Raises:
        NotAMEDSDatasetError: If ``parent`` is not a usable MEDS root.
        ValueError: If ``n`` exceeds the parent's train pool, or the worker arguments are invalid.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "one"
        >>> cfg = SubsetConfig(n_subjects=(2,), n_subjects_per_shard=2, code_metadata="copy")
        >>> member = build_subset(simple_static_MEDS, out, cfg, 2)
        >>> member.name, member.n_subjects, sorted(member.shards)
        ('N0000002', 2, ['held_out/0', 'train/0', 'tuning/0'])

        The result is a complete MEDS root that the rest of the package can read back:

        >>> from meds_subsetter.sizes import size_meds
        >>> size_meds(out / "N0000002").to_frame().select("split", "n_subjects", "n_measurements")
        shape: (4, 3)
        ┌───────────┬────────────┬────────────────┐
        │ split     ┆ n_subjects ┆ n_measurements │
        │ ---       ┆ ---        ┆ ---            │
        │ str       ┆ i64        ┆ i64            │
        ╞═══════════╪════════════╪════════════════╡
        │ held_out  ┆ 1          ┆ 11             │
        │ train     ┆ 2          ┆ 20             │
        │ tuning    ┆ 1          ┆ 7              │
        │ __total__ ┆ 4          ┆ 38             │
        └───────────┴────────────┴────────────────┘

        Rebuilding a member into a root it is already in *reconciles* that root with the new plan
        rather than adding to it. A larger ``n_subjects_per_shard`` means fewer, bigger shards, so the
        shards the previous build placed have to go: left behind, they would put each of those
        subjects in two data files at once -- the one MEDS invariant everything here leans on -- and
        silently double every count in the root.

        >>> four = build_subset(simple_static_MEDS, out, cfg, 4)
        >>> sorted(four.shards)
        ['held_out/0', 'train/0', 'train/1', 'tuning/0']
        >>> coarse = SubsetConfig(n_subjects=(4,), n_subjects_per_shard=4, code_metadata="copy")
        >>> rebuilt = build_subset(simple_static_MEDS, out, coarse, 4)
        >>> sorted(rebuilt.shards)
        ['held_out/0', 'train/0', 'tuning/0']

        What the manifest names is what is on disk, and the counts are the parent's own:

        >>> member_root = out / "N0000004"
        >>> [shard_name(member_root, p) for p in meds_shard_paths(member_root)]
        ['held_out/0', 'train/0', 'tuning/0']
        >>> report = size_meds(member_root)
        >>> report.n_shards, report.total.n_subjects, report.total.n_measurements
        (3, 6, 62)

        Asking for more subjects than the parent's train split holds names both numbers rather than
        quietly building a smaller subset, which would corrupt a sweep's x-axis:

        >>> build_subset(simple_static_MEDS, out, cfg, 9)
        Traceback (most recent call last):
            ...
        ValueError: Cannot build a 9-subject subset: the parent's 'train' split has only 4 subjects.
        >>> tmp.cleanup()
    """
    _check_workers(workers, worker)
    out_root = Path(out_root)
    resolved = _open_parent(Path(parent), out_root / CACHE_DIRNAME)
    store = ShardStore(out_root / SHARD_STORE_DIRNAME)
    _check_family(out_root, resolved, store, do_overwrite=do_overwrite)
    return _build_member(
        resolved,
        cfg,
        out_root,
        store,
        n,
        _rank_orders(resolved, cfg),
        index_dir=None if index_dir is None else Path(index_dir),
        task_name=task_name,
        workers=workers,
        worker=worker,
    )


def build_family(
    parent: Path,
    out_root: Path,
    cfg: SubsetConfig,
    *,
    index_dir: Path | None = None,
    task_name: str | None = None,
    workers: int = 1,
    worker: int = 0,
    do_overwrite: bool = False,
) -> FamilyManifest:
    r"""Build every member of a nested subset family, sharing shards between them.

    Args:
        parent: The parent MEDS root.
        out_root: Where the family lives. Created if absent.
        cfg: The family config: the sizes to build, the salt, the shard size, and the policies.
        index_dir: Optional directory of task/index label parquet files, read recursively (dotfiles
            skipped). Each member gets its own slice, resharded to match its data shards exactly.
        task_name: Name of the ``tasks/`` subdirectory; defaults to ``index_dir.name``.
        workers: How many processes are cooperating on this output root. ``1`` (the default) uses the
            single streaming pass; more uses the two-phase, ``rwlock``-guarded scheme, which produces
            byte-identical shards.
        worker: This process's index in ``[0, workers)``.
        do_overwrite: Whether to discard a family recorded against a different -- or a changed --
            parent and rebuild it against this one.

    Returns:
        The family manifest, also written to ``<out_root>/family.json``.

    Raises:
        NotAMEDSDatasetError: If ``parent`` is not a usable MEDS root.
        ValueError: If any requested size exceeds the parent's train pool, if the worker arguments are
            invalid, if a parent shard name collides with a rank bucket's (see :func:`_placements`), or
            if the output root already records a parent whose subjects or whose bytes have moved and
            ``do_overwrite`` is not set.

    Examples:
        The fixture has four train subjects (239684, 1195293, 68729, 814703), one tuning subject and one
        held-out subject. Two members, at two and four subjects, with two subjects per shard so that the
        smaller member's shard is a *full* shard of the larger one:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "sweep"
        >>> cfg = SubsetConfig(n_subjects=(2, 4), n_subjects_per_shard=2, code_metadata="copy")
        >>> family = build_family(simple_static_MEDS, out, cfg)
        >>> [(m.name, m.n_subjects) for m in family.members]
        [('N0000002', 2), ('N0000004', 4)]

        Each member is a complete MEDS root; the store holds one copy of each distinct shard, and every
        member's ``data/`` is links into it:

        >>> print_directory(out, config=PrintConfig(ignore_regex=r"\.cache"))
        ├── .shard_store
        │   ├── metadata
        │   │   └── codes
        │   │       └── parent.parquet
        │   ├── passthrough
        │   │   ├── held_out
        │   │   │   └── 0.parquet
        │   │   └── tuning
        │   │       └── 0.parquet
        │   └── train
        │       └── 6b0e7e5d
        │           ├── r0000000000-0000000002.parquet
        │           └── r0000000002-0000000004.parquet
        ├── N0000002
        │   ├── .meds_subsetter
        │   │   └── manifest.json
        │   ├── data
        │   │   ├── held_out
        │   │   │   └── 0.parquet
        │   │   ├── train
        │   │   │   └── 0.parquet
        │   │   └── tuning
        │   │       └── 0.parquet
        │   └── metadata
        │       ├── codes.parquet
        │       ├── dataset.json
        │       └── subject_splits.parquet
        ├── N0000004
        │   ├── .meds_subsetter
        │   │   └── manifest.json
        │   ├── data
        │   │   ├── held_out
        │   │   │   └── 0.parquet
        │   │   ├── train
        │   │   │   ├── 0.parquet
        │   │   │   └── 1.parquet
        │   │   └── tuning
        │   │       └── 0.parquet
        │   └── metadata
        │       ├── codes.parquet
        │       ├── dataset.json
        │       └── subject_splits.parquet
        └── family.json

        **The point of the whole package**: the two-subject member's train shard is not a copy of the
        four-subject member's ``train/0`` -- it is the same file:

        >>> small = out / "N0000002" / "data" / "train" / "0.parquet"
        >>> large = out / "N0000004" / "data" / "train" / "0.parquet"
        >>> small.samefile(large), small.resolve() == large.resolve()
        (True, True)
        >>> os.readlink(small)
        '../../../.shard_store/train/6b0e7e5d/r0000000000-0000000002.parquet'
        >>> family.members[0].shards["train/0"] == family.members[1].shards["train/0"]
        True

        The non-train splits are shared too, so the evaluation set is fixed across the sweep:

        >>> [
        ...     (out / m / "data" / s / "0.parquet").resolve()
        ...     for m in ("N0000002", "N0000004")
        ...     for s in ("tuning",)
        ... ] == [(out / ".shard_store" / "passthrough" / "tuning" / "0.parquet").resolve()] * 2
        True

        Those shared entries are the store's own bytes, not links back into the parent, whatever the
        family's ``link_mode``. That is what makes an output root self-contained -- movable, and safe
        to write into -- rather than a set of paths that resolve past the store into the parent MEDS
        root, where an in-place rewrite of one member's evaluation shard would edit the parent's own
        dataset and moving the family would break every non-train shard in it:

        >>> shared = out / ".shard_store" / "passthrough" / "tuning" / "0.parquet"
        >>> codes = out / ".shard_store" / "metadata" / "codes" / "parent.parquet"
        >>> shared.is_symlink(), shared.is_file(), codes.is_symlink(), codes.is_file()
        (False, True, False, True)
        >>> os.readlink(out / "N0000002" / "data" / "tuning" / "0.parquet")
        '../../../.shard_store/passthrough/tuning/0.parquet'
        >>> os.readlink(out / "N0000002" / "metadata" / "codes.parquet")
        '../../.shard_store/metadata/codes/parent.parquet'

        The members really are nested, and only in the train split:

        >>> subjects = [pl.read_parquet(out / m / "metadata" / "subject_splits.parquet")
        ...             for m in ("N0000002", "N0000004")]
        >>> [sorted(df.filter(pl.col("split") == "train")["subject_id"]) for df in subjects]
        [[239684, 814703], [68729, 239684, 814703, 1195293]]
        >>> set(subjects[0]["subject_id"]) <= set(subjects[1]["subject_id"])
        True

        Each member's train split is exactly ``selection.select_nested`` of the parent's train pool.
        The rank order is computed once for the family and sliced per member rather than re-derived,
        which is the same list -- and this is what says so:

        >>> from meds_subsetter.selection import select_nested
        >>> pool = pl.read_parquet(simple_static_MEDS / "metadata" / "subject_splits.parquet").filter(
        ...     pl.col("split") == "train"
        ... )["subject_id"].to_list()
        >>> [sorted(select_nested(pool, n, cfg.salt)) for n in (2, 4)] == [
        ...     sorted(df.filter(pl.col("split") == "train")["subject_id"]) for df in subjects
        ... ]
        True

        Ids are recorded per member; the train content id moves with ``N`` while the shared held-out
        split's does not:

        >>> a, b = family.members
        >>> a.train_set_id == b.train_set_id, a.content_ids["held_out"] == b.content_ids["held_out"]
        (False, True)
        >>> a.dataset_id == b.dataset_id
        False
        >>> a.sizes["total"]["n_subjects"], b.sizes["total"]["n_subjects"]
        (4, 6)

        The family costs far less than two independent copies would have, and the manifest says so:

        >>> d = family.disk
        >>> d["n_bytes_on_disk"] < d["n_bytes_independent_copies"], d["n_bytes_saved"] > 0
        (True, True)
        >>> d["n_store_files"]
        5

        Rebuilding is a cheap no-op: an equal manifest, and not one shard rewritten.

        >>> stamp = {p: p.stat().st_mtime_ns for p in sorted((out / ".shard_store").rglob("*.parquet"))}
        >>> build_family(simple_static_MEDS, out, cfg) == family
        True
        >>> {p: p.stat().st_mtime_ns for p in sorted((out / ".shard_store").rglob("*.parquet"))} == stamp
        True

        The multi-worker path is the same computation done differently, and it writes the same bytes:

        >>> par = Path(tmp.name) / "parallel"
        >>> parallel = build_family(simple_static_MEDS, par, cfg, workers=2)
        >>> store = out / ".shard_store"
        >>> [
        ...     p.read_bytes() == (par / ".shard_store" / p.relative_to(store)).read_bytes()
        ...     for p in sorted(store.rglob("*.parquet"))
        ... ]
        [True, True, True, True, True]
        >>> [m.dataset_id for m in parallel.members] == [m.dataset_id for m in family.members]
        True

        A second parent cannot be quietly handed the first one's shards:

        >>> other = Path(tmp.name) / "other"
        >>> (other / "data" / "train").mkdir(parents=True)
        >>> pl.DataFrame(
        ...     {
        ...         "subject_id": [7, 8],
        ...         "time": pl.Series([None, None], dtype=pl.Datetime("us")),
        ...         "code": ["A", "B"],
        ...     }
        ... ).write_parquet(other / "data" / "train" / "0.parquet")
        >>> build_family(other, out, cfg)
        Traceback (most recent call last):
            ...
        ValueError: ...family.json records a family built from '...' with subject_set_id sha256:...,
        but ...other has sha256:...; its shards are not interchangeable. Pass do_overwrite=True to
        discard and rebuild.
        >>> tmp.cleanup()

        With a task directory, every member gets its own labels, resharded one file per data shard --
        including an empty, schema-correct file for a shard whose subjects have no labels:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "task"
        >>> labels = simple_static_MEDS_dataset_with_task / "task_labels" / "boolean_value_task"
        >>> family = build_family(
        ...     simple_static_MEDS_dataset_with_task, out, cfg, index_dir=labels
        ... )
        >>> [(m.task_name, m.n_index_rows) for m in family.members]
        [('boolean_value_task', 14), ('boolean_value_task', 21)]
        >>> print_directory(out / "N0000002" / "tasks")
        └── boolean_value_task
            ├── held_out
            │   └── 0.parquet
            ├── train
            │   └── 0.parquet
            └── tuning
                └── 0.parquet
        >>> pl.read_parquet(out / "N0000002" / "tasks" / "boolean_value_task" / "train" / "0.parquet")[
        ...     "subject_id"
        ... ].unique().sort().to_list()
        [239684, 814703]
        >>> family.members[0].index_id == family.members[1].index_id
        False

        The labels shadow the data sharding, so they go stale in exactly the way the data does, and are
        reconciled in exactly the same way: rebuilding into fewer, bigger shards leaves no label file
        behind to be globbed up beside the new ones.

        >>> print_directory(out / "N0000004" / "tasks" / "boolean_value_task")
        ├── held_out
        │   └── 0.parquet
        ├── train
        │   ├── 0.parquet
        │   └── 1.parquet
        └── tuning
            └── 0.parquet
        >>> coarse = SubsetConfig(n_subjects=(4,), n_subjects_per_shard=4, code_metadata="copy")
        >>> _ = build_family(
        ...     simple_static_MEDS_dataset_with_task, out, coarse, index_dir=labels
        ... )
        >>> print_directory(out / "N0000004" / "tasks" / "boolean_value_task")
        ├── held_out
        │   └── 0.parquet
        ├── train
        │   └── 0.parquet
        └── tuning
            └── 0.parquet
        >>> tmp.cleanup()
    """
    _check_workers(workers, worker)
    out_root = Path(out_root)
    resolved = _open_parent(Path(parent), out_root / CACHE_DIRNAME)
    store = ShardStore(out_root / SHARD_STORE_DIRNAME)
    _check_family(out_root, resolved, store, do_overwrite=do_overwrite)
    _warn_if_pruning(cfg)

    ranked = _rank_orders(resolved, cfg)
    pool = ranked[cfg.train_split]
    # Checked for every member up front, not just as each one is reached: a family whose largest member
    # is impossible should not leave half a sweep on disk before saying so.
    for n in cfg.n_subjects:
        if n > len(pool):
            raise ValueError(
                f"Cannot build a {n}-subject subset: the parent's {cfg.train_split!r} split has only "
                f"{len(pool)} subjects."
            )

    members = tuple(
        _build_member(
            resolved,
            cfg,
            out_root,
            store,
            n,
            ranked,
            index_dir=None if index_dir is None else Path(index_dir),
            task_name=task_name,
            workers=workers,
            worker=worker,
        )
        for n in cfg.n_subjects
    )
    manifest = FamilyManifest(
        out_root=str(out_root),
        parent=str(resolved.root),
        parent_ids={
            "subject_set_id": resolved.subject_set_id,
            "train_set_id": combine_subject_digests(_cached_subject_digests(resolved, cfg.train_split)),
            "fingerprint": resolved.fingerprint,
        },
        config=cfg.to_dict(),
        members=members,
        disk=_disk_report(out_root, store, members),
    )
    _publish(lambda staged: atomic_write_json(manifest.to_dict(), staged), out_root / FAMILY_MANIFEST_NAME)
    return manifest
