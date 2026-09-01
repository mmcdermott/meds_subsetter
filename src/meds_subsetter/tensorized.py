r"""Subsetting a MEDS-Torch-Data (MTD) *tensorized* cohort, and the pipeline that avoids needing to.

There are two routes from a parent cohort to a tensorized subset, and they are good for different
things:

1. **Subset the raw MEDS dataset, then tensorize it** with :func:`pinned_pipeline`. This is the route
   to prefer. It produces a cohort that is byte-identical to what the parent's own tokenization would
   have produced for those subjects, because the pipeline it emits pins the vocabulary and the
   normalization statistics to the parent's. It costs one preprocessing run per subset.
2. **Subset an already-tensorized cohort**, with :func:`build_tensorized_family`. No preprocessing
   run at all -- the ``.nrt`` tensors are sliced and restacked directly -- so a whole sweep is minutes
   rather than hours. Use it when the parent cohort already exists.

Both keep the parent's ``metadata/codes.parquet`` **verbatim**; on this path that is not a policy
choice. ``MEDSTorchDataConfig.vocab_size`` is ``max(code/vocab_index) + 1``, so dropping even the
lexicographically-last unused code changes the model's input dimension, and a refit permutes every
index. :class:`~meds_subsetter.config.CodeMetadataPolicy` is therefore ignored here, and a config that
asks to prune is refused rather than quietly overridden.

The layout this module reads and writes, verified against a real ``MTD_preprocess`` run::

    <cohort>/tokenization/schemas/<split>/<n>.parquet   subject_id, static_code, static_numeric_value,
                                                        start_time, time, measurements_per_event
    <cohort>/data/<split>/<n>.nrt                       HuggingFace safetensors
    <cohort>/metadata/codes.parquet                     global; carries code/vocab_index

A tensorized cohort has no ``subject_splits.parquet`` and no ``dataset.json``: split membership is the
shard path prefix and nothing else. And **subject identity is positional** -- there is no
``subject_id`` in the ``.nrt``; schemas row ``i`` is NRT index ``i``. That alignment is checked on
every shard read rather than assumed, because a subject whose rows are all static (null-time) lands in
the schemas frame but not in the ``.nrt`` at all, which silently shifts every subsequent subject's
tensors onto the wrong subject.

``nested_ragged_tensors`` is an optional dependency; install ``meds-subsetter[torch]`` to use this
module.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import meds
import polars as pl

from . import __package_name__, __version__
from .config import CodeMetadataPolicy, NonTrainPolicy, SubsetConfig
from .digests import combine_digests, digest, frame_digest, short_id, subject_set_digest
from .materialize import atomic_write_json, dir_size, place, unique_inode_size
from .selection import assign_shards, rank_subjects, select_nested
from .sizes import TENSORIZED_SCHEMAS_SUBDIR, size_tensorized
from .subset import (
    FAMILY_MANIFEST_NAME,
    MEMBER_MANIFEST_DIRNAME,
    MEMBER_MANIFEST_NAME,
    SHARD_STORE_DIRNAME,
    NotAMEDSDatasetError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

#: Where the tensors live, relative to the cohort root.
TENSORIZED_DATA_SUBDIR = "data"

#: Suffix of a tensor shard.
NRT_SUFFIX = ".nrt"

_SUBJECT_ID = "subject_id"
_ROW_INDEX = "__row_index__"
_SHARD = "__shard__"


def _require_nrt():
    """Import ``nested_ragged_tensors`` lazily, with an error that names the extra to install.

    Returns:
        The ``JointNestedRaggedTensorDict`` class.

    Raises:
        ImportError: If ``nested_ragged_tensors`` is not installed.
    """
    try:
        from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
    except ImportError as e:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "Subsetting a tensorized cohort needs `nested_ragged_tensors`, which is an optional "
            "dependency of this package. Install it with `pip install 'meds-subsetter[torch]'`."
        ) from e
    return JointNestedRaggedTensorDict


def is_tensorized_cohort(root: Path) -> bool:
    """Return whether ``root`` looks like a MEDS-Torch-Data tensorized cohort.

    The test is the presence of ``tokenization/schemas``, which is the one directory a tensorized
    cohort has and a raw MEDS root does not. It is what ``--kind auto`` dispatches on.

    Args:
        root: The directory to test.

    Returns:
        ``True`` if ``root`` carries a tokenization schemas directory.

    Examples:
        >>> is_tensorized_cohort(tiny_tensorized_cohort)
        True
        >>> is_tensorized_cohort(simple_static_MEDS)
        False
    """
    return (root / TENSORIZED_SCHEMAS_SUBDIR).is_dir()


def schema_path(root: Path, shard: str) -> Path:
    """Return the schemas parquet of ``shard``.

    Args:
        root: The cohort root.
        shard: The shard name, e.g. ``"train/0"``.

    Returns:
        ``<root>/tokenization/schemas/<shard>.parquet``.

    Examples:
        >>> schema_path(tiny_tensorized_cohort, "train/0").name
        '0.parquet'
    """
    return root / TENSORIZED_SCHEMAS_SUBDIR / f"{shard}.parquet"


def nrt_path(root: Path, shard: str) -> Path:
    """Return the tensor file of ``shard``.

    Args:
        root: The cohort root.
        shard: The shard name, e.g. ``"train/0"``.

    Returns:
        ``<root>/data/<shard>.nrt``.

    Examples:
        >>> nrt_path(tiny_tensorized_cohort, "train/0").name
        '0.nrt'
    """
    return root / TENSORIZED_DATA_SUBDIR / f"{shard}{NRT_SUFFIX}"


def tensorized_shards(root: Path) -> list[str]:
    """Return the cohort's shard names, sorted, skipping dot-directories.

    Args:
        root: The cohort root.

    Returns:
        Shard names relative to the schemas directory, suffix stripped.

    Raises:
        NotAMEDSDatasetError: If there is no schemas directory, or it holds no shards.

    Examples:
        >>> tensorized_shards(tiny_tensorized_cohort)
        ['held_out/0', 'train/0', 'train/1', 'tuning/0']

        A raw MEDS root is refused by name rather than reported as empty:

        >>> tensorized_shards(simple_static_MEDS)
        Traceback (most recent call last):
            ...
        meds_subsetter.subset.NotAMEDSDatasetError: No tensorized schemas under
        /...tokenization/schemas
    """
    base = root / TENSORIZED_SCHEMAS_SUBDIR
    if not base.is_dir():
        raise NotAMEDSDatasetError(f"No tensorized schemas under {base}")
    names = sorted(
        p.relative_to(base).with_suffix("").as_posix()
        for p in base.rglob("*.parquet")
        if not any(part.startswith(".") for part in p.relative_to(base).parts)
    )
    if not names:
        raise NotAMEDSDatasetError(f"No tensorized schemas under {base}")
    return names


def subject_locations(root: Path) -> pl.DataFrame:
    """Map every subject of a tensorized cohort to the shard and row that holds it.

    The row index is the whole point: it is simultaneously the schemas-frame row and the NRT index,
    and this function is where that alignment is *checked* rather than assumed.

    Args:
        root: The cohort root.

    Returns:
        A frame of ``subject_id``, ``__shard__``, ``__row_index__``, sorted by shard then row.

    Raises:
        NotAMEDSDatasetError: If a shard has no tensor file, if its schemas frame and tensors
            disagree on how many subjects it holds, or if a subject appears in two shards.

    Examples:
        >>> locs = subject_locations(tiny_tensorized_cohort)
        >>> locs.sort("subject_id")
        shape: (6, 3)
        ┌────────────┬────────────┬───────────────┐
        │ subject_id ┆ __shard__  ┆ __row_index__ │
        │ ---        ┆ ---        ┆ ---           │
        │ i64        ┆ str        ┆ u32           │
        ╞════════════╪════════════╪═══════════════╡
        │ 68729      ┆ train/1    ┆ 0             │
        │ 239684     ┆ train/0    ┆ 0             │
        │ 754281     ┆ tuning/0   ┆ 0             │
        │ 814703     ┆ train/1    ┆ 1             │
        │ 1195293    ┆ train/0    ┆ 1             │
        │ 1500733    ┆ held_out/0 ┆ 0             │
        └────────────┴────────────┴───────────────┘
    """
    jnrt = _require_nrt()
    frames = []
    for shard in tensorized_shards(root):
        tensors_fp = nrt_path(root, shard)
        if not tensors_fp.is_file():
            raise NotAMEDSDatasetError(
                f"No tensor file at {tensors_fp} for schema shard {schema_path(root, shard)}"
            )
        ids = pl.read_parquet(schema_path(root, shard), columns=[_SUBJECT_ID], glob=False)
        n_tensors = len(jnrt(tensors_fp=tensors_fp))
        if n_tensors != ids.height:
            raise NotAMEDSDatasetError(
                f"Shard {shard!r} of {root} has {ids.height} rows in its schemas frame but "
                f"{n_tensors} subjects in {tensors_fp.name}. Subject identity in a tensorized cohort "
                "is positional, so the two must agree; a subject whose rows are all static is the "
                "usual cause, since tokenization drops it from the tensors but not from the schemas."
            )
        frames.append(
            ids.with_columns(
                pl.lit(shard, dtype=pl.String).alias(_SHARD),
                pl.int_range(pl.len(), dtype=pl.UInt32).alias(_ROW_INDEX),
            )
        )
    out = pl.concat(frames).select(_SUBJECT_ID, _SHARD, _ROW_INDEX)
    if out[_SUBJECT_ID].n_unique() != out.height:
        dupes = out.group_by(_SUBJECT_ID).len().filter(pl.col("len") > 1)[_SUBJECT_ID].sort().to_list()[:5]
        raise NotAMEDSDatasetError(f"Subjects appear in more than one shard of {root}: {dupes}")
    return out


def tensorized_splits(root: Path) -> dict[str, tuple[int, ...]]:
    """Derive split membership from shard-name prefixes.

    A tensorized cohort carries no ``subject_splits.parquet``; the shard path prefix is the only
    record of which split a subject is in, and it is what ``meds_torchdata`` itself filters on.

    Args:
        root: The cohort root.

    Returns:
        Split label to sorted subject ids.

    Raises:
        NotAMEDSDatasetError: If any shard name carries no split prefix, since "the train split" is
            then undefined.

    Examples:
        >>> {k: len(v) for k, v in sorted(tensorized_splits(tiny_tensorized_cohort).items())}
        {'held_out': 1, 'train': 4, 'tuning': 1}
        >>> tensorized_splits(tiny_tensorized_cohort)["train"]
        (68729, 239684, 814703, 1195293)
    """
    locs = subject_locations(root)
    flat = [s for s in locs[_SHARD].unique().to_list() if "/" not in s]
    if flat:
        raise NotAMEDSDatasetError(
            f"Shards {sorted(flat)[:5]} of {root} carry no split prefix, so the cohort's splits "
            "cannot be determined. A tensorized cohort records splits only in its shard paths."
        )
    # The split label is everything before the last path component: MEDS split names may nest
    # (`taskA/held_out` is legal), so this is a strip-the-shard-index, not a take-the-first-part.
    labelled = locs.with_columns(pl.col(_SHARD).str.replace(r"/[^/]*$", "").alias("split"))
    grouped = labelled.group_by("split").agg(pl.col(_SUBJECT_ID).unique().sort()).sort("split")
    return {row[0]: tuple(row[1]) for row in grouped.iter_rows()}


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TensorizedMember:
    """One member of a tensorized subset family.

    Attributes:
        name: The member's directory name under the family root.
        root: Its absolute path.
        n_subjects: How many train subjects it holds.
        subject_set_id: Digest of its full subject set, across all splits.
        train_set_id: Digest of its train subject set.
        shards: Member shard name to the store-relative key its files came from. Two members sharing
            an entry share the bytes on disk.
        sizes: :meth:`~meds_subsetter.sizes.SizeReport.to_dict` of the finished member.
    """

    name: str
    root: str
    n_subjects: int
    subject_set_id: str
    train_set_id: str
    shards: dict[str, str]
    sizes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready, sorted-key form.

        Returns:
            A plain dict.
        """
        return dict(sorted(dataclasses.asdict(self).items()))


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TensorizedFamily:
    """A whole family of tensorized subsets, as written to ``family.json``.

    Attributes:
        out_root: The family root.
        parent: The parent cohort's path.
        parent_ids: Its ``subject_set_id``, ``train_set_id`` and ``codes_id``.
        config: The resolved :class:`~meds_subsetter.config.SubsetConfig`.
        members: One entry per built member, in ascending size.
        disk: What the family cost and what it would have cost as independent copies, in the same
            shape :class:`~meds_subsetter.subset.FamilyManifest` uses.
        tool: The package that wrote it.
        version: That package's version.
    """

    out_root: str
    parent: str
    parent_ids: dict[str, str]
    config: dict[str, Any]
    members: tuple[TensorizedMember, ...]
    disk: dict[str, int]
    tool: str = __package_name__
    version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready, sorted-key form.

        Returns:
            A plain dict, with ``members`` as a list.
        """
        out = dict(sorted(dataclasses.asdict(self).items()))
        out["members"] = [m.to_dict() for m in self.members]
        return out


def _slice_shard(root: Path, plan: pl.DataFrame, out_schema: Path, out_nrt: Path) -> None:
    """Write one output shard by slicing the parent's schemas frames and tensors.

    ``plan`` is consumed in ``(__shard__, __row_index__)`` order rather than in rank order. Both are
    deterministic, but this one takes exactly one contiguous read per source shard, and -- because it depends
    only on the parent -- it gives the *same* bytes for a shard however many members of the family happen to
    contain it. That is what lets members share the file.
    """
    jnrt = _require_nrt()
    import numpy as np

    plan = plan.sort(_SHARD, _ROW_INDEX)
    schema_parts, tensor_parts = [], []
    for shard, rows in plan.group_by(_SHARD, maintain_order=True):
        name = shard[0]
        idx = rows[_ROW_INDEX].to_numpy()
        schema_parts.append(pl.read_parquet(schema_path(root, name), glob=False)[idx.tolist()])
        tensor_parts.append(jnrt(tensors_fp=nrt_path(root, name))[np.asarray(idx, dtype=np.int64)])

    out_schema.parent.mkdir(parents=True, exist_ok=True)
    tmp_schema = out_schema.with_name(out_schema.name + ".tmp")
    pl.concat(schema_parts).write_parquet(tmp_schema)
    tmp_schema.replace(out_schema)

    out_nrt.parent.mkdir(parents=True, exist_ok=True)
    tmp_nrt = out_nrt.with_name(out_nrt.name + ".tmp")
    tmp_nrt.unlink(missing_ok=True)
    combined = tensor_parts[0] if len(tensor_parts) == 1 else jnrt.concatenate(tensor_parts)
    combined.save(tmp_nrt)
    tmp_nrt.replace(out_nrt)


def build_tensorized_family(
    parent: Path,
    out_root: Path,
    cfg: SubsetConfig,
    *,
    do_overwrite: bool = False,
) -> TensorizedFamily:
    r"""Build a nested family of tensorized subsets, sharing shard files between the members.

    The selection, the rank-bucketed sharding, and therefore the disk sharing are exactly the raw
    path's: subjects are ranked by a salted digest, a member of size ``N`` takes the first ``N``, and
    shard ``k`` holds ranks ``[k * n_subjects_per_shard, (k + 1) * n_subjects_per_shard)``. So every
    *full* shard is identical across every member, and a smaller member's shards are literally the
    larger member's files.

    Non-train splits follow ``cfg.non_train_policy``; under the default ``passthrough`` they are linked
    straight from the parent, so the evaluation set is fixed across the size axis.

    Args:
        parent: The parent tensorized cohort.
        out_root: Where to build the family. ``family.json`` is written here.
        cfg: The family's configuration. ``code_metadata`` must be ``copy``.
        do_overwrite: Rebuild members that already exist rather than reusing them.

    Returns:
        The family manifest, also written to ``<out_root>/family.json``.

    Raises:
        NotAMEDSDatasetError: If ``parent`` is not a usable tensorized cohort.
        ValueError: If ``cfg.code_metadata`` is ``prune``, or a requested size exceeds the parent's
            train split.
        ImportError: If ``nested_ragged_tensors`` is not installed.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "sweep"
        >>> cfg = SubsetConfig(
        ...     n_subjects=(2, 4), n_subjects_per_shard=2, code_metadata="copy", link_mode="symlink"
        ... )
        >>> family = build_tensorized_family(tiny_tensorized_cohort, out, cfg)
        >>> [(m.name, m.n_subjects) for m in family.members]
        [('N0000002', 2), ('N0000004', 4)]

        Each member is a loadable cohort: schemas, tensors and the parent's vocabulary.

        >>> sorted(p.name for p in (out / "N0000002").iterdir())
        ['.meds_subsetter', 'data', 'metadata', 'tokenization']
        >>> sorted(pl.read_parquet(out / "N0000002" / "tokenization/schemas/train/0.parquet")["subject_id"])
        [239684, 814703]

        The nesting and the sharing, demonstrated together: the small member's train shard *is* the
        large member's file, so the family costs one copy of it rather than two.

        >>> small = out / "N0000002" / "tokenization/schemas/train/0.parquet"
        >>> large = out / "N0000004" / "tokenization/schemas/train/0.parquet"
        >>> small.resolve() == large.resolve()
        True
        >>> (out / "N0000002" / "data/train/0.nrt").resolve() == (
        ...     out / "N0000004" / "data/train/0.nrt"
        ... ).resolve()
        True

        The train subjects nest, and the tensors travel with them -- the small member's tensors are
        the parent's tensors for those subjects, sliced, not recomputed:

        >>> from meds_subsetter.sizes import size_tensorized
        >>> rep = size_tensorized(out / "N0000002")
        >>> {s.split: (s.n_subjects, s.n_events) for s in rep.splits if s.split == "train"}
        {'train': (2, 5)}

        The vocabulary is the parent's, byte for byte, which is what keeps a sweep's members
        comparable:

        >>> parent_codes = (tiny_tensorized_cohort / "metadata" / "codes.parquet").read_bytes()
        >>> (out / "N0000002" / "metadata" / "codes.parquet").read_bytes() == parent_codes
        True

        Rebuilding is a no-op that returns the same manifest:

        >>> build_tensorized_family(tiny_tensorized_cohort, out, cfg) == family
        True

        Pruning the vocabulary is refused rather than silently ignored, because on this path it would
        change ``vocab_size`` and permute every index:

        >>> build_tensorized_family(
        ...     tiny_tensorized_cohort, out, SubsetConfig(n_subjects=(2,), code_metadata="prune")
        ... )
        Traceback (most recent call last):
            ...
        ValueError: A tensorized cohort's metadata/codes.parquet must be copied verbatim...
        >>> tmp.cleanup()
    """
    if cfg.code_metadata is not CodeMetadataPolicy.COPY:
        raise ValueError(
            "A tensorized cohort's metadata/codes.parquet must be copied verbatim, not pruned: "
            "MEDSTorchDataConfig.vocab_size is max(code/vocab_index) + 1, so dropping any code "
            "changes the model's input dimension and a refit permutes every index. Pass "
            "code_metadata='copy' (--code-metadata copy)."
        )

    locs = subject_locations(parent)
    splits = tensorized_splits(parent)
    if cfg.train_split not in splits:
        raise NotAMEDSDatasetError(
            f"{parent} has no {cfg.train_split!r} split; it has {sorted(splits)}. "
            "Split membership in a tensorized cohort comes from the shard path prefixes."
        )
    train_pool = list(splits[cfg.train_split])
    ranked = rank_subjects(train_pool, cfg.salt)
    biggest = cfg.n_subjects[-1]
    if biggest > len(ranked):
        raise ValueError(
            f"Cannot take {biggest} subjects from the {cfg.train_split!r} split of {parent}, "
            f"which has {len(ranked)}."
        )

    store = out_root / SHARD_STORE_DIRNAME
    salt_key = short_id(digest([cfg.salt]), 8)
    codes_src = parent / meds.code_metadata_filepath
    codes_id = frame_digest(pl.scan_parquet(codes_src, glob=False)) if codes_src.is_file() else digest([])

    members: list[TensorizedMember] = []
    for n in cfg.n_subjects:
        selected = select_nested(ranked, n, cfg.salt)
        member_root = out_root / cfg.subset_name(n)
        shards: dict[str, str] = {}

        for spec in assign_shards(selected, cfg.n_subjects_per_shard):
            key = f"train/{salt_key}/{spec.key}"
            stored_schema = store / f"{key}.parquet"
            stored_nrt = store / f"{key}{NRT_SUFFIX}"
            if do_overwrite or not (stored_schema.is_file() and stored_nrt.is_file()):
                plan = locs.filter(pl.col(_SUBJECT_ID).is_in(list(spec.subject_ids)))
                _slice_shard(parent, plan, stored_schema, stored_nrt)
            name = f"{cfg.train_split}/{spec.index}"
            place(stored_schema, schema_path(member_root, name), cfg.link_mode)
            place(stored_nrt, nrt_path(member_root, name), cfg.link_mode)
            shards[name] = key

        shards |= _place_non_train(parent, member_root, store, locs, splits, cfg, do_overwrite)
        place(codes_src, member_root / meds.code_metadata_filepath, cfg.link_mode)

        member_subjects = sorted(_member_subjects(member_root))
        member = TensorizedMember(
            name=cfg.subset_name(n),
            root=str(member_root),
            n_subjects=n,
            subject_set_id=subject_set_digest(member_subjects),
            train_set_id=subject_set_digest(selected),
            shards=dict(sorted(shards.items())),
            sizes=size_tensorized(member_root).to_dict(),
        )
        atomic_write_json(member.to_dict(), member_root / MEMBER_MANIFEST_DIRNAME / MEMBER_MANIFEST_NAME)
        members.append(member)

    family = TensorizedFamily(
        out_root=str(out_root),
        parent=str(parent),
        parent_ids={
            "subject_set_id": subject_set_digest(locs[_SUBJECT_ID].to_list()),
            "train_set_id": subject_set_digest(train_pool),
            "codes_id": codes_id,
        },
        config=cfg.to_dict(),
        members=tuple(members),
        disk=_disk_report(out_root, members),
    )
    atomic_write_json(family.to_dict(), out_root / FAMILY_MANIFEST_NAME)
    return family


def _disk_report(out_root: Path, members: Sequence[TensorizedMember]) -> dict[str, int]:
    """Measure what the family cost, and what it would have cost as independent cohorts.

    ``n_bytes_on_disk`` charges each inode once, so a hardlinked or symlinked shard is paid for
    exactly as often as it really is; ``n_bytes_independent_copies`` resolves every link, which is
    what the same members would have cost standalone. The difference is the saving the shared store
    bought.

    Args:
        out_root: The family root.
        members: The built members.

    Returns:
        The same keys :func:`~meds_subsetter.subset.build_family` reports.
    """
    manifest = out_root / FAMILY_MANIFEST_NAME
    own = manifest.stat().st_size if manifest.is_file() else 0
    independent = sum(dir_size(Path(m.root), follow_links=True) for m in members)
    on_disk = unique_inode_size(out_root) - own
    store = out_root / SHARD_STORE_DIRNAME
    store_files = [p for p in store.rglob("*") if p.is_file()] if store.is_dir() else []
    return {
        "n_bytes_on_disk": on_disk,
        "n_bytes_apparent": dir_size(out_root) - own,
        "n_bytes_independent_copies": independent,
        "n_bytes_saved": independent - on_disk,
        "n_bytes_cache": 0,
        "n_store_files": len(store_files),
        "n_store_bytes": sum(p.stat().st_size for p in store_files),
    }


def _member_subjects(member_root: Path) -> list[int]:
    """Return every subject in a built member, across all of its splits."""
    return [
        sid
        for shard in tensorized_shards(member_root)
        for sid in pl.read_parquet(schema_path(member_root, shard), columns=[_SUBJECT_ID], glob=False)[
            _SUBJECT_ID
        ].to_list()
    ]


def _place_non_train(
    parent: Path,
    member_root: Path,
    store: Path,
    locs: pl.DataFrame,
    splits: Mapping[str, Sequence[int]],
    cfg: SubsetConfig,
    do_overwrite: bool,
) -> dict[str, str]:
    """Place the splits that are not being subsetted, per ``cfg.non_train_policy``.

    Returns:
        Member shard name to store key (or to the parent's shard name, for pass-through).
    """
    if cfg.non_train_policy is NonTrainPolicy.DROP:
        return {}

    salt_key = short_id(digest([cfg.salt]), 8)
    out: dict[str, str] = {}
    for split, subjects in sorted(splits.items()):
        if split == cfg.train_split:
            continue
        if cfg.non_train_policy is NonTrainPolicy.PASSTHROUGH:
            for shard in tensorized_shards(parent):
                if not shard.startswith(f"{split}/"):
                    continue
                place(schema_path(parent, shard), schema_path(member_root, shard), cfg.link_mode)
                place(nrt_path(parent, shard), nrt_path(member_root, shard), cfg.link_mode)
                out[shard] = f"parent:{shard}"
            continue

        ranked = rank_subjects(list(subjects), cfg.salt)
        n = min(len(ranked), cfg.n_subjects_per_shard * max(1, len(ranked) // cfg.n_subjects_per_shard))
        for spec in assign_shards(
            select_nested(ranked, n or len(ranked), cfg.salt), cfg.n_subjects_per_shard
        ):
            key = f"{split}/{salt_key}/{spec.key}"
            stored_schema = store / f"{key}.parquet"
            stored_nrt = store / f"{key}{NRT_SUFFIX}"
            if do_overwrite or not (stored_schema.is_file() and stored_nrt.is_file()):
                plan = locs.filter(pl.col(_SUBJECT_ID).is_in(list(spec.subject_ids)))
                _slice_shard(parent, plan, stored_schema, stored_nrt)
            name = f"{split}/{spec.index}"
            place(stored_schema, schema_path(member_root, name), cfg.link_mode)
            place(stored_nrt, nrt_path(member_root, name), cfg.link_mode)
            out[name] = key
    return out


def pinned_pipeline(raw_subset: Path, out_dir: Path, parent_cohort: Path) -> dict[str, Any]:
    r"""Build the MEDS-Transforms pipeline that tensorizes a raw subset *without* vocabulary drift.

    Running stock ``MTD_preprocess`` on a subset refits both the vocabulary and the normalization
    statistics from the subset's own train split. Measured on a 200-subject parent cut to 4 subjects:
    ``vocab_size`` fell from 49 to 25, **no** vocabulary index agreed with the parent's, and
    ``normalization``'s inner join silently dropped 29% of the subset's rows. The damage grows as the
    subset shrinks, so it is worst exactly where a scaling sweep is most sensitive.

    The pipeline returned here fixes that by dropping ``fit_normalization`` -- the stage that refits
    the statistics -- and pinning ``fit_vocabulary_indices`` at the parent's finished ``metadata/``.
    ``fit_vocabulary_indices`` is kept rather than also dropped so that it remains the last metadata
    stage, which is what makes MEDS-Transforms resolve its ``reducer_output_dir`` to
    ``${output_dir}/metadata`` -- where ``meds_torchdata`` looks for ``codes.parquet``. Re-running it
    over an already-indexed frame is idempotent.

    Args:
        raw_subset: The raw MEDS subset to tensorize, e.g. a member of a
            :func:`~meds_subsetter.subset.build_family` sweep.
        out_dir: Where the tensorized cohort should be written.
        parent_cohort: The parent's *tensorized* cohort, whose ``metadata/`` holds the vocabulary to
            pin to.

    Returns:
        The pipeline, ready to serialize as YAML.

    Examples:
        >>> pipeline = pinned_pipeline(Path("/d/N0000100"), Path("/d/N0000100_tensorized"), Path("/d/full"))
        >>> pipeline["stages"]
        [{'fit_vocabulary_indices': {'metadata_input_dir': '/d/full/metadata'}},
         'normalization', 'tokenization', 'tensorization']

        ``fit_normalization`` is deliberately absent -- its presence is the bug:

        >>> any("fit_normalization" in str(s) for s in pipeline["stages"])
        False
    """
    return {
        "input_dir": str(raw_subset),
        "output_dir": str(out_dir),
        "etl_metadata": {"pipeline_name": "tensorization"},
        "stages": [
            {"fit_vocabulary_indices": {"metadata_input_dir": str(parent_cohort / "metadata")}},
            "normalization",
            "tokenization",
            "tensorization",
        ],
    }


def write_pinned_pipeline(path: Path, raw_subset: Path, out_dir: Path, parent_cohort: Path) -> Path:
    r"""Write :func:`pinned_pipeline` to ``path`` as YAML and return it.

    Args:
        path: Where to write the pipeline YAML.
        raw_subset: The raw MEDS subset to tensorize.
        out_dir: Where the tensorized cohort should be written.
        parent_cohort: The parent tensorized cohort to pin the vocabulary to.

    Returns:
        ``path``.

    Examples:
        >>> import yaml
        >>> tmp = tempfile.TemporaryDirectory()
        >>> fp = write_pinned_pipeline(
        ...     Path(tmp.name) / "tensorize.yaml", Path("/d/N100"), Path("/d/N100_t"), Path("/d/full")
        ... )
        >>> loaded = yaml.safe_load(fp.read_text())
        >>> loaded["stages"][0]
        {'fit_vocabulary_indices': {'metadata_input_dir': '/d/full/metadata'}}

        Run it with ``MEDS_transform-pipeline <path>``. The comment at the top says why it is not
        just ``MTD_preprocess``:

        >>> print(fp.read_text().split("\n")[0])
        # Generated by meds_subsetter. Do NOT replace this with `MTD_preprocess`: that pipeline
        >>> tmp.cleanup()
    """
    import yaml

    header = (
        "# Generated by meds_subsetter. Do NOT replace this with `MTD_preprocess`: that pipeline\n"
        "# runs `fit_normalization` and refits `fit_vocabulary_indices` from this subset's own train\n"
        "# split, which permutes the vocabulary between members of a sweep and silently drops rows\n"
        "# whose codes the refit dropped. This pipeline pins both to the parent cohort instead.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        header + yaml.safe_dump(pinned_pipeline(raw_subset, out_dir, parent_cohort), sort_keys=False)
    )
    return path


def digest_of_ids(ids: Mapping[str, str]) -> str:
    """Combine a mapping of ids into one, order-insensitively.

    Args:
        ids: Named ids, e.g. per-split content ids.

    Returns:
        A single combined digest.

    Examples:
        >>> a = digest_of_ids({"train": digest(["1"]), "held_out": digest(["2"])})
        >>> a == digest_of_ids({"held_out": digest(["2"]), "train": digest(["1"])})
        True
    """
    return combine_digests(f"{k}={v}" for k, v in ids.items())
