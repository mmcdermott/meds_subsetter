"""End-to-end tests of the installed console scripts, run as real subprocesses.

The doctests exercise each ``*_main`` in-process, which is fast and readable but proves nothing about
the console-script entry points declared in ``pyproject.toml``, about exit codes as a shell sees them,
or about the four commands agreeing with each other. That is what these cover: a full workflow --
subset, size, fingerprint, resample -- against a dataset written to disk, checking the artifacts and
cross-checking every number against an independent computation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from meds_testing_helpers.dataset import MEDSDataset
from meds_testing_helpers.static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT_WITH_TASKS

#: Console scripts are installed next to the interpreter running the tests, so the tests exercise the
#: same tree they were collected from rather than whatever happens to be on ``PATH``.
_BIN = Path(sys.executable).parent


def run(command: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run an installed console script and capture its output."""
    return subprocess.run([str(_BIN / command), *args], capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def parent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real MEDS root on disk, with task labels: 6 subjects, 4 of them in ``train``."""
    root = tmp_path_factory.mktemp("parent") / "meds"
    MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT_WITH_TASKS).write(root)
    return root


@pytest.fixture(scope="module")
def index_dir(parent: Path) -> Path:
    """The parent's task labels, in the layout ``meds_testing_helpers`` writes them."""
    return parent / "task_labels" / "boolean_value_task"


def test_console_scripts_are_installed() -> None:
    """Every declared entry point exists and answers ``--help`` without a traceback."""
    for command in ("meds-subset", "meds-resample", "meds-size", "meds-fingerprint"):
        result = run(command, "--help")
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("usage:")
        assert "Traceback" not in result.stderr


def test_module_dispatch_matches_console_scripts(parent: Path) -> None:
    """``python -m meds_subsetter <sub>`` is the same command as the console script."""
    via_module = subprocess.run(
        [sys.executable, "-m", "meds_subsetter", "size", str(parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    via_script = run("meds-size", str(parent))
    assert via_module.returncode == via_script.returncode == 0
    assert via_module.stdout == via_script.stdout


def test_unknown_subcommand_is_a_usage_error() -> None:
    """An unknown subcommand exits 2 with usage on stderr, leaving stdout clean for pipelines."""
    result = subprocess.run(
        [sys.executable, "-m", "meds_subsetter", "frobnicate"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "unknown subcommand" in result.stderr


def test_missing_dataset_is_an_input_error_not_a_traceback(tmp_path: Path) -> None:
    """A path that is not a MEDS root exits 3 with a one-line message and no traceback."""
    result = run("meds-size", str(tmp_path / "nope"))
    assert result.returncode == 3
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr


def test_full_workflow(parent: Path, index_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Subset, size, fingerprint and resample, cross-checking every claim against the raw data."""
    out = tmp_path_factory.mktemp("workflow") / "sweep"
    built = run(
        "meds-subset",
        str(parent),
        str(out),
        "-n",
        "2",
        "4",
        "--n-subjects-per-shard",
        "2",
        "--code-metadata",
        "copy",
        "--index-dir",
        str(index_dir),
    )
    assert built.returncode == 0, built.stderr

    family = json.loads((out / "family.json").read_text())
    assert [m["n_subjects"] for m in family["members"]] == [2, 4]

    small, large = out / "N0000002", out / "N0000004"

    # (a) Each member is a valid MEDS root, and its splits file keeps the physical column order
    # MEDS-Transforms unpacks positionally.
    for member in (small, large):
        splits = pl.read_parquet(member / "metadata" / "subject_splits.parquet")
        assert splits.columns == ["subject_id", "split"]
        assert (member / "metadata" / "dataset.json").is_file()

    # (b) The selection nests, and only train was subsetted.
    def train_subjects(member: Path) -> set[int]:
        frame = pl.read_parquet(sorted((member / "data" / "train").glob("*.parquet")))
        return set(frame["subject_id"].to_list())

    assert train_subjects(small) < train_subjects(large)
    assert len(train_subjects(small)) == 2
    assert len(train_subjects(large)) == 4

    # (c) The full shards are literally the same file, which is the disk-sharing claim.
    assert (small / "data/train/0.parquet").resolve() == (large / "data/train/0.parquet").resolve()
    assert family["disk"]["n_bytes_saved"] > 0

    # (d) No subject is in two shards, and the non-train splits are the parent's bytes.
    shards = [pl.read_parquet(p) for p in sorted((large / "data/train").glob("*.parquet"))]
    seen: set[int] = set()
    for shard in shards:
        ids = set(shard["subject_id"].to_list())
        assert not (ids & seen)
        seen |= ids
    for split in ("tuning", "held_out"):
        for shard in sorted((large / "data" / split).glob("*.parquet")):
            original = parent / "data" / split / shard.name
            assert shard.read_bytes() == original.read_bytes()

    # (e) Labels were resharded to match the data shards, one file per data shard.
    label_shards = sorted(p.name for p in (large / "tasks/boolean_value_task/train").glob("*.parquet"))
    data_shards = sorted(p.name for p in (large / "data/train").glob("*.parquet"))
    assert label_shards == data_shards

    # (f) `meds-size` agrees with an independent count of the same files.
    sized = run("meds-size", str(large), "-o", str(out / "sizes.parquet"))
    assert sized.returncode == 0, sized.stderr
    frame = pl.read_parquet(out / "sizes.parquet")
    reported = frame.filter(pl.col("split") == "train").row(0, named=True)

    raw = pl.read_parquet(sorted((large / "data/train").glob("*.parquet")))
    assert reported["n_subjects"] == raw["subject_id"].n_unique()
    assert reported["n_measurements"] == raw.height
    timed = raw.filter(pl.col("time").is_not_null())
    assert reported["n_events"] == timed.select("subject_id", "time").unique().height
    assert reported["n_static_measurements"] == raw.height - timed.height

    # (g) `meds-fingerprint` reports the same train_set_id the family recorded.
    printed = run("meds-fingerprint", str(large))
    assert printed.returncode == 0, printed.stderr
    ids = json.loads(printed.stdout)
    recorded = next(m for m in family["members"] if m["name"] == "N0000004")
    assert ids["dataset"]["train_set_id"] == recorded["train_set_id"]

    # (h) Resampling draws an evaluation set disjoint from the training subset by default.
    boots = out / "boots"
    drawn = run(
        "meds-resample",
        str(parent),
        str(boots),
        "--n-resamples",
        "3",
        "--size",
        "2",
        "--index-dir",
        str(index_dir),
        "--exclude-from",
        str(small),
    )
    assert drawn.returncode == 0, drawn.stderr

    replicates = sorted(p for p in (boots / "resamples").iterdir() if p.is_dir())
    assert len(replicates) == 3
    for replicate in replicates:
        subjects = pl.read_parquet(replicate / "subjects.parquet")
        assert subjects.height == 2
        assert not (set(subjects["subject_id"].to_list()) & train_subjects(small))
        # Multiplicity reaches the labels: a subject drawn k times contributes k copies of its rows.
        labels = pl.read_parquet(sorted((replicate / "labels").rglob("*.parquet")))
        parent_labels = pl.read_parquet(sorted(index_dir.glob("*.parquet")))
        for subject_id, drawn_times in subjects.group_by("subject_id").len().iter_rows():
            expected = parent_labels.filter(pl.col("subject_id") == subject_id).height * drawn_times
            assert labels.filter(pl.col("subject_id") == subject_id).height == expected

    # (i) Replicates are distinct but each is reproducible.
    drawn_ids = [tuple(pl.read_parquet(r / "subjects.parquet")["subject_id"].to_list()) for r in replicates]
    assert len(set(drawn_ids)) > 1
    again = run(
        "meds-resample",
        str(parent),
        str(out / "boots2"),
        "--n-resamples",
        "3",
        "--size",
        "2",
        "--index-dir",
        str(index_dir),
        "--exclude-from",
        str(small),
    )
    assert again.returncode == 0, again.stderr
    repeated = [
        tuple(pl.read_parquet(r / "subjects.parquet")["subject_id"].to_list())
        for r in sorted(p for p in (out / "boots2" / "resamples").iterdir() if p.is_dir())
    ]
    assert repeated == drawn_ids


def test_rebuilding_a_family_is_a_no_op(parent: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A second identical build rewrites nothing: same manifest, same inodes, same mtimes."""
    out = tmp_path_factory.mktemp("idempotent") / "sweep"
    args = (str(parent), str(out), "-n", "2", "--n-subjects-per-shard", "2", "--code-metadata", "copy")
    assert run("meds-subset", *args).returncode == 0

    shard = out / ".shard_store"
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in sorted(shard.rglob("*")) if p.is_file()}
    manifest_before = (out / "family.json").read_text()

    assert run("meds-subset", *args).returncode == 0
    after = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in sorted(shard.rglob("*")) if p.is_file()}
    assert after == before
    assert (out / "family.json").read_text() == manifest_before


def test_subsets_of_the_same_size_agree_across_families(
    parent: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Two independently built families over one parent pick the same subjects and report one id.

    This is what makes ``train_set_id`` usable as a join key across separately-run experiments.
    """
    base = tmp_path_factory.mktemp("families")
    args = ("-n", "2", "--n-subjects-per-shard", "2", "--code-metadata", "copy")
    assert run("meds-subset", str(parent), str(base / "a"), *args).returncode == 0
    assert run("meds-subset", str(parent), str(base / "b"), *args).returncode == 0

    def train_set_id(root: Path) -> str:
        return json.loads((root / "family.json").read_text())["members"][0]["train_set_id"]

    assert train_set_id(base / "a") == train_set_id(base / "b")


def test_tensorized_subset_tensors_match_the_parent(
    tiny_tensorized_cohort: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A tensorized subset's tensors are the parent's, sliced -- not recomputed and not reordered.

    Subject identity in a ``.nrt`` is positional, so an off-by-one in the slice would silently pair
    every subject with another subject's tensors. Checking bit-identity per subject is the only test
    that would catch that; a shape or count check would not.
    """
    import numpy as np
    from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

    from meds_subsetter.tensorized import nrt_path, schema_path, subject_locations

    out = tmp_path_factory.mktemp("tensorized") / "sweep"
    built = run(
        "meds-subset",
        str(tiny_tensorized_cohort),
        str(out),
        "-n",
        "2",
        "4",
        "--n-subjects-per-shard",
        "2",
        "--code-metadata",
        "copy",
    )
    assert built.returncode == 0, built.stderr

    parent_locations = subject_locations(tiny_tensorized_cohort)
    for member in ("N0000002", "N0000004"):
        root = out / member
        for subject_id, shard, row in subject_locations(root).iter_rows():
            _, parent_shard, parent_row = parent_locations.filter(pl.col("subject_id") == subject_id).row(0)
            mine = JointNestedRaggedTensorDict(tensors_fp=nrt_path(root, shard))[np.array([row])]
            theirs = JointNestedRaggedTensorDict(tensors_fp=nrt_path(tiny_tensorized_cohort, parent_shard))[
                np.array([parent_row])
            ]
            assert mine.equals(theirs), f"tensors differ for subject {subject_id} in {member}"
            assert pl.read_parquet(schema_path(root, shard))[[row]].equals(
                pl.read_parquet(schema_path(tiny_tensorized_cohort, parent_shard))[[parent_row]]
            )

    # The vocabulary is the parent's, which is what keeps the members' models comparable.
    parent_codes = (tiny_tensorized_cohort / "metadata" / "codes.parquet").read_bytes()
    for member in ("N0000002", "N0000004"):
        assert (out / member / "metadata" / "codes.parquet").read_bytes() == parent_codes
