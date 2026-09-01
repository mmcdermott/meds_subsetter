"""Test set-up and fixtures code."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

#: Subjects of :func:`tiny_tensorized_cohort`, by shard. Chosen to mirror ``simple_static_MEDS``'s
#: train/tuning/held_out split and its 2-subjects-per-train-shard layout, so doctests can move between
#: the raw and tensorized paths without renumbering anything.
_COHORT = {
    "train/0": (239684, 1195293),
    "train/1": (68729, 814703),
    "tuning/0": (754281,),
    "held_out/0": (1500733,),
}


def _write_tensorized_shard(root: Path, shard: str, subject_ids: tuple[int, ...]) -> None:
    """Write one ``tokenization/schemas`` parquet and its matching ``data`` ``.nrt``.

    The layout is MEDS-Torch-Data's, verified against a real ``MTD_preprocess`` run: the schemas
    parquet carries the static data and the per-subject event structure, the ``.nrt`` carries the
    dynamic tensors, and the two are joined *positionally* -- schemas row ``i`` is NRT index ``i``.
    """
    n_events = [2 + (i % 3) for i, _ in enumerate(subject_ids)]
    time_delta_days = [[float(e) for e in range(n)] for n in n_events]
    measurements_per_event = [[1 + (e % 2) for e in range(n)] for n in n_events]
    code = [[[10 + e + m for m in range(mpe)] for e, mpe in enumerate(mps)] for mps in measurements_per_event]
    numeric_value = [[[float(m) for m in range(mpe)] for mpe in mps] for mps in measurements_per_event]

    schemas = pl.DataFrame(
        {
            "subject_id": list(subject_ids),
            "static_code": [[1, 2] for _ in subject_ids],
            "static_numeric_value": [[0.0, 1.0] for _ in subject_ids],
            "start_time": [datetime(2020, 1, 1) for _ in subject_ids],
            "time": [[datetime(2020, 1, 1 + e) for e in range(n)] for n in n_events],
            "measurements_per_event": measurements_per_event,
        },
        schema={
            "subject_id": pl.Int64,
            "static_code": pl.List(pl.UInt8),
            "static_numeric_value": pl.List(pl.Float32),
            "start_time": pl.Datetime("us"),
            "time": pl.List(pl.Datetime("us")),
            "measurements_per_event": pl.List(pl.UInt32),
        },
    )
    schema_fp = root / "tokenization" / "schemas" / f"{shard}.parquet"
    schema_fp.parent.mkdir(parents=True, exist_ok=True)
    schemas.write_parquet(schema_fp)

    nrt_fp = root / "data" / f"{shard}.nrt"
    nrt_fp.parent.mkdir(parents=True, exist_ok=True)
    JointNestedRaggedTensorDict(
        {"time_delta_days": time_delta_days, "code": code, "numeric_value": numeric_value},
        schema={"time_delta_days": np.float32, "code": np.int64, "numeric_value": np.float32},
    ).save(nrt_fp)


@pytest.fixture(scope="session")
def tiny_tensorized_cohort(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal MEDS-Torch-Data tensorized cohort: 6 subjects over 4 shards, plus code metadata.

    Building one with the real ``MTD_preprocess`` would pull in torch and the whole preprocessing
    pipeline for what is, on disk, three kinds of file. The layout written here was checked against a
    real ``MTD_preprocess`` output, and it is the layout ``meds_torchdata`` actually reads:
    ``tokenization/schemas/<split>/<n>.parquet``, ``data/<split>/<n>.nrt``, and ``metadata/codes.parquet``.
    """
    root = tmp_path_factory.mktemp("tensorized_cohort")
    for shard, subject_ids in _COHORT.items():
        _write_tensorized_shard(root, shard, subject_ids)
    codes = pl.DataFrame(
        {
            "code": [f"C{i}" for i in range(1, 16)],
            "code/vocab_index": list(range(1, 16)),
            "description": [f"code {i}" for i in range(1, 16)],
        }
    )
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    codes.write_parquet(root / "metadata" / "codes.parquet")
    return root


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    simple_static_MEDS: Path,
    simple_static_MEDS_dataset_with_task: Path,
    tiny_tensorized_cohort: Path,
):
    """Pre-populate the doctest namespace.

    ``yaml_disk``, ``print_directory``, and ``PrintConfig`` register themselves via their own pytest
    plugins, so they are not bound here. The MEDS dataset fixtures come from ``meds_testing_helpers``;
    binding them into the namespace lets doctests read as ``subset_meds(simple_static_MEDS, ...)``
    rather than carrying dataset-construction boilerplate.
    """
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
            "json": json,
            "np": np,
            "pl": pl,
            "pa": pa,
            "simple_static_MEDS": simple_static_MEDS,
            "simple_static_MEDS_dataset_with_task": simple_static_MEDS_dataset_with_task,
            "tiny_tensorized_cohort": tiny_tensorized_cohort,
        }
    )
