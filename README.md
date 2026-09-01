# meds_subsetter

> [!CAUTION]
> **This repository is AI-generated.** Its design, implementation, tests, and documentation were
> produced by an AI agent (Claude Code) working from a specification, with automated adversarial
> review at each stage. Every claim it makes about the MEDS ecosystem was verified by running code
> against the real packages, and the test suite is extensive — but it has not had a human line-by-line
> review. Treat it as a starting point that needs your scrutiny before you rely on its numbers, not as
> reviewed research software.

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/meds-subsetter)](https://pypi.org/project/meds-subsetter/)
[![Tests](https://github.com/mmcdermott/meds_subsetter/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/meds_subsetter/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/mmcdermott/meds_subsetter/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/meds_subsetter/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/mmcdermott/meds_subsetter.svg)](https://github.com/mmcdermott/meds_subsetter/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/meds_subsetter/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

Cut a [MEDS](https://medical-event-data-standard.github.io/) dataset down to 100, 1 000, 10 000
subjects; measure exactly how big each cut is; and get a reproducible id for each one — so you can run
a scaling-law sweep, or a fixed-train variance study, without paying for the full dataset at every
point on the curve.

The subsets **nest** (the 100-subject train split is a subset of the 1 000-subject one) and **share
their shard files on disk**, so a whole sweep costs about as much as its largest member.

## What it does

|                 |                                                                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Subset**      | Take the first `N` subjects of a salted-hash ranking of the train split. Non-train splits pass through unchanged, so evaluation is fixed across the size axis.      |
| **Nest**        | Rank order is a property of the subject and the salt alone, so a smaller subset is a strict *prefix* of a larger one — not an independent draw.                     |
| **Reshard**     | Shard `k` holds ranks `[k·S, (k+1)·S)`. With `S` fixed, every full shard is byte-identical across the family, so members share the file rather than copying it.     |
| **Size**        | Exact counts of subjects, events, and measurements, before and after tokenization — the latter read from `.nrt` safetensors headers without touching a tensor byte. |
| **Fingerprint** | Order-insensitive, content-sensitive ids for a train split, a test split, or an index dataframe. Invariant to resharding; sensitive to a single changed value.      |
| **Resample**    | Bootstrap or subsample evaluation sets as *manifests* over the unmodified parent, so variance can be estimated without materializing N copies.                      |
| **Tensorized**  | All of the above directly on a [MEDS-Torch-Data](https://github.com/mmcdermott/meds-torch-data) cohort, by slicing its tensors — no preprocessing re-run.           |

It handles datasets with or without an index/task dataframe, and works either **before**
pre-processing (on the raw MEDS dataset) or **after** it (on the tensorized cohort).

## Installation

```bash
pip install meds-subsetter
pip install 'meds-subsetter[torch]' # to subset tensorized MEDS-Torch-Data cohorts
```

## Quick start

Build a nested family of subsets at 100 and 1 000 train subjects:

```bash
meds-subset /data/mimic /data/sweep -n 100 1000 --code-metadata copy
```

```text
Built 2 members from /data/mimic into /data/sweep
  N0000100  n_subjects=100   shards=3  train_set_id=db3c85926fe915f8
  N0001000  n_subjects=1000  shards=4  train_set_id=5f7fa61a7c2ce1f4
  disk: 33538 bytes, vs 40746 as independent copies (saved 7208); store 6 files, cache 6087 bytes
  manifest: /data/sweep/family.json
```

Each member is a complete, valid MEDS root. The sharing is visible in the tree:

```text
/data/sweep/
├── family.json
├── .shard_store/
│   └── train/6b0e7e5d/r0000000000-0000000100.parquet   ← one file, two members
├── N0000100/
│   └── data/train/0.parquet -> ../../../.shard_store/train/6b0e7e5d/r0000000000-0000000100.parquet
└── N0001000/
    └── data/train/0.parquet -> ../../../.shard_store/train/6b0e7e5d/r0000000000-0000000100.parquet
```

Then measure and fingerprint them:

```bash
meds-size /data/sweep/N0000100 -o sizes.parquet
meds-fingerprint /data/sweep/N0000100
```

And draw evaluation sets for a variance study:

```bash
meds-resample /data/mimic /data/sweep --n-resamples 20 --size 500 \
	--index-dir /data/labels/mortality --exclude-from /data/sweep/N0001000
```

### From Python

```python
>>> from pathlib import Path
>>> from meds_subsetter.config import SubsetConfig
>>> from meds_subsetter.subset import build_family
>>> tmp = tempfile.TemporaryDirectory()
>>> cfg = SubsetConfig(n_subjects=(2, 4), n_subjects_per_shard=2, code_metadata="copy")
>>> family = build_family(simple_static_MEDS, Path(tmp.name) / "sweep", cfg)
>>> [(m.name, m.n_subjects) for m in family.members]
[('N0000002', 2), ('N0000004', 4)]

```

The smaller member's train split is a subset of the larger one's, and its shard *is* the larger
member's file:

```python
>>> import polars as pl
>>> def train_subjects(member):
...     return set(pl.read_parquet(sorted((Path(member.root) / "data/train").glob("*.parquet")))["subject_id"])
>>> small, large = family.members
>>> train_subjects(small) < train_subjects(large)
True
>>> (Path(small.root) / "data/train/0.parquet").resolve() == (
...     Path(large.root) / "data/train/0.parquet"
... ).resolve()
True
>>> family.disk["n_bytes_saved"] > 0
True

```

Sizes and ids come back on the manifest, and `meds-fingerprint` recomputes them byte for byte:

```python
>>> from meds_subsetter.subset import fingerprint
>>> fingerprint(Path(large.root), "train")["train_set_id"] == large.train_set_id
True
>>> tmp.cleanup()

```

## The commands

| Command                    | What it does                                                                      |
| -------------------------- | --------------------------------------------------------------------------------- |
| `meds-subset PARENT OUT`   | Build a nested family of subsets. `--kind auto` picks the raw or tensorized path. |
| `meds-size ROOT`           | Report sizes as a tidy parquet plus a JSON sidecar, or a table on stdout.         |
| `meds-fingerprint ROOT`    | Report content ids as JSON.                                                       |
| `meds-resample PARENT OUT` | Draw resampled/bootstrap evaluation manifests.                                    |

All four are also reachable as `python -m meds_subsetter <subcommand>`. Each takes `--config` (a YAML
config file) with individual flags overriding it, returns `0` on success, `2` on a malformed command
line, and `3` on an input it cannot use — printing `error: …` with no traceback.

`meds_subsetter` additionally registers a `subset_subjects` **MEDS-Transforms stage**, so subject
filtering can be composed into an existing pipeline and parallelized with
`--multirun worker="range(0,N)"` like any other stage.

## How it works, and why

### Nesting is a total order, not a sample

A subject's rank is `sha256(salt ∥ subject_id)`, and the selection of size `N` is the first `N`
subjects in that order. Ties are impossible because the subject id is appended to the sort key, so
nesting holds unconditionally — a prefix of a totally ordered list is a prefix of every longer prefix.
The consequence is that two points on a scaling curve differ by *added subjects*, not by an
independent redraw, which is what makes the curve's error bars mean anything.

The one limitation, stated plainly: top-`N` selection is nested in `N` but not stable if the candidate
*pool* changes. A family is defined against one fixed parent, and `build_family` refuses to extend a
family whose parent's fingerprint has moved unless you pass `--force`.

### Sharing falls out of the sharding

Shard `k` holds ranks `[k·S, (k+1)·S)`, so for a fixed `S` the shard is a function of the salt and `k`
alone — not of how big the member is. Every *full* shard is therefore identical across every member,
and only the final partial shard of each member is unique. Members reference a shared store by
symlink (the default), hardlink, or copy.

This is why the package does not use MEDS-Transforms' `reshard_to_split`: that stage permutes subjects
with `np.random` and packs them with `np.array_split`, so shard membership depends on the whole
cohort's size. Measured on a 200-subject parent and its 100-subject prefix, **zero** shards were
shared. Its shared RNG also means shrinking `train` silently re-permutes `tuning` and `held_out`.

### Counting

`n_events` counts distinct `(subject_id, time)` pairs **with non-null time**. This matches the
MEDS-Torch-Data tokenizer and the `.nrt` on disk exactly. The sibling `meds_summary_stats` counts each
subject's static rows as one additional event; that convention is recoverable as
`n_events + n_subjects_with_static`, and both quantities are reported.

There is deliberately **no single "post-tokenization token count"**. Truncation at `max_seq_len` loses
tokens, `STEP_THROUGH` windowing gains them (measured at 1.67× on a real cohort), and `RANDOM`
sampling makes the realized count a random variable. `meds-size` reports the config-free corpus
counts; `sizes.project_trained_tokens` gives the per-epoch trained-on count as an explicit function of
the batching config.

For a scaling-law fit, rename one column and hand the parquet over:

```python
>>> rows = pl.DataFrame({"train_set_id": ["a"], "n_subjects": [100], "n_events": [4200]})
>>> rows.rename({"n_subjects": "dataset_size__n_subjects"}).columns
['train_set_id', 'dataset_size__n_subjects', 'n_events']

```

Emit exactly one `dataset_size__*` column: under subject subsetting every size measure is
proportional, so two of them are collinear in log space and their exponents cannot be separated.

### Ids

A dataset's id is built from per-subject digests, combined order-insensitively. So it is invariant to
resharding — which the package itself performs — where a file-bytes digest would not be. Rows are
encoded with a length prefix per field rather than a delimiter, because `text_value` is free clinical
text and legitimately contains newlines and separators. Floats are hashed as their exact IEEE bits, so
no formatting choice can lose a distinction.

Two datasets with the same `train_set_id` were trained on the same content, however it was sharded.
Two with the same `subject_set_id` cover the same subjects, which is cheaper to check.

### Resamples are manifests, never datasets

A bootstrap draws the same subject more than once, and multiplicity cannot survive a MEDS root:
`SubjectSplitSchema` is closed so there is no weight column, duplicate schema rows corrupt
MEDS-Torch-Data's `end_event_index` (a 3-event subject was measured reporting 6), `reshard_to_split`
deduplicates draws away, and four independent consumers silently collapse duplicates.

So a resample is a directory of `subjects.parquet` and MEDS-Label-schema `labels/`, over the
*unmodified* parent. Duplicated label rows pass through MEDS-Torch-Data unharmed — `len(dataset)`
becomes the number of draws — which was verified end to end. Draws come from the train split and, by
default, exclude the training subset, because nothing in the ecosystem enforces train/test
disjointness for you.

### Metadata

> [!IMPORTANT]
> `--code-metadata prune` is the default for a raw subset, and it is a hazard for a *family*.
> Pruning makes the vocabulary a function of `N`, so a downstream `MTD_preprocess` refits
> `fit_vocabulary_indices` from each member's own `codes.parquet` and permutes the indices — worst at
> the small-`N` end, i.e. correlated with the very axis a sweep is measuring. Measured on a 200 → 4
> subject subset: `vocab_size` fell 49 → 25, **no** index agreed with the parent's, and
> `normalization`'s inner join silently dropped 29% of the rows. **Pass `--code-metadata copy` for any
> family you intend to compare across.**

On the tensorized path, copying is not a policy but a requirement, and pruning is refused:
`vocab_size` is `max(code/vocab_index) + 1`, so dropping even an unused code changes the model's input
dimension.

To go from a raw subset to a correctly-tensorized one, use the pinned pipeline rather than stock
`MTD_preprocess`:

```python
>>> from meds_subsetter.tensorized import pinned_pipeline
>>> pinned_pipeline(Path("/d/N0000100"), Path("/d/N0000100_t"), Path("/d/full"))["stages"]
[{'fit_vocabulary_indices': {'metadata_input_dir': '/d/full/metadata'}},
 'normalization', 'tokenization', 'tensorization']

```

`fit_normalization` is absent because it is what refits the statistics;
`fit_vocabulary_indices` is kept, pinned at the parent's metadata, so it stays the last metadata stage
and its output lands where MEDS-Torch-Data looks for `codes.parquet`.

## Scaling

The default path is one streaming pass over the parent's train split: `scan_parquet` → join the
subject-to-shard assignment → sort → `sink_parquet` through `pl.PartitionBy`. It never materializes a
split, and it produces every not-yet-stored shard of the largest member in that one pass; smaller
members then reuse those exact files.

`--workers N` switches to a two-phase scheme — fragment per input shard, then concatenate per output
shard — with each output guarded by MEDS-Transforms' `rwlock_wrap`, so independent processes (a Slurm
array, say) cooperate and reruns are cheap no-ops. Total I/O stays at 2× bytes regardless of worker
count.

## Development

```bash
git clone https://github.com/mmcdermott/meds_subsetter.git
cd meds_subsetter
uv sync
uv run pre-commit install
uv run pytest -v
```

Doctests are the primary test surface; `tests/` holds the hypothesis property tests and the
console-script integration tests. See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the full guide.

> [!WARNING]
> There is no folder in this repository for `data`. Datasets — public or private — must be stored
> outside the repository to avoid leaking sensitive data, bloating the repo, or over-fitting the
> project to a single data resource. The same applies to API keys, tokens, and credentials. Note
> that anything committed to `git` history can be recovered through the published repository even
> if removed from the current tree or pushed to a non-main branch; if something sensitive is
> accidentally committed, you need to
> [purge it from history](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Related packages

- [MEDS](https://github.com/Medical-Event-Data-Standard/meds) — the data standard.
- [MEDS-Transforms](https://github.com/mmcdermott/MEDS_transforms) — the pipeline framework this
    package registers a stage with and borrows its locking from.
- [MEDS-Torch-Data](https://github.com/mmcdermott/meds-torch-data) — the tensorized view this package
    can subset directly.
- [meds_summary_stats](https://github.com/Medical-Event-Data-Standard/meds_summary_stats) — dataset
    statistics and ETL regression testing; the digest conventions here match its.
