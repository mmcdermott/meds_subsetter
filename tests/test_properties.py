"""Property-based tests for the invariants the whole package rests on.

The doctests in each module show these properties on hand-picked examples, which is what makes them
readable. These tests try to *break* them instead, over thousands of generated inputs. They are here
rather than in a docstring because that is exactly the trade-off ``CONTRIBUTORS.md`` describes: a
hypothesis strategy in a docstring would obscure the example it is meant to illustrate.

Four invariants are load-bearing, in the sense that a silent violation would corrupt a scaling-law
sweep rather than crash it:

1. **Nesting.** A smaller subset is a strict prefix of a larger one, so the members of a family are
   comparable rather than independent draws.
2. **Shard sharing.** With a fixed shard size, a smaller member's full shards are the same subject
   tuples, under the same keys, as the larger member's -- which is what lets them be one file.
3. **Digest invariance.** A content id depends on content alone: not on row order, not on how rows
   were distributed across shards, not on process or thread count.
4. **Draw reproducibility.** A bootstrap replicate is a pure function of (pool, size, salt, index).
"""

from __future__ import annotations

import polars as pl
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from meds_subsetter.digests import combine_subject_digests, pinned_schema, subject_digests
from meds_subsetter.selection import (
    assign_shards,
    draw_with_replacement,
    draw_without_replacement,
    rank_subjects,
    select_nested,
    uniform_score,
)

#: Subject ids as MEDS defines them: non-null int64. Negative and very large values are included
#: because nothing in the standard forbids them and the rank key must not care.
subject_ids = st.integers(min_value=-(2**62), max_value=2**62)

#: A pool is a *set* of subjects: `rank_subjects` rejects duplicates by design, so generating them
#: here would only test that rejection, which a doctest already covers.
pools = st.lists(subject_ids, min_size=1, max_size=200, unique=True)

salts = st.text(min_size=1, max_size=32)


@given(pool=pools, salt=salts, data=st.data())
@settings(max_examples=300, deadline=None)
def test_selection_is_nested(pool: list[int], salt: str, data: st.DataObject) -> None:
    """A smaller selection is a strict prefix of a larger one, for every pool, salt and pair of sizes."""
    n_small = data.draw(st.integers(min_value=0, max_value=len(pool)))
    n_large = data.draw(st.integers(min_value=n_small, max_value=len(pool)))
    small = select_nested(pool, n_small, salt)
    large = select_nested(pool, n_large, salt)
    assert large[:n_small] == small


@given(pool=pools, salt=salts)
@settings(max_examples=200, deadline=None)
def test_selection_is_order_independent(pool: list[int], salt: str) -> None:
    """The rank order depends on the subject set, not on the order the pool was listed in.

    A pool comes from `subject_splits.parquet` or from a glob, and neither promises an order.
    """
    assert rank_subjects(pool, salt) == rank_subjects(list(reversed(pool)), salt)


@given(pool=pools, salt=salts)
@settings(max_examples=200, deadline=None)
def test_selection_preserves_the_pool(pool: list[int], salt: str) -> None:
    """Ranking is a permutation: nobody is dropped, nobody is invented, nobody is duplicated."""
    ranked = rank_subjects(pool, salt)
    assert sorted(ranked) == sorted(pool)


@given(subject=subject_ids, salt=salts)
@settings(max_examples=500, deadline=None)
def test_uniform_score_is_a_probability(subject: int, salt: str) -> None:
    """The reported score is always in [0, 1), including at the top of the digest range."""
    score = uniform_score(subject, salt)
    assert 0.0 <= score < 1.0


@given(pool=pools, salt=salts, data=st.data())
@settings(max_examples=300, deadline=None)
def test_full_shards_are_shared_across_members(pool: list[int], salt: str, data: st.DataObject) -> None:
    """Every *full* shard of a smaller member is the larger member's shard: same subjects, same key.

    This is the disk-sharing claim, stated as arithmetic. Only the final, partial shard of the smaller
    member may differ, and only when its size is not a multiple of the shard size.
    """
    per_shard = data.draw(st.integers(min_value=1, max_value=16))
    n_small = data.draw(st.integers(min_value=0, max_value=len(pool)))
    n_large = data.draw(st.integers(min_value=n_small, max_value=len(pool)))

    small = assign_shards(select_nested(pool, n_small, salt), per_shard)
    large = assign_shards(select_nested(pool, n_large, salt), per_shard)

    full = [s for s in small if s.n_subjects == per_shard]
    assert len(large) >= len(full)
    for spec in full:
        counterpart = large[spec.index]
        assert counterpart.subject_ids == spec.subject_ids
        assert counterpart.key == spec.key


@given(pool=pools, salt=salts, per_shard=st.integers(min_value=1, max_value=16))
@settings(max_examples=200, deadline=None)
def test_shards_partition_the_selection(pool: list[int], salt: str, per_shard: int) -> None:
    """Shards are disjoint, cover the selection exactly, and none exceeds the configured size.

    A subject in two shards would violate the MEDS one-subject-one-file invariant that every scan,
    count and digest in this package relies on.
    """
    ranked = rank_subjects(pool, salt)
    specs = assign_shards(ranked, per_shard)
    flat = [s for spec in specs for s in spec.subject_ids]
    assert flat == ranked
    assert len(set(flat)) == len(flat)
    assert all(0 < spec.n_subjects <= per_shard for spec in specs)


@given(
    n_subjects=st.integers(min_value=1, max_value=8),
    n_rows=st.integers(min_value=1, max_value=6),
    n_shards=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**16),
)
@settings(max_examples=200, deadline=None)
def test_digest_is_invariant_to_sharding_and_row_order(
    n_subjects: int, n_rows: int, n_shards: int, seed: int
) -> None:
    """Resharding a dataset, and shuffling rows within a shard, cannot change the dataset id.

    This is what makes an id comparable between a parent and a resharded subset of it, and it is the
    property the whole `train_set_id` story depends on.

    Shard boundaries fall *between subjects*, because that is what MEDS requires -- "data about a
    single subject cannot be split across parquet files" -- and it is the assumption every per-shard
    computation in this package is entitled to make. A frame that splits one subject over two shards
    is not a resharding of this dataset; it is a different, invalid one.
    """
    rows = [
        (sid, float(sid * 10 + r), f"C{r % 3}") for sid in range(1, n_subjects + 1) for r in range(n_rows)
    ]
    frame = pl.DataFrame(
        rows, schema={"subject_id": pl.Int64, "numeric_value": pl.Float32, "code": pl.String}, orient="row"
    )
    schema = frame.schema
    whole = combine_subject_digests(subject_digests(frame.lazy(), schema=schema))

    # Partition by subject, then deal whole subjects round-robin into shards and shuffle each shard.
    by_subject = {sid: frame.filter(pl.col("subject_id") == sid) for sid in range(1, n_subjects + 1)}
    buckets: list[list[pl.DataFrame]] = [[] for _ in range(n_shards)]
    for i, sid in enumerate(sorted(by_subject)):
        buckets[i % n_shards].append(by_subject[sid])
    shards = [
        pl.concat(b).sample(fraction=1.0, shuffle=True, seed=seed + i) for i, b in enumerate(buckets) if b
    ]
    per_shard = pl.concat([subject_digests(s.lazy(), schema=schema) for s in shards])
    assert combine_subject_digests(per_shard) == whole


@given(
    n_subjects=st.integers(min_value=1, max_value=6),
    bump=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_digest_detects_any_changed_value(n_subjects: int, bump: int) -> None:
    """Changing one numeric value anywhere changes the dataset id.

    A content hash that missed this would let two genuinely different training sets report the same
    `train_set_id`, which is precisely the confusion the ids exist to prevent.
    """
    rows = [(sid, float(sid)) for sid in range(1, n_subjects + 1)]
    frame = pl.DataFrame(rows, schema={"subject_id": pl.Int64, "numeric_value": pl.Float32}, orient="row")
    schema = frame.schema
    target = bump % n_subjects + 1
    changed = frame.with_columns(
        pl.when(pl.col("subject_id") == target)
        .then(pl.col("numeric_value") + 1.0)
        .otherwise(pl.col("numeric_value"))
        .cast(pl.Float32)
        .alias("numeric_value")
    )
    a = combine_subject_digests(subject_digests(frame.lazy(), schema=schema))
    b = combine_subject_digests(subject_digests(changed.lazy(), schema=schema))
    assert a != b


@given(pool=pools, salt=salts, data=st.data())
@settings(max_examples=200, deadline=None)
def test_bootstrap_draws_are_reproducible_and_nested(pool: list[int], salt: str, data: st.DataObject) -> None:
    """A replicate is a pure function of its inputs, and lengthening it only appends.

    Reproducibility is what lets a manifest record a seed instead of a subject list; the prefix
    property is what lets a bigger evaluation set be a superset of a smaller one.
    """
    n_small = data.draw(st.integers(min_value=0, max_value=40))
    n_large = data.draw(st.integers(min_value=n_small, max_value=n_small + 40))
    replicate = data.draw(st.integers(min_value=0, max_value=64))

    small = draw_with_replacement(pool, n_small, salt, replicate)
    assert small == draw_with_replacement(pool, n_small, salt, replicate)
    assert draw_with_replacement(pool, n_large, salt, replicate)[:n_small] == small
    assert set(small) <= set(pool)
    assert len(small) == n_small


@given(pool=pools, salt=salts, data=st.data())
@settings(max_examples=200, deadline=None)
def test_replicates_are_separated(pool: list[int], salt: str, data: st.DataObject) -> None:
    """Two replicates of a non-degenerate pool are not the same draw.

    A pool of one subject can only ever produce one draw, so it is excluded rather than special-cased.
    """
    assume(len(pool) > 4)
    size = data.draw(st.integers(min_value=5, max_value=min(40, len(pool))))
    a = draw_with_replacement(pool, size, salt, 0)
    b = draw_with_replacement(pool, size, salt, 1)
    assert a != b


@given(pool=pools, salt=salts, data=st.data())
@settings(max_examples=200, deadline=None)
def test_draws_without_replacement_never_repeat(pool: list[int], salt: str, data: st.DataObject) -> None:
    """Sampling without replacement yields distinct subjects drawn from the pool, and nests in size."""
    n_small = data.draw(st.integers(min_value=0, max_value=len(pool)))
    n_large = data.draw(st.integers(min_value=n_small, max_value=len(pool)))
    replicate = data.draw(st.integers(min_value=0, max_value=16))

    small = draw_without_replacement(pool, n_small, salt, replicate)
    large = draw_without_replacement(pool, n_large, salt, replicate)
    assert len(set(small)) == len(small)
    assert set(small) <= set(pool)
    assert large[:n_small] == small


@given(
    widths=st.lists(st.integers(min_value=2, max_value=4), min_size=1, max_size=4),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_pinned_schema_is_the_union(widths: list[int], tmp_path_factory) -> None:
    """The pinned schema is the union of the shards' columns, whatever order they sort in.

    MEDS makes `numeric_value` optional, so shards legitimately differ in width. Inferring the schema
    from whichever shard sorts first -- which is what a multi-file `scan_parquet` does -- would drop
    real columns out of every id computed from it.
    """
    columns = ["subject_id", "code", "numeric_value", "text_value"]
    root = tmp_path_factory.mktemp("shards")
    for i, width in enumerate(widths):
        frame = pl.DataFrame(
            {
                "subject_id": [i],
                "code": ["A"],
                "numeric_value": pl.Series([1.0], dtype=pl.Float32),
                "text_value": ["t"],
            }
        ).select(columns[:width])
        frame.write_parquet(root / f"{i:03d}.parquet")

    schema = pinned_schema(sorted(root.glob("*.parquet")))
    assert set(schema) == set(columns[: max(widths)])
