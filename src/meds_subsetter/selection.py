r"""Pure, I/O-free subject selection, shard assignment, and resampling draws.

Everything in this module is a deterministic function of its arguments: no filesystem, no clock, no
process-global randomness, and deliberately no ``polars``. That is what makes the guarantees below
checkable by inspection, and it is why the rest of the package can treat these functions as a fixed
point while it changes freely around them.

Two properties are load-bearing for the whole package:

* **Nesting.** :func:`select_nested` orders subjects by a salted SHA-256 digest and takes a prefix, so
  a smaller subset of a family is always a literal prefix of a larger one.
* **Rank-bucketed sharding.** :func:`assign_shards` buckets by *rank*, which is a property of the
  subject and the salt alone -- never of the cohort size. Every full shard is therefore byte-identical
  across every member of a family, and a nested subset costs one partial shard plus links.

Examples:
    >>> pool = list(range(100, 140))
    >>> select_nested(pool, 8, "sweep") == select_nested(pool, 20, "sweep")[:8]
    True
    >>> [s.n_subjects for s in assign_shards(select_nested(pool, 25, "sweep"), 10)]
    [10, 10, 5]
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Field separator inside hashed strings. ``\x1f`` is ASCII UNIT SEPARATOR, which cannot occur in a
#: decimal subject id, so the *last* separator in ``f"{salt}{SUBJECT_SEP}{subject_id}"`` always splits
#: the two fields. The encoding is therefore injective for every salt, including one containing
#: ``\x1f`` itself, and no salt/id pair can collide with a different one.
SUBJECT_SEP = "\x1f"

#: Width of the counter/uniform draw taken from each digest, in bytes.
_DRAW_BYTES = 8

#: Exclusive upper bound of an ``_DRAW_BYTES``-wide unsigned draw.
_DRAW_LIMIT = 2 ** (8 * _DRAW_BYTES)

#: Bits of a draw kept in a uniform score: the width of a float64 significand, so that the quotient
#: below is exact. Dividing the *full* 64-bit draw by ``2**64`` would not be --- it is a
#: correctly-rounded division, so every draw at or above ``2**64 - 1024`` rounds up to exactly ``1.0``
#: and escapes the documented half-open range. This is how :func:`random.random` is defined too.
_SCORE_BITS = 53

#: Exclusive upper bound of a ``_SCORE_BITS``-wide draw, i.e. the uniform score's divisor.
_SCORE_LIMIT = 2**_SCORE_BITS


def _require_unique(subject_ids: Iterable[int]) -> list[int]:
    """Materialize ``subject_ids`` as a list, rejecting repeats.

    A repeated subject id would be selected or drawn twice, silently inflating a subset's size while
    leaving its subject *set* unchanged, so it is an error rather than something to deduplicate.

    Args:
        subject_ids: The ids to check. Consumed exactly once.

    Returns:
        The ids as a list, in the order given.

    Raises:
        ValueError: If any id appears more than once, naming the first repeat.

    Examples:
        >>> _require_unique([3, 1, 2])
        [3, 1, 2]
        >>> _require_unique(iter([1]))
        [1]
        >>> _require_unique([])
        []
        >>> _require_unique([1, 2, 1])
        Traceback (most recent call last):
            ...
        ValueError: Duplicate subject_id 1
    """
    ids = list(subject_ids)
    seen: set[int] = set()
    for subject_id in ids:
        if subject_id in seen:
            raise ValueError(f"Duplicate subject_id {subject_id}")
        seen.add(subject_id)
    return ids


def rank_digest(subject_id: int, salt: str) -> bytes:
    r"""Return the salted SHA-256 digest that fixes a subject's position in the rank order.

    This is *the* primitive the entire nesting guarantee rests on. Every subset id, shard key, and
    family manifest in this package is downstream of these 32 bytes. **Changing this function --- the
    hash, the separator, the field order, or the encoding --- invalidates every subset family that has
    ever been built, because previously emitted subsets would no longer be prefixes of the orders it
    produces.** Treat it as part of the on-disk format.

    ``hashlib`` is used rather than the builtin :func:`hash` (salted per process by ``PYTHONHASHSEED``)
    or polars' ``hash`` (documented as unstable across versions).

    Args:
        subject_id: The subject id to rank.
        salt: The family salt. Distinct salts give unrelated, independent orders.

    Returns:
        The raw 32-byte SHA-256 digest of ``f"{salt}{SUBJECT_SEP}{subject_id}"`` encoded as UTF-8.

    Examples:
        >>> rank_digest(1, "s").hex()
        'e2ce58fb211682f2bdd2d43c58c87d1313bff93e728a88d0ac901e1c0795ddcc'
        >>> len(rank_digest(1, "s"))
        32

        It is a pure function, so it is stable across calls, processes, and machines:

        >>> rank_digest(1, "s") == rank_digest(1, "s")
        True

        Different salts give unrelated digests, which is how two independent families are drawn from
        one dataset:

        >>> rank_digest(1, "s") == rank_digest(1, "trial-2026")
        False

        The separator keeps the two fields unambiguous, so a salt ending in digits cannot masquerade
        as part of a longer subject id:

        >>> rank_digest(23, "trial") == rank_digest(3, "trial2")
        False
    """
    return hashlib.sha256(f"{salt}{SUBJECT_SEP}{subject_id}".encode()).digest()


def rank_key(subject_id: int, salt: str) -> tuple[bytes, int]:
    """Return the total-order sort key for a subject: ``(rank_digest(...), subject_id)``.

    The trailing ``subject_id`` is not decoration. Sorting on the digest alone would be a *partial*
    order: two subjects whose digests collided would compare equal, and their relative order would
    then be an accident of the sort's input order --- which differs between a 100-subject and a
    1000-subject selection, breaking nesting exactly when a family is extended. Appending the
    (unique) subject id makes the order **total**, so ties are impossible by construction and nesting
    holds unconditionally: a prefix of a totally-ordered list is a prefix of every longer prefix.

    Args:
        subject_id: The subject id to rank.
        salt: The family salt.

    Returns:
        A tuple that sorts subjects into rank order.

    Examples:
        >>> key = rank_key(7, "salt")
        >>> key[0].hex()[:8], key[1]
        ('3eb6cbf4', 7)
        >>> key[0] == rank_digest(7, "salt")
        True

        Sorting by the key is exactly what :func:`rank_subjects` does:

        >>> sorted([rank_key(i, "demo") for i in (101, 102, 103)]) == [
        ...     rank_key(i, "demo") for i in (102, 103, 101)
        ... ]
        True

        No two distinct subjects ever compare equal, whatever their digests:

        >>> rank_key(1, "s") == rank_key(2, "s")
        False

        A real SHA-256 collision cannot be exhibited, so the tie-break is shown on synthetic keys
        that share a digest. Two things make it work: tuple comparison falls through to the second
        field, and that field is unique across subjects --- so the fallback always reaches a decision
        and always reaches the *same* one, whatever order the sort saw its input in:

        >>> d = rank_digest(1, "s")
        >>> [key[1] for key in sorted([(d, 9), (d, 2), (d, 5)])]
        [2, 5, 9]
        >>> len({rank_key(i, "s")[1] for i in range(100)})
        100
    """
    return (rank_digest(subject_id, salt), subject_id)


def _uniform_from_digest(digest: bytes) -> float:
    r"""Map a digest's leading bytes to a uniform score in ``[0, 1)``.

    Split out from :func:`uniform_score` so the top of the range --- unreachable from any subject id
    one could write down in a doctest --- can be exercised directly.

    Args:
        digest: The digest to score. Only its leading ``_DRAW_BYTES`` bytes are read.

    Returns:
        ``(int.from_bytes(digest[:8], "big") >> 11) / 2**53``, in ``[0, 1)``.

    Examples:
        >>> _uniform_from_digest(bytes(32))
        0.0
        >>> _uniform_from_digest(b"\xff" * 32)
        0.9999999999999999

        The upper end is open even for the largest digest SHA-256 can emit, which is the whole
        reason only 53 bits are kept --- the naive full-width division rounds it up to ``1.0``:

        >>> _uniform_from_digest(b"\xff" * 32) < 1.0
        True
        >>> int.from_bytes(b"\xff" * 8, "big") / 2**64
        1.0

        It is monotone in the draw, so it preserves the rank order of the bytes it reads:

        >>> scores = [_uniform_from_digest(bytes([b]) + bytes(31)) for b in (0, 1, 128, 255)]
        >>> scores == sorted(scores)
        True
    """
    draw = int.from_bytes(digest[:_DRAW_BYTES], "big")
    return (draw >> (8 * _DRAW_BYTES - _SCORE_BITS)) / _SCORE_LIMIT


def uniform_score(subject_id: int, salt: str) -> float:
    r"""Return the subject's rank digest as a uniform score in ``[0, 1)``.

    **For reporting only.** Selection and sharding must go through :func:`rank_key`, which is a total
    order; this score keeps only 53 bits of the leading 8 bytes and so is not tie-free. It exists so a
    manifest can say "this subset is everything below 0.13" and so a histogram of scores can show
    that a salt is behaving.

    Args:
        subject_id: The subject id to score.
        salt: The family salt.

    Returns:
        ``(int.from_bytes(rank_digest(subject_id, salt)[:8], "big") >> 11) / 2**53``, in ``[0, 1)``.
        The 11-bit shift keeps the draw inside a float64 significand, which is what makes the upper
        end genuinely open; see :data:`_SCORE_BITS`.

    Examples:
        >>> round(uniform_score(1, "s"), 6)
        0.885961
        >>> round(uniform_score(2, "s"), 6)
        0.306495

        The score is always a legal probability, *including* at the very top of the digest space.
        Sampled ids only ever exercise the middle of the range, so the boundary is pinned directly
        on the largest digest SHA-256 could ever emit:

        >>> all(0.0 <= uniform_score(i, "s") < 1.0 for i in range(2000))
        True
        >>> _uniform_from_digest(b"\xff" * 32) < 1.0
        True

        Because it is monotone in the digest's leading bytes it agrees with the rank order (barring a
        collision in those bytes), which is what makes it a fair thing to report:

        >>> pool = [101, 102, 103, 104, 105]
        >>> sorted(pool, key=lambda s: uniform_score(s, "demo")) == rank_subjects(pool, "demo")
        True
        >>> [round(uniform_score(s, "demo"), 3) for s in rank_subjects(pool, "demo")]
        [0.379, 0.405, 0.432, 0.773, 0.967]
    """
    return _uniform_from_digest(rank_digest(subject_id, salt))


def rank_subjects(subject_ids: Iterable[int], salt: str) -> list[int]:
    """Return ``subject_ids`` sorted ascending by :func:`rank_key`.

    The result is a function of the *set* of ids and the salt only --- never of the input order or the
    pool's size --- which is what lets a family's members share shards.

    Args:
        subject_ids: The candidate pool. Consumed exactly once.
        salt: The family salt.

    Returns:
        The ids in ascending rank order.

    Raises:
        ValueError: If any subject id appears more than once.

    Examples:
        >>> pool = [101, 102, 103, 104, 105]
        >>> rank_subjects(pool, "demo")
        [102, 105, 104, 103, 101]

        Input order is irrelevant --- the result depends only on the *set* of ids:

        >>> rank_subjects([105, 101, 104, 102, 103], "demo") == rank_subjects(pool, "demo")
        True

        A different salt gives an unrelated order:

        >>> rank_subjects(pool, "other")
        [104, 101, 105, 103, 102]

        Degenerate pools are fine:

        >>> rank_subjects([], "demo")
        []
        >>> rank_subjects([42], "demo")
        [42]

        Duplicates are an error, and the offender is named:

        >>> rank_subjects([101, 102, 101], "demo")
        Traceback (most recent call last):
            ...
        ValueError: Duplicate subject_id 101
    """
    return sorted(_require_unique(subject_ids), key=lambda subject_id: rank_key(subject_id, salt))


def select_nested(subject_ids: Iterable[int], n: int, salt: str) -> list[int]:
    """Select ``n`` subjects from ``subject_ids`` such that smaller selections nest inside larger ones.

    This is requirement (c): for one fixed pool and salt, ``select_nested(pool, a, salt)`` is a literal
    prefix of ``select_nested(pool, b, salt)`` whenever ``a <= b``. A scaling sweep can therefore add a
    larger point without disturbing any smaller one.

    Known limitation, documented rather than fixed: top-``n`` is nested in ``n`` but **not** stable if
    the candidate pool itself changes. A family is defined against one fixed parent, and a changed pool
    is made detectable elsewhere by binding the parent's ``subject_set_id`` into the family manifest.

    Args:
        subject_ids: The candidate pool. Consumed exactly once.
        n: How many subjects to select. Must be ``>= 0`` and at most the pool size.
        salt: The family salt.

    Returns:
        The first ``n`` subjects in rank order.

    Raises:
        ValueError: If ``n`` is negative, if ``n`` exceeds the pool size, or if the pool has duplicates.

    Examples:
        >>> pool = [101, 102, 103, 104, 105]
        >>> select_nested(pool, 2, "demo")
        [102, 105]
        >>> select_nested(pool, 4, "demo")
        [102, 105, 104, 103]

        Nesting holds for every pair of sizes, unconditionally:

        >>> sels = [select_nested(pool, n, "demo") for n in range(6)]
        >>> all(sels[a] == sels[b][:a] for a in range(6) for b in range(a, 6))
        True

        Boundaries:

        >>> select_nested(pool, 0, "demo")
        []
        >>> select_nested(pool, 5, "demo") == rank_subjects(pool, "demo")
        True
        >>> select_nested([], 0, "demo")
        []
        >>> select_nested([42], 1, "demo")
        [42]

        Asking for more than the pool holds is an error, not a short answer: silently returning fewer
        subjects than requested would corrupt a scaling sweep's x-axis.

        >>> select_nested(pool, 6, "demo")
        Traceback (most recent call last):
            ...
        ValueError: Cannot select 6 subjects from a pool of 5
        >>> select_nested(pool, -1, "demo")
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 0; got -1
        >>> select_nested([101, 101], 1, "demo")
        Traceback (most recent call last):
            ...
        ValueError: Duplicate subject_id 101
    """
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    ranked = rank_subjects(subject_ids, salt)
    if n > len(ranked):
        raise ValueError(f"Cannot select {n} subjects from a pool of {len(ranked)}")
    return ranked[:n]


@dataclass(frozen=True, slots=True)
class ShardSpec:
    """One output shard: a contiguous, half-open range of the rank order and the subjects in it.

    Attributes:
        index: The shard's position within its selection, ``0``-based. Becomes the ``<n>`` in a MEDS
            shard name such as ``train/0``.
        start: First rank in the shard, inclusive.
        stop: Last rank in the shard, exclusive.
        subject_ids: The subjects at ranks ``[start, stop)``, in rank order.

    Examples:
        >>> spec = ShardSpec(index=0, start=0, stop=3, subject_ids=(102, 105, 104))
        >>> spec.key
        'r0000000000-0000000003'
        >>> spec.n_subjects
        3

        The key is zero-padded so shard keys sort lexicographically in rank order:

        >>> keys = [ShardSpec(index=i, start=i * 5, stop=i * 5 + 5, subject_ids=(0,) * 5).key
        ...         for i in (0, 1, 2)]
        >>> keys == sorted(keys)
        True
        >>> keys[2]
        'r0000000010-0000000015'

        Specs are frozen, so a shard's identity cannot drift after it has been keyed:

        >>> spec.stop = 4
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'stop'

        The rank range and the subject list must agree --- the key claims the range, so a mismatch
        would make two different shards collide on one filename:

        >>> ShardSpec(index=0, start=0, stop=3, subject_ids=(1, 2))
        Traceback (most recent call last):
            ...
        ValueError: ShardSpec covers ranks [0, 3) but holds 2 subject ids
        >>> ShardSpec(index=0, start=5, stop=2, subject_ids=())
        Traceback (most recent call last):
            ...
        ValueError: ShardSpec covers ranks [5, 2) but holds 0 subject ids
    """

    index: int
    start: int
    stop: int
    subject_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check that the declared rank range matches the number of subjects carried.

        Raises:
            ValueError: If ``stop - start`` is not the number of subject ids.
        """
        if self.stop - self.start != len(self.subject_ids):
            raise ValueError(
                f"ShardSpec covers ranks [{self.start}, {self.stop}) but holds "
                f"{len(self.subject_ids)} subject ids"
            )

    @property
    def key(self) -> str:
        """The shard's content-addressable name: its rank range, zero-padded to 10 digits each.

        Returns:
            ``f"r{start:010d}-{stop:010d}"``.
        """
        return f"r{self.start:010d}-{self.stop:010d}"

    @property
    def n_subjects(self) -> int:
        """The number of subjects in the shard.

        Returns:
            ``len(self.subject_ids)``.
        """
        return len(self.subject_ids)


def assign_shards(ranked: Sequence[int], n_per_shard: int) -> list[ShardSpec]:
    """Cut a rank-ordered selection into contiguous rank buckets.

    Shard ``i`` covers ranks ``[i * n_per_shard, min((i + 1) * n_per_shard, len(ranked)))``. Because a
    subject's rank depends only on itself and the salt --- never on how many subjects were selected ---
    every **full** shard is identical across every member of a family. This is the whole disk-sharing
    story: the 100-subject subset's ``train/0`` *is* the 1000-subject subset's ``train/0``, so a family
    costs roughly its largest member plus one partial shard per member.

    Args:
        ranked: Subjects in rank order, as returned by :func:`select_nested` or :func:`rank_subjects`.
        n_per_shard: Subjects per shard. Must be ``>= 1``.

    Returns:
        One :class:`ShardSpec` per bucket, in rank order. Empty input gives an empty list.

    Raises:
        ValueError: If ``n_per_shard`` is less than 1.

    Examples:
        >>> pool = list(range(100, 140))
        >>> shards = assign_shards(select_nested(pool, 25, "s"), 10)
        >>> [s.key for s in shards]
        ['r0000000000-0000000010', 'r0000000010-0000000020', 'r0000000020-0000000025']
        >>> [(s.index, s.n_subjects) for s in shards]
        [(0, 10), (1, 10), (2, 5)]
        >>> shards[0].subject_ids
        (109, 135, 111, 130, 113, 118, 102, 105, 133, 117)

        Now the property that pays for the whole design. A smaller member of the same family produces
        the *same* leading shards --- identical keys and identical subject lists, hence identical
        bytes on disk --- and differs only in its final partial shard:

        >>> small = assign_shards(select_nested(pool, 12, "s"), 10)
        >>> [s.key for s in small]
        ['r0000000000-0000000010', 'r0000000010-0000000012']
        >>> small[0] == shards[0]
        True

        Even that trailing partial shard is a prefix of its counterpart in the larger member, so
        growing a family only ever appends:

        >>> small[-1].subject_ids == shards[1].subject_ids[:2]
        True

        Across a whole sweep, every full shard with a given key is byte-for-byte the same object:

        >>> ladder = [assign_shards(select_nested(pool, n, "s"), 10) for n in (5, 12, 25, 40)]
        >>> full = {s.key: s for member in ladder for s in member if s.n_subjects == 10}
        >>> all(s == full[s.key] for member in ladder for s in member if s.n_subjects == 10)
        True
        >>> sorted(full)
        ['r0000000000-0000000010', 'r0000000010-0000000020',
         'r0000000020-0000000030', 'r0000000030-0000000040']

        So the four family members need only seven distinct shard files between them, not ten:

        >>> len({s.key for member in ladder for s in member})
        7
        >>> sum(len(member) for member in ladder)
        10

        Boundaries:

        >>> assign_shards([], 10)
        []
        >>> assign_shards([7], 10)
        [ShardSpec(index=0, start=0, stop=1, subject_ids=(7,))]
        >>> [s.key for s in assign_shards([1, 2, 3], 1)]
        ['r0000000000-0000000001', 'r0000000001-0000000002', 'r0000000002-0000000003']
        >>> [s.n_subjects for s in assign_shards([1, 2, 3, 4], 2)]
        [2, 2]
        >>> assign_shards([1, 2, 3], 0)
        Traceback (most recent call last):
            ...
        ValueError: n_per_shard must be >= 1; got 0
    """
    if n_per_shard < 1:
        raise ValueError(f"n_per_shard must be >= 1; got {n_per_shard}")
    n = len(ranked)
    return [
        ShardSpec(
            index=index,
            start=start,
            stop=min(start + n_per_shard, n),
            subject_ids=tuple(ranked[start : start + n_per_shard]),
        )
        for index, start in enumerate(range(0, n, n_per_shard))
    ]


def _replicate_salt(salt: str, replicate: int) -> str:
    """Return the salt namespace for one resample replicate.

    Args:
        salt: The family salt.
        replicate: The replicate index.

    Returns:
        ``f"{salt}|resample|{replicate}"``.

    Examples:
        >>> _replicate_salt("s", 0)
        's|resample|0'
        >>> _replicate_salt("s", 0) == _replicate_salt("s", 1)
        False
    """
    return f"{salt}|resample|{replicate}"


def _rejection_cutoff(n: int) -> int:
    """Return the exclusive cutoff at or above which an 8-byte draw must be rejected.

    Equivalently: the largest multiple of ``n`` that does not exceed ``2**64``. When ``n`` divides
    ``2**64`` that is ``2**64`` itself --- one past the largest possible draw --- so nothing is ever
    rejected.

    Taking ``u % n`` over the full ``[0, 2**64)`` range would over-weight the first ``2**64 % n``
    elements of the pool. Rejecting draws at or above this cutoff removes that bias exactly; the
    rejection probability is ``(2**64 % n) / 2**64``, which for any realistic cohort is far below
    ``2**-40``, so the loop practically never iterates.

    Args:
        n: The pool size. Must be positive.

    Returns:
        ``2**64 - (2**64 % n)``.

    Examples:
        A pool whose size divides ``2**64`` needs no rejection at all:

        >>> _rejection_cutoff(4) == 2**64
        True

        Otherwise the tail that would bias the modulus is cut off:

        >>> _rejection_cutoff(3) == 2**64 - 1
        True
        >>> _rejection_cutoff(10) == 2**64 - 6
        True

        The cutoff is always an exact multiple of ``n``, which is what makes ``u % n`` uniform, and it
        is the *largest* such multiple that does not exceed ``2**64`` --- any smaller one would throw
        away draws for nothing, any larger one would let the bias back in:

        >>> all(_rejection_cutoff(n) % n == 0 for n in range(1, 200))
        True
        >>> all(_rejection_cutoff(n) <= 2**64 < _rejection_cutoff(n) + n for n in range(1, 200))
        True
    """
    return _DRAW_LIMIT - (_DRAW_LIMIT % n)


def draw_with_replacement(pool: Sequence[int], n_draws: int, salt: str, replicate: int) -> list[int]:
    r"""Draw ``n_draws`` subjects from ``pool`` uniformly at random *with* replacement.

    The randomness is a SHA-256 counter stream with rejection sampling:

    ``u = int.from_bytes(sha256(f"{salt}|resample|{replicate}\x1f{i}".encode()).digest()[:8], "big")``
    for ``i = 0, 1, 2, ...``; ``u`` is rejected (and ``i`` advanced) while ``u >= _rejection_cutoff(n)``,
    otherwise the draw is ``sorted(pool)[u % n]``.

    ``numpy.random`` is deliberately **not** used: NumPy promises value stability only for the legacy
    ``RandomState``, and ``Generator`` streams are explicitly allowed to change between NumPy versions.
    A bootstrap manifest has to be reproducible years later from the salt and replicate alone, so the
    stream is pinned to ``hashlib`` here, exactly as the rank order is.

    Because the stream depends only on ``(salt, replicate, i)``, the draws are prefix-nested in
    ``n_draws``: enlarging a bootstrap never disturbs the draws already made.

    Args:
        pool: The subjects to draw from. Order is irrelevant --- the draw indexes ``sorted(pool)``.
        n_draws: How many draws to take. Must be ``>= 0``.
        salt: The family salt.
        replicate: The replicate index; distinct replicates give independent streams.

    Returns:
        ``n_draws`` subject ids, in draw order, with repeats.

    Raises:
        ValueError: If ``n_draws`` is negative, if ``pool`` has duplicates, or if a positive number of
            draws is requested from an empty pool.

    Examples:
        >>> pool = [10, 20, 30, 40]
        >>> draws = draw_with_replacement(pool, 8, "s", 0)
        >>> draws
        [20, 10, 30, 20, 30, 30, 40, 40]

        Determinism --- the same arguments always give the same draws:

        >>> draw_with_replacement(pool, 8, "s", 0) == draws
        True

        Replicates are separated:

        >>> draw_with_replacement(pool, 8, "s", 1)
        [40, 10, 20, 40, 20, 10, 30, 40]

        Repeats occur, which is the entire point of a bootstrap:

        >>> len(draws), len(set(draws))
        (8, 4)
        >>> draws.count(30)
        3

        The draws are prefix-nested in ``n_draws``:

        >>> draw_with_replacement(pool, 3, "s", 0)
        [20, 10, 30]
        >>> all(draw_with_replacement(pool, k, "s", 0) == draws[:k] for k in range(9))
        True

        Only the *set* of pool members matters, not the order they arrive in:

        >>> draw_with_replacement([40, 30, 20, 10], 8, "s", 0) == draws
        True

        Boundaries:

        >>> draw_with_replacement([42], 5, "s", 0)
        [42, 42, 42, 42, 42]
        >>> draw_with_replacement(pool, 0, "s", 0)
        []
        >>> draw_with_replacement([], 0, "s", 0)
        []
        >>> draw_with_replacement([], 3, "s", 0)
        Traceback (most recent call last):
            ...
        ValueError: Cannot draw 3 subjects from an empty pool
        >>> draw_with_replacement(pool, -1, "s", 0)
        Traceback (most recent call last):
            ...
        ValueError: n_draws must be >= 0; got -1
        >>> draw_with_replacement([10, 10], 1, "s", 0)
        Traceback (most recent call last):
            ...
        ValueError: Duplicate subject_id 10
    """
    if n_draws < 0:
        raise ValueError(f"n_draws must be >= 0; got {n_draws}")
    ordered = sorted(_require_unique(pool))
    if n_draws == 0:
        return []
    n = len(ordered)
    if n == 0:
        raise ValueError(f"Cannot draw {n_draws} subjects from an empty pool")

    cutoff = _rejection_cutoff(n)
    prefix = _replicate_salt(salt, replicate)
    draws: list[int] = []
    counter = 0
    while len(draws) < n_draws:
        digest = hashlib.sha256(f"{prefix}{SUBJECT_SEP}{counter}".encode()).digest()
        counter += 1
        u = int.from_bytes(digest[:_DRAW_BYTES], "big")
        if u >= cutoff:
            continue
        draws.append(ordered[u % n])
    return draws


def draw_without_replacement(pool: Sequence[int], n_draws: int, salt: str, replicate: int) -> list[int]:
    """Draw ``n_draws`` distinct subjects from ``pool``, uniformly, *without* replacement.

    Sampling without replacement degenerates to a prefix of a per-replicate rank order: the rank digest
    is salted with the replicate, giving each replicate its own total order, and the first ``n_draws``
    subjects of that order are the draw. This inherits nesting from :func:`select_nested` for free ---
    growing a replicate only appends --- and needs no rejection sampling.

    The replicate's rank order shares the ``f"{salt}|resample|{replicate}"`` namespace with the counter
    stream of :func:`draw_with_replacement`. That is harmless: a given resample manifest uses exactly
    one of the two schemes, and no independence between them is claimed or required.

    Args:
        pool: The subjects to draw from. Order is irrelevant.
        n_draws: How many draws to take. Must be ``>= 0`` and at most the pool size.
        salt: The family salt.
        replicate: The replicate index; distinct replicates give independent orders.

    Returns:
        ``n_draws`` distinct subject ids, in draw order.

    Raises:
        ValueError: If ``n_draws`` is negative, if ``n_draws`` exceeds the pool size, or if ``pool``
            has duplicates.

    Examples:
        >>> pool = [10, 20, 30, 40]
        >>> draw_without_replacement(pool, 3, "s", 0)
        [20, 30, 40]

        Replicates are separated:

        >>> draw_without_replacement(pool, 3, "s", 1)
        [10, 30, 40]
        >>> draw_without_replacement(pool, 3, "s", 2)
        [20, 40, 30]

        Draws are distinct, and a full-size draw is a permutation of the pool:

        >>> draws = draw_without_replacement(list(range(1, 13)), 12, "s", 0)
        >>> draws
        [11, 8, 6, 3, 4, 12, 7, 9, 10, 2, 5, 1]
        >>> sorted(draws) == list(range(1, 13))
        True

        And they are prefix-nested in ``n_draws``:

        >>> all(draw_without_replacement(pool, k, "s", 0) == [20, 30, 40, 10][:k] for k in range(5))
        True

        Boundaries:

        >>> draw_without_replacement([42], 1, "s", 0)
        [42]
        >>> draw_without_replacement(pool, 0, "s", 0)
        []
        >>> draw_without_replacement([], 0, "s", 0)
        []

        Every error this function raises is phrased in its own vocabulary --- ``n_draws`` and
        "draw", never the ``n``/"select" of the :func:`select_nested` it delegates to:

        >>> draw_without_replacement(pool, 5, "s", 0)
        Traceback (most recent call last):
            ...
        ValueError: Cannot draw 5 distinct subjects from a pool of 4
        >>> draw_without_replacement([], 1, "s", 0)
        Traceback (most recent call last):
            ...
        ValueError: Cannot draw 1 distinct subjects from a pool of 0
        >>> draw_without_replacement(pool, -1, "s", 0)
        Traceback (most recent call last):
            ...
        ValueError: n_draws must be >= 0; got -1
        >>> draw_without_replacement([10, 10], 1, "s", 0)
        Traceback (most recent call last):
            ...
        ValueError: Duplicate subject_id 10
    """
    if n_draws < 0:
        raise ValueError(f"n_draws must be >= 0; got {n_draws}")
    ordered = _require_unique(pool)
    if n_draws > len(ordered):
        raise ValueError(f"Cannot draw {n_draws} distinct subjects from a pool of {len(ordered)}")
    return select_nested(ordered, n_draws, _replicate_salt(salt, replicate))
