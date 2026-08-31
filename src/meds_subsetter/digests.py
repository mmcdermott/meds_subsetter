r"""Canonical, order-insensitive content digests for MEDS data.

Every id this package reports -- ``subject_set_id``, a split's ``content_id``, ``train_set_id``,
``index_id``, ``dataset_id`` -- is built here. Those ids are written to disk and compared across
machines, package versions, and *shardings*, so the encoding below is a pinned wire format: changing
any of it changes every id the package has ever emitted.

Three properties are load-bearing, and each one rules out an obvious cheaper alternative:

* **Reproducible.** Only :mod:`hashlib` is used. The builtin ``hash`` is salted per process, and
  polars' ``hash`` / ``hash_rows`` are documented as unstable across versions (they also disagree
  between ``String`` and ``Categorical`` columns holding the same values).
* **Reshard invariant.** A digest is a function of *content*, not of how that content was split into
  files. So a frame is reduced to per-subject digests, and those are combined order-insensitively --
  never by hashing file bytes, and never in row order. One caveat is explicit rather than implied:
  MEDS shards of the same dataset may differ in their *columns* (``numeric_value`` is optional,
  extra columns are allowed), and a row's rendering necessarily names the fields it has. So
  :func:`subject_digests` and :func:`frame_digest` take a ``schema`` argument that pins the rendered
  column set dataset-wide; pass it whenever per-shard digests will be combined or compared.
* **Injection proof.** Row fields are length-prefixed rather than merely delimited, and column names
  are constrained to characters that cannot open a field. ``text_value`` is free clinical text and
  legitimately contains newlines, ``=``, and even control characters, so a delimiter-only encoding
  would let one column's value forge another column's field.

The canonical encoding, in full:

* :func:`digest` frames each item as ``<len-in-bytes>:<utf8 bytes>`` and concatenates the frames.
  This is self-delimiting, so no byte string is ever illegal.
* :func:`canonical_row_expr` renders a row by taking the columns in **sorted name order**, rendering
  each as ``name=<len-in-bytes>:<value>``, and joining the fields with :data:`FIELD_SEP`. A null
  field gets length :data:`NULL_LEN` and an empty value, which keeps ``null`` distinct from ``""``.
* :func:`combine_digests` sorts digests before hashing them, which makes the combination
  order-insensitive while *preserving multiplicity* -- unlike an XOR fold, in which a duplicated row
  or subject would silently annihilate.

Examples:
    The whole stack, end to end. Two shardings of one dataset, digested independently, agree:

    >>> rows = {
    ...     "subject_id": [1, 1, 2],
    ...     "code": ["A", "B", "A"],
    ...     "numeric_value": pl.Series([1.0, None, 2.0], dtype=pl.Float32),
    ... }
    >>> whole = pl.DataFrame(rows)
    >>> one_shard = combine_subject_digests(subject_digests(whole.lazy()))
    >>> two_shards = combine_subject_digests(
    ...     pl.concat([subject_digests(whole[:2].lazy()), subject_digests(whole[2:].lazy())])
    ... )
    >>> one_shard == two_shards
    True
    >>> short_id(one_shard)
    '5e5a226084191ea6'
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: Prefix on every digest this module returns; matches the sibling ``meds_summary_stats.digests``.
DIGEST_PREFIX = "sha256:"

#: ASCII unit separator, used between the fields of a canonical row string.
FIELD_SEP = "\x1f"

#: Length written for a null field, so that ``null`` never encodes the same as the empty string.
NULL_LEN = -1

_HEX_DIGITS = frozenset("0123456789abcdef")
_ROW_COL = "__canonical_row__"


def digest(items: Iterable[str]) -> str:
    r"""Return the canonical digest of ``items``, in the order given.

    Each item is framed as ``f"{len(item.encode())}:"`` followed by its UTF-8 bytes. The length
    prefix makes the encoding self-delimiting, so items may contain *any* characters -- newlines,
    :data:`FIELD_SEP`, NUL -- without ever colliding with a different sequence of items.

    Order is significant; this function does not sort. Use :func:`combine_digests` when the inputs
    are an unordered collection.

    Args:
        items: The strings to digest. Consumed exactly once.

    Returns:
        A ``"sha256:<hex>"`` string.

    Examples:
        >>> digest(["a", "b"])
        'sha256:facdde7abf1eac5b301273ab2e282f79bab0100f058833a7b8bdf9f80741e149'

        Order matters, and the framing is what a lazy iterator sees too:

        >>> digest(["b", "a"]) == digest(["a", "b"])
        False
        >>> digest(iter(["a", "b"])) == digest(["a", "b"])
        True

        The empty sequence is well defined, so an empty split still has a comparable digest:

        >>> digest([])
        'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

        The length prefixes are the point: a plain concatenation would give ``["ab", "c"]`` and
        ``["a", "bc"]`` the same bytes, and this encoding does not:

        >>> digest(["ab", "c"])
        'sha256:430fb1b4ac43316eca81fab27a1930ab8eff8fef6a1dc7903dce44bbc2790dc5'
        >>> digest(["a", "bc"])
        'sha256:5310a58788781ab25d5ad7c3f85035824b4eb7bdfa394e0ac2186271472b5492'

        Nor does an empty item vanish, the way it would under a delimiter-free scheme:

        >>> digest(["", "a"]) == digest(["a"])
        False

        Unlike ``meds_summary_stats.digests.digest``, embedded newlines are accepted rather than
        rejected -- MEDS ``text_value`` is free clinical text and routinely contains them:

        >>> digest(["line1\nline2"])
        'sha256:7f1489732e7a3815609dbe9d34fd28dd51129de54c29477c095244471c5e574a'

        As is the field separator itself:

        >>> digest(["x\x1fy"])
        'sha256:79edb8cd625c1959b72cd971e64f2a0e668b709012908aaa2057e91f42d6394b'
        >>> digest(["x\x1fy"]) == digest(["x", "y"])
        False
    """
    h = hashlib.sha256()
    for item in items:
        encoded = item.encode("utf-8")
        h.update(f"{len(encoded)}:".encode())
        h.update(encoded)
    return f"{DIGEST_PREFIX}{h.hexdigest()}"


def combine_digests(digests: Iterable[str]) -> str:
    """Combine digests into one, independently of the order they arrive in.

    Sorting before hashing is what buys order-independence, and it keeps the result an ordinary
    sha256 over a canonical string rather than a bespoke arithmetic fold. Duplicates are retained:
    two subjects with byte-identical histories are not the same as one, and an XOR-style fold would
    silently erase both.

    Args:
        digests: The digests to combine. Any strings work, but these are normally the output of
            :func:`digest`.

    Returns:
        A single ``"sha256:<hex>"`` string.

    Examples:
        >>> a, b = digest(["a"]), digest(["b"])
        >>> combine_digests([a, b])
        'sha256:25d301eb98b1969220a11630858c51adb4f9bc009c742fdd7497ba7baaa70ee5'

        Order-insensitive:

        >>> combine_digests([a, b]) == combine_digests([b, a])
        True

        But not content-insensitive:

        >>> combine_digests([a, b]) == combine_digests([a])
        False

        Multiplicity is preserved -- there is no XOR-style annihilation of duplicates:

        >>> combine_digests([a, a])
        'sha256:518941e201ff89199de5efe6903b16b57bfff5d2e0a6d215414d4ea308d8c2fb'
        >>> combine_digests([a, a]) == combine_digests([a])
        False
        >>> combine_digests([a, a, b]) == combine_digests([a, b])
        False

        The empty combination is well defined:

        >>> combine_digests([]) == digest([])
        True
    """
    return digest(sorted(digests))


def short_id(d: str, n: int = 16) -> str:
    """Return the first ``n`` hex characters of a digest, without the ``sha256:`` prefix.

    Short ids are for directory names and human-facing reports. They are truncations, not digests:
    never compare them where a full digest comparison is what you mean.

    Args:
        d: A digest, with or without the :data:`DIGEST_PREFIX`.
        n: How many hex characters to keep. Must be in ``[1, 64]``.

    Returns:
        The first ``n`` characters of the digest's hex body.

    Raises:
        ValueError: If ``n`` is outside ``[1, 64]``, or ``d`` is not a 64-character lowercase hex
            digest once the prefix is stripped.

    Examples:
        >>> short_id(digest(["a"]))
        '4162fddd39a3e422'
        >>> short_id(digest(["a"]), 8)
        '4162fddd'
        >>> short_id(digest(["a"]), 64) == digest(["a"]).removeprefix(DIGEST_PREFIX)
        True

        The prefix is optional, so a short id is stable whether or not it round-tripped through a
        JSON field that dropped it:

        >>> short_id("4162fddd39a3e4225e8e2392eced237fbeb34e6e218b5647d27bd4d2b9c0da24", 8)
        '4162fddd'

        Out-of-range widths are refused rather than clamped:

        >>> short_id(digest(["a"]), 0)
        Traceback (most recent call last):
            ...
        ValueError: n must be in [1, 64]; got 0
        >>> short_id(digest(["a"]), 65)
        Traceback (most recent call last):
            ...
        ValueError: n must be in [1, 64]; got 65

        As is anything that is not a digest -- including a short id, which would otherwise silently
        truncate a second time:

        >>> short_id("not-a-digest")
        Traceback (most recent call last):
            ...
        ValueError: Expected a 64-character lowercase hex digest; got 'not-a-digest'
        >>> short_id(short_id(digest(["a"])))
        Traceback (most recent call last):
            ...
        ValueError: Expected a 64-character lowercase hex digest; got '4162fddd39a3e422'
    """
    if not 1 <= n <= 64:
        raise ValueError(f"n must be in [1, 64]; got {n}")
    body = d.removeprefix(DIGEST_PREFIX)
    if len(body) != 64 or not _HEX_DIGITS.issuperset(body):
        raise ValueError(f"Expected a 64-character lowercase hex digest; got {d!r}")
    return body[:n]


def _framed(value: pl.Expr) -> pl.Expr:
    """Frame a nullable ``String`` expression as ``<len-in-bytes>:<value>``.

    Args:
        value: A ``String`` expression. Nulls render with length :data:`NULL_LEN` and no value.

    Returns:
        A non-null ``String`` expression.
    """
    length = (
        pl.when(value.is_null()).then(pl.lit(str(NULL_LEN))).otherwise(value.str.len_bytes().cast(pl.String))
    )
    return pl.concat_str(length, pl.lit(":"), value.fill_null(pl.lit("")))


def _field_expr(name: str, dtype: pl.DataType) -> pl.Expr:
    """Return the canonical ``name=<len>:<value>`` field expression for one column.

    Args:
        name: The column name.
        dtype: The column's fully-specified dtype.

    Returns:
        A ``String`` expression, never null.

    Raises:
        TypeError: If ``dtype`` has no canonical encoding.
    """
    col = pl.col(name)
    if dtype.is_float():
        value = col.cast(pl.Float64).reinterpret(dtype=pl.UInt64).cast(pl.String)
    elif isinstance(dtype, pl.Datetime):
        value = col.dt.cast_time_unit("us").dt.epoch("us").cast(pl.String)
    elif isinstance(dtype, pl.List) and dtype.inner == pl.String:
        value = col.list.eval(_framed(pl.element())).list.join("")
    elif dtype.is_integer() or isinstance(
        dtype, pl.String | pl.Boolean | pl.Categorical | pl.Enum | pl.Date | pl.Null
    ):
        # ``Null`` casts to an all-null String, which frames as the null field it is -- so a column
        # a writer left untyped digests the same as the typed all-null column it stands for.
        value = col.cast(pl.String)
    else:
        raise TypeError(f"Cannot canonically encode column {name!r} of dtype {dtype}")
    return pl.concat_str(pl.lit(f"{name}="), _framed(value))


def canonical_row_expr(schema: Mapping[str, pl.DataType]) -> pl.Expr:
    r"""Build the expression that renders each row as one canonical UTF-8 string.

    Columns are taken in **sorted name order**, so a frame's digest does not depend on physical
    column order. Each column contributes ``name=<len-in-bytes>:<value>``, and the fields are joined
    with :data:`FIELD_SEP`. Because every value carries its own byte length, and because column
    names may contain neither :data:`FIELD_SEP` nor ``=`` (both are refused below), the row string
    is uniquely decodable: no value, however adversarial, can forge a field boundary.

    The rendered field set is exactly ``schema``'s. It is therefore a property of the frame handed
    in, not of the dataset -- which is why :func:`subject_digests` and :func:`frame_digest` accept
    a pinned schema to render heterogeneous shards against.

    Per-dtype value rendering, chosen so that the string is a function of the *value* and never of a
    formatting default:

    ======================  ==========================================================
    dtype                   rendering
    ======================  ==========================================================
    ``Float32``/``Float64`` cast to ``Float64``, then the raw IEEE-754 bits as a
                            ``UInt64`` -- no decimal formatting is involved at all
    ``Datetime(*)``         microseconds since epoch, after ``cast_time_unit("us")``
    ``List(String)``        each element framed ``<len>:<value>``, concatenated
    integer, ``String``,    ``cast(pl.String)``
    ``Boolean``,
    ``Categorical``,
    ``Enum``, ``Date``,
    ``Null``
    ======================  ==========================================================

    Hashing float *bits* means ``-0.0`` and ``0.0`` digest differently, and a NaN is hashed by its
    bit pattern rather than compared. That is deliberate: this is a content hash, not a numeric
    comparison, and it must answer "are these the same bytes on disk?" rather than "are these
    numerically equal?".

    Args:
        schema: Column name to fully-specified dtype, e.g. ``lf.collect_schema()`` or
            ``pl.Schema({...})``.

    Returns:
        A ``String`` expression yielding one canonical row string per row. It is never null.

    Raises:
        ValueError: If ``schema`` is empty, or a column name contains :data:`FIELD_SEP` or ``=``.
        TypeError: If any column has a dtype with no canonical encoding.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 2],
        ...         "code": ["A", "B", "A"],
        ...         "numeric_value": pl.Series([1.0, None, 2.0], dtype=pl.Float32),
        ...     }
        ... )
        >>> rows = df.select(canonical_row_expr(df.lazy().collect_schema()).alias("r"))["r"]
        >>> for r in rows:
        ...     print(repr(r))
        'code=1:A\x1fnumeric_value=19:4607182418800017408\x1fsubject_id=1:1'
        'code=1:B\x1fnumeric_value=-1:\x1fsubject_id=1:1'
        'code=1:A\x1fnumeric_value=19:4611686018427387904\x1fsubject_id=1:2'

        Note the null: it renders with length ``-1`` and an empty value, which no real string can
        produce, so a null ``numeric_value`` never collides with an empty one:

        >>> nulls = pl.DataFrame({"v": [None, ""]}, schema={"v": pl.String})
        >>> nulls.select(canonical_row_expr(nulls.lazy().collect_schema()).alias("r"))["r"].to_list()
        ['v=-1:', 'v=0:']

        Floats go in as raw bits, so ``-0.0`` and ``0.0`` are distinguishable -- deliberately, since
        they are distinct bytes on disk:

        >>> zeros = pl.DataFrame({"v": pl.Series([0.0, -0.0], dtype=pl.Float64)})
        >>> zeros.select(canonical_row_expr(zeros.lazy().collect_schema()).alias("r"))["r"].to_list()
        ['v=1:0', 'v=19:9223372036854775808']

        Datetimes are normalized to microseconds first, so a frame that round-tripped through a
        writer that widened the time unit still digests the same:

        >>> ns = pl.DataFrame({"time": pl.Series([datetime(2021, 1, 1)], dtype=pl.Datetime("ns"))})
        >>> us = ns.with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> ns.select(canonical_row_expr(ns.lazy().collect_schema()).alias("r"))["r"].to_list()
        ['time=16:1609459200000000']
        >>> us.select(canonical_row_expr(us.lazy().collect_schema()).alias("r"))["r"].to_list()
        ['time=16:1609459200000000']

        ``List(String)`` -- i.e. ``parent_codes`` in ``metadata/codes.parquet`` -- gets an
        element-wise framing, because a bare ``cast(pl.String)`` on a list column raises. A null
        list, an empty list, and a list holding one empty string all stay distinct:

        >>> lists = pl.DataFrame(
        ...     {"parent_codes": pl.Series([["a", "bc"], None, [], [""]], dtype=pl.List(pl.String))}
        ... )
        >>> lists.select(canonical_row_expr(lists.lazy().collect_schema()).alias("r"))["r"].to_list()
        ['parent_codes=7:1:a2:bc', 'parent_codes=-1:', 'parent_codes=0:', 'parent_codes=2:0:']

        Sorted-name order means physical column order is irrelevant:

        >>> flipped = df.select("numeric_value", "code", "subject_id")
        >>> flipped.select(canonical_row_expr(flipped.lazy().collect_schema()).alias("r"))["r"][0]
        'code=1:A\x1fnumeric_value=19:4607182418800017408\x1fsubject_id=1:1'

        Rendering is a function of the *value*, not of the storage dtype, so a shard that came back
        from parquet with a narrower integer or a dictionary-encoded ``code`` still digests the
        same. This is exactly where polars' own ``hash_rows`` differs:

        >>> def rowstrs(f):
        ...     expr = canonical_row_expr(f.lazy().collect_schema())
        ...     return f.select(expr.alias("r"))["r"].to_list()
        >>> as_str = pl.DataFrame({"code": pl.Series(["x", "y"], dtype=pl.String)})
        >>> as_cat = pl.DataFrame({"code": pl.Series(["x", "y"], dtype=pl.Categorical)})
        >>> rowstrs(as_str) == rowstrs(as_cat) == ["code=1:x", "code=1:y"]
        True
        >>> narrow = pl.DataFrame({"code": pl.Series([1, 2], dtype=pl.Int32)})
        >>> wide = pl.DataFrame({"code": pl.Series([1, 2], dtype=pl.Int64)})
        >>> rowstrs(narrow) == rowstrs(wide) == ["code=1:1", "code=1:2"]
        True

        The same holds for an all-null column that a writer typed as ``Null`` -- which is what a
        parquet round-trip of real ACES label output gives back. It renders as the null field it is,
        so an ``index_id`` never depends on the writer's type inference:

        >>> untyped = pl.DataFrame({"boolean_value": [None, None]})
        >>> untyped.schema["boolean_value"]
        Null
        >>> typed = untyped.with_columns(pl.col("boolean_value").cast(pl.Boolean))
        >>> rowstrs(untyped) == rowstrs(typed) == ["boolean_value=-1:", "boolean_value=-1:"]
        True

        The remaining golden values of the pinned wire format, so that a change in polars'
        ``cast(pl.String)`` output cannot move an id silently. A NaN is rendered by its bits, which
        means the two sign variants are distinguishable and neither is elided:

        >>> rowstrs(pl.DataFrame({"v": pl.Series([float("nan"), -float("nan")], dtype=pl.Float32)}))
        ['v=19:9221120237041090560', 'v=20:18444492273895866368']
        >>> rowstrs(pl.DataFrame({"d": pl.Series([datetime(2021, 1, 1).date()], dtype=pl.Date)}))
        ['d=10:2021-01-01']
        >>> rowstrs(pl.DataFrame({"e": pl.Series(["a"], dtype=pl.Enum(["a", "b"]))}))
        ['e=1:a']

        An unhandled dtype is refused loudly, naming the column, rather than being coerced into a
        digest nobody can reproduce:

        >>> binary = pl.DataFrame({"blob": [b"x"]})
        >>> canonical_row_expr(binary.lazy().collect_schema())
        Traceback (most recent call last):
            ...
        TypeError: Cannot canonically encode column 'blob' of dtype Binary
        >>> canonical_row_expr(pl.Schema({"when": pl.Time}))
        Traceback (most recent call last):
            ...
        TypeError: Cannot canonically encode column 'when' of dtype Time
        >>> canonical_row_expr(pl.Schema({"codes": pl.List(pl.Int64)}))
        Traceback (most recent call last):
            ...
        TypeError: Cannot canonically encode column 'codes' of dtype List(Int64)

        A column with no fields, and a column name that would itself break the framing, are also
        errors rather than silent ambiguity. Names are the one part of a field that is *not*
        length-prefixed, so a name containing :data:`FIELD_SEP` or ``=`` could forge a field
        boundary that no value can; both are refused, which is what makes the row string uniquely
        decodable:

        >>> canonical_row_expr(pl.Schema({}))
        Traceback (most recent call last):
            ...
        ValueError: schema must have at least one column
        >>> canonical_row_expr(pl.Schema({"a\x1fb": pl.String}))
        Traceback (most recent call last):
            ...
        ValueError: Column name may not contain '\x1f': 'a\x1fb'
        >>> canonical_row_expr(pl.Schema({"x=5:y": pl.String}))
        Traceback (most recent call last):
            ...
        ValueError: Column name may not contain '=': 'x=5:y'

        That last name is not hypothetical: without the check, a column ``x=5:y`` holding ``"z"``
        and a column ``x`` holding ``"y=1:z"`` would both render ``'x=5:y=1:z'`` and collide.

        >>> rowstrs(pl.DataFrame({"x": ["y=1:z"]}))
        ['x=5:y=1:z']
    """
    names = sorted(schema)
    if not names:
        raise ValueError("schema must have at least one column")
    for name in names:
        for forbidden in (FIELD_SEP, "="):
            if forbidden in name:
                raise ValueError(f"Column name may not contain {forbidden!r}: {name!r}")
    return pl.concat_str([_field_expr(name, schema[name]) for name in names], separator=FIELD_SEP)


def _rendered(lf: pl.LazyFrame, schema: Mapping[str, pl.DataType] | None) -> tuple[pl.LazyFrame, pl.Schema]:
    """Align ``lf`` to an optional pinned ``schema`` and report the schema to render it against.

    Columns named by ``schema`` but absent from ``lf`` are added as typed nulls, so a shard that
    legally omits an optional column renders identically to one that carries it all-null. Columns
    present in ``lf`` but absent from ``schema`` are refused rather than dropped.

    Args:
        lf: The frame to align.
        schema: The pinned column set, or ``None`` to render against ``lf``'s own schema.

    Returns:
        The aligned frame and the schema to hand :func:`canonical_row_expr`.

    Raises:
        ValueError: If ``lf`` has a column that ``schema`` does not name.
    """
    frame_schema = lf.collect_schema()
    if schema is None:
        return lf, frame_schema
    extra = [name for name in frame_schema if name not in schema]
    if extra:
        raise ValueError(f"Frame has columns absent from the pinned schema: {extra}")
    missing = {name: dtype for name, dtype in schema.items() if name not in frame_schema}
    if not missing:
        return lf, frame_schema
    filled = lf.with_columns([pl.lit(None, dtype=dtype).alias(name) for name, dtype in missing.items()])
    return filled, pl.Schema({**frame_schema, **missing})


def subject_digests(
    lf: pl.LazyFrame,
    *,
    subject_col: str = "subject_id",
    schema: Mapping[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    r"""Digest each subject's rows, independently of row order and of sharding.

    Each subject's canonical row strings are **sorted** and then fed to :func:`digest`. Sorting is
    what makes the result independent of the order rows happen to sit in a file, and hashing the
    sorted list rather than folding per-row digests is what keeps duplicate rows -- which MEDS
    legitimately has -- from cancelling.

    MEDS guarantees a subject lives in exactly one data file, so *given a fixed rendered column
    set*, the digest of a subject computed from one shard is the digest of that subject computed
    from the whole dataset. That is what makes the family of subsets in this package comparable
    across different shardings.

    The "fixed rendered column set" caveat is load-bearing rather than pedantic: MEDS makes
    ``numeric_value`` optional and permits extra columns, so shards of one dataset may differ in
    their columns, and the field set is otherwise read off whatever frame is handed in. Pass
    ``schema`` -- the dataset-wide column set, e.g. the ``collect_schema()`` of a
    ``scan_parquet(..., missing_columns="insert")`` over every shard -- whenever per-shard digests
    will be combined or compared. Omit it only when the frame already spans everything being
    compared, or when every shard's schema is known to be identical.

    The canonical row strings are materialized to compute this, so prefer one call per shard --
    which the above invariance makes free -- over one call across a whole dataset.

    Args:
        lf: The frame to digest. Every column participates, ``subject_col`` included.
        subject_col: The subject identifier column to group by.
        schema: The column set to render against. Columns it names that ``lf`` lacks are rendered
            as nulls; columns ``lf`` has that it does not name are an error. ``None`` renders
            against ``lf``'s own schema.

    Returns:
        A two-column ``DataFrame`` of ``subject_col`` and ``digest``, sorted by ``subject_col``.

    Raises:
        KeyError: If ``subject_col`` is not a column of ``lf`` (or of ``schema``, when given).
        ValueError: If ``lf`` has a column absent from ``schema``, the rendered column set is
            empty, or a column name contains :data:`FIELD_SEP` or ``=``.
        TypeError: If any column has a dtype with no canonical encoding.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 2, 2, 2],
        ...         "code": ["A", "B", "A", "C", "C"],
        ...         "numeric_value": pl.Series([1.0, None, 2.0, 3.0, 3.0], dtype=pl.Float32),
        ...     }
        ... )
        >>> sd = subject_digests(df.lazy())
        >>> sd.columns
        ['subject_id', 'digest']
        >>> sd["subject_id"].to_list()
        [1, 2]
        >>> [short_id(d, 12) for d in sd["digest"]]
        ['b124d1493d82', '9aaa6a1c928c']

        Row order within a subject is irrelevant:

        >>> shuffled = df[[1, 0, 4, 3, 2]]
        >>> subject_digests(shuffled.lazy()).equals(sd)
        True

        And so is which shard a subject came from, as long as a subject is not split across shards:

        >>> from_shards = pl.concat(
        ...     [subject_digests(df[:2].lazy()), subject_digests(df[2:].lazy())]
        ... ).sort("subject_id")
        >>> from_shards.equals(sd)
        True

        Duplicate rows do not cancel -- subject 2's two identical ``C`` rows are load-bearing:

        >>> deduped = df.unique(maintain_order=True)
        >>> subject_digests(deduped.lazy()).equals(sd)
        False

        Changing a single numeric value anywhere changes exactly that subject's digest:

        >>> bumped = df.with_columns(
        ...     numeric_value=pl.Series([1.0, None, 2.0, 3.0, 3.5], dtype=pl.Float32)
        ... )
        >>> bd = subject_digests(bumped.lazy())
        >>> bd["digest"][0] == sd["digest"][0], bd["digest"][1] == sd["digest"][1]
        (True, False)

        An empty frame yields an empty, correctly-typed result rather than an error:

        >>> empty = pl.DataFrame(schema={"subject_id": pl.Int64, "code": pl.String})
        >>> print(subject_digests(empty.lazy()))
        shape: (0, 2)
        ┌────────────┬────────┐
        │ subject_id ┆ digest │
        │ ---        ┆ ---    │
        │ i64        ┆ str    │
        ╞════════════╪════════╡
        └────────────┴────────┘

        Shards of one dataset may legally differ in their *columns*, not just their rows: MEDS makes
        ``numeric_value`` optional and permits extra columns, so a shard can be three columns wide.
        The rendered field set comes from the frame, so without help a subject in a narrow shard
        digests differently from the same subject read back through ``missing_columns="insert"``.
        Pass ``schema`` -- the dataset-wide column set -- to pin it, and the two agree:

        >>> wide = pl.DataFrame(
        ...     {"subject_id": [1], "code": ["A"], "numeric_value": pl.Series([1.0], dtype=pl.Float32)}
        ... )
        >>> narrow = pl.DataFrame({"subject_id": [2], "code": ["A"]})
        >>> pinned = pl.Schema({"subject_id": pl.Int64, "code": pl.String, "numeric_value": pl.Float32})
        >>> subject_digests(narrow.lazy())["digest"][0] == subject_digests(
        ...     narrow.lazy(), schema=pinned
        ... )["digest"][0]
        False

        Pinned, the per-shard digests *are* the whole-dataset digests, which is what the cached
        per-subject digests of a family rely on:

        >>> aligned = pl.concat([wide, narrow], how="diagonal")
        >>> per_shard = pl.concat(
        ...     [subject_digests(wide.lazy(), schema=pinned), subject_digests(narrow.lazy(), schema=pinned)]
        ... )
        >>> per_shard.equals(subject_digests(aligned.lazy(), schema=pinned))
        True
        >>> per_shard.equals(subject_digests(aligned.lazy()))
        True

        A column the pinned schema does not mention is refused rather than silently dropped -- that
        would quietly digest away real content:

        >>> subject_digests(wide.lazy(), schema=pl.Schema({"subject_id": pl.Int64, "code": pl.String}))
        Traceback (most recent call last):
            ...
        ValueError: Frame has columns absent from the pinned schema: ['numeric_value']

        A missing subject column is a hard error; digesting the wrong grouping silently would be far
        worse than failing:

        >>> subject_digests(pl.DataFrame({"code": ["A"]}).lazy())
        Traceback (most recent call last):
            ...
        KeyError: "Column 'subject_id' not found; frame has ['code']"

        As is a column name that cannot be canonically framed:

        >>> subject_digests(pl.DataFrame({"subject_id": [1], "a\x1fb": ["x"]}).lazy())
        Traceback (most recent call last):
            ...
        ValueError: Column name may not contain '\x1f': 'a\x1fb'
    """
    lf, render_schema = _rendered(lf, schema)
    if subject_col not in render_schema:
        raise KeyError(f"Column {subject_col!r} not found; frame has {list(render_schema)}")
    grouped = (
        lf.select(pl.col(subject_col), canonical_row_expr(render_schema).alias(_ROW_COL))
        .group_by(subject_col)
        .agg(pl.col(_ROW_COL))
        .sort(subject_col)
        .collect()
    )
    hexes = [digest(sorted(rows)) for rows in grouped[_ROW_COL]]
    return pl.DataFrame([grouped[subject_col], pl.Series("digest", hexes, dtype=pl.String)])


def frame_digest(lf: pl.LazyFrame, *, schema: Mapping[str, pl.DataType] | None = None) -> str:
    r"""Digest every row of a frame, with no subject grouping.

    This is the fingerprint for a frame that has no meaningful subject partition -- a task/index
    label dataframe, or ``metadata/codes.parquet``. Per-row digests are combined with
    :func:`combine_digests`, so the result is independent of row order but sensitive to how many
    times each row appears.

    As in :func:`subject_digests`, the rendered field set is otherwise read off the frame, so pass
    ``schema`` when digesting shards that may differ in which optional columns they carry.

    Args:
        lf: The frame to digest. Every column participates.
        schema: The column set to render against. Columns it names that ``lf`` lacks are rendered
            as nulls; columns ``lf`` has that it does not name are an error. ``None`` renders
            against ``lf``'s own schema.

    Returns:
        A ``"sha256:<hex>"`` string.

    Raises:
        ValueError: If the rendered column set is empty, ``lf`` has a column absent from
            ``schema``, or a column name contains :data:`FIELD_SEP` or ``=``.
        TypeError: If any column has a dtype with no canonical encoding.

    Examples:
        >>> labels = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 2],
        ...         "prediction_time": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
        ...         "boolean_value": [True, False],
        ...     }
        ... )
        >>> short_id(frame_digest(labels.lazy()))
        'd13ee82d64fd44d2'

        Row order does not matter:

        >>> frame_digest(labels[[1, 0]].lazy()) == frame_digest(labels.lazy())
        True

        Multiplicity does:

        >>> frame_digest(pl.concat([labels, labels]).lazy()) == frame_digest(labels.lazy())
        False

        With one row per subject, this agrees with the subject-wise path by construction, which is
        what lets an index digest and a data digest be reported side by side:

        >>> frame_digest(labels.lazy()) == combine_subject_digests(subject_digests(labels.lazy()))
        True

        Injection resistance, which is the whole reason for the length prefixes. These two frames
        differ only in *where* the field boundary falls inside otherwise identical text, and a
        delimiter-only encoding would give both the same string ``code=a\x1ftext_value=b\x1ftext_value=c``:

        >>> a = pl.DataFrame({"code": ["a\x1ftext_value=b"], "text_value": ["c"]})
        >>> b = pl.DataFrame({"code": ["a"], "text_value": ["b\x1ftext_value=c"]})
        >>> a.select(canonical_row_expr(a.lazy().collect_schema()).alias("r"))["r"][0]
        'code=14:a\x1ftext_value=b\x1ftext_value=1:c'
        >>> b.select(canonical_row_expr(b.lazy().collect_schema()).alias("r"))["r"][0]
        'code=1:a\x1ftext_value=14:b\x1ftext_value=c'
        >>> frame_digest(a.lazy()) == frame_digest(b.lazy())
        False

        Newlines in free text are ordinary content, not a parse hazard:

        >>> notes = pl.DataFrame({"code": ["N"], "text_value": ["line1\nline2\n=\x1f"]})
        >>> short_id(frame_digest(notes.lazy()))
        'c144e6e30aa1aeee'

        Real ACES label output leaves the value columns entirely null, and a parquet round-trip
        hands them back as ``Null``-dtyped columns. Those digest exactly as the typed all-null
        column they stand for, so an ``index_id`` is a function of the labels rather than of the
        writer that produced them:

        >>> untyped = pl.DataFrame({"subject_id": [1], "boolean_value": [None]})
        >>> untyped.schema["boolean_value"]
        Null
        >>> typed = untyped.with_columns(pl.col("boolean_value").cast(pl.Boolean))
        >>> frame_digest(untyped.lazy()) == frame_digest(typed.lazy())
        True

        As with :func:`subject_digests`, ``schema`` pins the rendered column set so that label
        shards which differ in which optional value columns they carry still combine:

        >>> only_bool = pl.DataFrame({"subject_id": [1], "boolean_value": [True]})
        >>> with_float = pl.DataFrame(
        ...     {"subject_id": [1], "boolean_value": [True], "float_value": [None]}
        ... )
        >>> label_schema = pl.Schema(
        ...     {"subject_id": pl.Int64, "boolean_value": pl.Boolean, "float_value": pl.Float64}
        ... )
        >>> frame_digest(only_bool.lazy(), schema=label_schema) == frame_digest(
        ...     with_float.lazy(), schema=label_schema
        ... )
        True
        >>> frame_digest(only_bool.lazy()) == frame_digest(with_float.lazy())
        False

        A frame with no columns has no content to digest, and is refused, as is a column name that
        cannot be canonically framed:

        >>> frame_digest(pl.DataFrame().lazy())
        Traceback (most recent call last):
            ...
        ValueError: schema must have at least one column
        >>> frame_digest(pl.DataFrame({"a=b": ["x"]}).lazy())
        Traceback (most recent call last):
            ...
        ValueError: Column name may not contain '=': 'a=b'
    """
    lf, render_schema = _rendered(lf, schema)
    rows = lf.select(canonical_row_expr(render_schema).alias(_ROW_COL)).collect()[_ROW_COL]
    return combine_digests(digest([row]) for row in rows)


def subject_set_digest(subject_ids: Iterable[int]) -> str:
    """Digest a set of subject ids, ignoring order and multiplicity.

    This answers "are these the same subjects?" without reading a single data row, which is what
    makes it cheap enough to bind into every manifest. It says nothing about whether those subjects'
    *data* match -- that is :func:`combine_subject_digests`.

    Ids are sorted, **silently deduplicated**, and rendered as decimal strings.

    Args:
        subject_ids: The subject ids. Consumed exactly once.

    Returns:
        A ``"sha256:<hex>"`` string.

    Examples:
        >>> subject_set_digest([1, 2])
        'sha256:c1a20b6bd4b602a201c037210cde6429b3b32e277260df7defefea3dbbda8fd0'

        Order-insensitive, and duplicates are dropped rather than counted -- a set of subjects is a
        set:

        >>> subject_set_digest([2, 1]) == subject_set_digest([1, 2])
        True
        >>> subject_set_digest([2, 1, 2, 1]) == subject_set_digest([1, 2])
        True

        Membership is all that matters, so a different pool is detectable:

        >>> subject_set_digest([1, 2, 3]) == subject_set_digest([1, 2])
        False

        Ids are decimal, so it composes with anything that reads the parquet column:

        >>> subject_set_digest(pl.Series([2, 1], dtype=pl.Int64)) == subject_set_digest([1, 2])
        True

        The empty set is well defined:

        >>> subject_set_digest([]) == digest([])
        True
    """
    return digest(str(subject_id) for subject_id in sorted(set(subject_ids)))


def combine_subject_digests(df: pl.DataFrame) -> str:
    """Reduce a :func:`subject_digests` frame to a single, reshard-invariant dataset id.

    Args:
        df: A frame with a ``digest`` column, normally from :func:`subject_digests`.

    Returns:
        A ``"sha256:<hex>"`` string.

    Raises:
        KeyError: If ``df`` has no ``digest`` column.

    Examples:
        >>> df = pl.DataFrame({"subject_id": [1, 1, 2], "code": ["A", "B", "A"]})
        >>> sd = subject_digests(df.lazy())
        >>> short_id(combine_subject_digests(sd))
        '7297df144b22fa5d'

        The combine is order-insensitive, so concatenating per-shard results in any order -- or
        digesting the whole dataset at once -- gives the same id:

        >>> combine_subject_digests(sd[[1, 0]]) == combine_subject_digests(sd)
        True
        >>> shards = pl.concat([subject_digests(df[2:].lazy()), subject_digests(df[:2].lazy())])
        >>> combine_subject_digests(shards) == combine_subject_digests(sd)
        True

        Dropping a subject changes it:

        >>> combine_subject_digests(sd[:1]) == combine_subject_digests(sd)
        False

        A frame without a ``digest`` column is refused:

        >>> combine_subject_digests(pl.DataFrame({"subject_id": [1]}))
        Traceback (most recent call last):
            ...
        KeyError: "Expected a 'digest' column; frame has ['subject_id']"
    """
    if "digest" not in df.columns:
        raise KeyError(f"Expected a 'digest' column; frame has {df.columns}")
    return combine_digests(df["digest"])
