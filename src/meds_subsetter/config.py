"""Frozen configuration objects for subsetting and resampling.

Everything that can change what lands on disk -- which sizes get built, how subjects are ranked into
shards, how shard files are linked, what happens to code metadata and to the non-train splits -- lives in
two frozen dataclasses here. Both round-trip losslessly through plain JSON/YAML-native dicts, so a resolved
config can be written into the family manifest beside the data it produced and read back to rebuild it.

Configs are validated at construction and are immutable afterwards, so an invalid config cannot exist and
no downstream stage has to re-check its inputs.

Examples:
    >>> cfg = SubsetConfig(n_subjects=[1000, 10, 100])
    >>> cfg.n_subjects
    (10, 100, 1000)
    >>> [cfg.subset_name(n) for n in cfg.n_subjects]
    ['N0000010', 'N0000100', 'N0001000']
    >>> SubsetConfig.from_dict(cfg.to_dict()) == cfg
    True
"""

from __future__ import annotations

import dataclasses
import operator
import string
from collections.abc import Iterable
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SALT = "meds_subsetter"


class LinkMode(StrEnum):
    """How a subset root's shard files point at the shared shard store.

    Every *full* shard is byte-identical across every member of a nested family, so a family costs about
    as much disk as its largest member plus one partial shard per member -- provided the members reference
    the store rather than copying it. The three modes differ only in how that reference is made, and their
    failure modes are genuinely opposed, so the choice is exposed.

    Attributes:
        SYMLINK: The default. Each shard is a symlink into ``.shard_store/``. ``ls -l`` then shows the
            provenance of every shard, and a downstream tool that overwrites a shard in place writes
            *through* the link and visibly hits the store -- loud, shared breakage in preference to two
            members of a family silently disagreeing. The cost is that ``cp`` without ``-a``
            dereferences symlinks, so copying a subset root elsewhere silently materializes it.
        HARDLINK: Also zero-copy, and ``cp`` without ``-a`` behaves. The trade-offs are the mirror image:
            the sharing is invisible in the tree, an in-place overwrite silently breaks exactly one
            member, and hardlinks cannot cross filesystems. Pick it when consumers are known to be
            read-only and the store shares a filesystem with the subset roots.
        COPY: Independent copies. Costs the summed size of the family rather than the size of its largest
            member, and gives up the point of the store; use it only when subsets must be shipped
            somewhere the store will not follow.

    Examples:
        >>> LinkMode("symlink") is LinkMode.SYMLINK
        True
        >>> [m.value for m in LinkMode]
        ['symlink', 'hardlink', 'copy']
        >>> LinkMode("hard-link")
        Traceback (most recent call last):
            ...
        ValueError: 'hard-link' is not a valid LinkMode
    """

    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    COPY = "copy"


class CodeMetadataPolicy(StrEnum):
    """What to write to a subset's ``metadata/codes.parquet``.

    Attributes:
        PRUNE: Filter the parent's code metadata down to the codes actually observed in the subset's
            data. The default, and the right answer for a single standalone subset. **Hazard:** it makes
            the vocabulary a function of ``N``. A downstream ``MTD_preprocess`` refits
            ``fit_vocabulary_indices`` from whatever ``codes.parquet`` it is handed, so pruning permutes
            the vocabulary indices between members of a family, and the damage is worst at the small-``N``
            end -- i.e. it correlates with the sweep axis the family exists to measure. Building a
            multi-member family with ``PRUNE`` therefore warns and points here.
        COPY: Write the parent's ``codes.parquet`` verbatim. The MEDS guarantee is one-directional
            (``codes.parquet`` covers at least the codes in ``data/``), so a superset is equally valid,
            and this is the only policy under which a nested family shares one vocabulary. It is forced
            for tensorized (MEDS-Torch-Data) cohorts, where ``code/vocab_index`` must not move.

    Examples:
        >>> [m.value for m in CodeMetadataPolicy]
        ['prune', 'copy']
        >>> CodeMetadataPolicy("keep")
        Traceback (most recent call last):
            ...
        ValueError: 'keep' is not a valid CodeMetadataPolicy
    """

    PRUNE = "prune"
    COPY = "copy"


class NonTrainPolicy(StrEnum):
    """What to do with the splits that are not being subsetted (``tuning``, ``held_out``, and any other).

    Attributes:
        PASSTHROUGH: The default. Non-train shards are shared byte-for-byte across the whole family and
            linked in unchanged. This is what a scaling sweep wants -- a fixed evaluation set across the
            size axis, so training-set size is the only thing that varies between members -- and it is
            also what makes those splits shareable on disk at all.
        SUBSET: Apply the same rank-prefix selection independently to every split, so each split shrinks
            with ``N``. Use it when the evaluation set itself is the object of study, and accept that
            evaluation numbers are then not comparable across members.
        DROP: Emit the train split only. The result is not a complete dataset, but it is the smallest
            thing that can still be trained on.

    Examples:
        >>> [m.value for m in NonTrainPolicy]
        ['passthrough', 'subset', 'drop']
        >>> NonTrainPolicy("keep")
        Traceback (most recent call last):
            ...
        ValueError: 'keep' is not a valid NonTrainPolicy
    """

    PASSTHROUGH = "passthrough"
    SUBSET = "subset"
    DROP = "drop"


def _jsonable(value: Any) -> Any:
    """Render one field value in a JSON- and YAML-native form (enums as strings, tuples as lists)."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def _as_dict(cfg: Any) -> dict[str, Any]:
    """Serialize a config dataclass to a plain dict with sorted keys."""
    out = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.init}
    return {k: _jsonable(v) for k, v in sorted(out.items())}


def _checked_kwargs(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Copy ``raw``, raising if it is not a mapping or carries any key ``cls`` does not define.

    Keys are reported through ``repr`` rather than sorted directly: YAML mappings may carry non-string keys,
    and sorting a mix of ``str`` and ``int`` would raise a bare ``TypeError`` in the middle of building an
    error message.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be built from a mapping; got {type(raw).__name__}")
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    if unknown := sorted(repr(k) for k in set(raw) - known):
        raise ValueError(f"Unknown {cls.__name__} keys: [{', '.join(unknown)}]")
    return dict(raw)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    r"""Read a YAML mapping from ``path``, treating an empty document as an empty mapping.

    This is the *partial* read that :meth:`SubsetConfig.from_yaml` is built on, and the one a
    command line wants: a config file that a user will complete with flags is not yet a valid
    config, so it must be readable without being validated.

    The file is decoded as UTF-8 rather than through the locale's preferred codec, so the same bytes
    yield the same config -- and therefore the same salt, ranks, and shard assignment -- everywhere.
    PyYAML's own parse errors name the document ``<unicode string>``, which is useless to a user
    staring at a path, so they are re-raised naming the file.

    Args:
        path: The YAML file to read.

    Returns:
        The mapping it contains; ``{}`` for an empty document.

    Raises:
        ValueError: If the file is not valid YAML, or does not contain a mapping.
        OSError: If the file cannot be read.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> p = Path(tmp.name) / "cfg.yaml"
        >>> _ = p.write_text("salt: fam1\nn_subjects_per_shard: 25\n")
        >>> load_yaml_mapping(p)
        {'salt': 'fam1', 'n_subjects_per_shard': 25}

        A partial config is fine -- completing it is the caller's job:

        >>> _ = p.write_text("salt: fam1\n")
        >>> load_yaml_mapping(p)
        {'salt': 'fam1'}

        An empty document is an empty mapping, not an error:

        >>> _ = p.write_text("")
        >>> load_yaml_mapping(p)
        {}

        A non-mapping document, and malformed YAML, both name the file:

        >>> _ = p.write_text("- a\n- b\n")
        >>> load_yaml_mapping(p)
        Traceback (most recent call last):
            ...
        ValueError: /...cfg.yaml must contain a YAML mapping; got list
        >>> _ = p.write_text("salt: [unclosed\n")
        >>> load_yaml_mapping(p)
        Traceback (most recent call last):
            ...
        ValueError: /...cfg.yaml is not valid YAML: while parsing a flow sequence...
        >>> tmp.cleanup()
    """
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"{p} is not valid YAML: {e}") from e
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a YAML mapping; got {type(raw).__name__}")
    return raw


#: Backwards-compatible private alias; :func:`load_yaml_mapping` is the supported name.
_load_yaml = load_yaml_mapping


def _as_positive_int(name: str, value: Any) -> int:
    """Return ``value`` as a builtin positive ``int``, raising ``ValueError`` if it is not one.

    Anything that is losslessly an integer (``__index__``, so a ``numpy`` scalar included) is accepted and
    narrowed to a builtin ``int``; ``bool`` is not, since ``True`` as a size is always a mistake. Narrowing
    here is what keeps ``to_dict`` JSON- and YAML-native, and rejecting integer-*valued* floats and strings
    here is what keeps every later stage from having to re-check.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    try:
        out = operator.index(value)
    except TypeError as e:
        raise ValueError(f"{name} must be an integer; got {value!r}") from e
    if out < 1:
        raise ValueError(f"{name} must be positive; got {out!r}")
    return out


def _require_non_empty(name: str, value: str) -> None:
    """Raise unless ``value`` is a non-empty string."""
    if not value:
        raise ValueError(f"{name} must be non-empty; got {value!r}")


def _require_template(name: str, template: str, key: str, values: Iterable[Any]) -> None:
    """Raise unless ``template`` consumes ``{key}`` and renders a distinct name for each of ``values``.

    The ``{key}`` check parses the template's replacement fields instead of looking for a substring, so
    an escaped ``{{key}}`` -- which renders literally, and so names every member of the family the same
    thing -- is rejected rather than accepted. Rendering every configured value then catches both a
    malformed template and a lossy one that would collapse two members onto one directory.
    """
    try:
        fields = {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    except ValueError as e:
        raise ValueError(
            f"{name} must be a valid format string; got {template!r} ({type(e).__name__}: {e})"
        ) from e
    if key not in {f.split("[")[0].split(".")[0] for f in fields}:
        raise ValueError(f"{name} must contain '{{{key}'; got {template!r}")

    seen: dict[str, Any] = {}
    for value in values:
        try:
            rendered = template.format(**{key: value})
        except (IndexError, KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"{name} must be a valid format string; got {template!r} ({type(e).__name__}: {e})"
            ) from e
        if rendered in seen:
            raise ValueError(
                f"{name} must render a distinct name per {key}; {template!r} renders both "
                f"{seen[rendered]!r} and {value!r} as {rendered!r}"
            )
        seen[rendered] = value


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SubsetConfig:
    """Defines a nested family of subject subsets and how it is laid out on disk.

    Attributes:
        n_subjects: The train-split sizes to build, in subjects. Stored sorted ascending, and required to
            be a non-empty set of distinct positive integers: the family is nested by construction, so
            the ``N=10`` member's subjects are exactly the first 10 of the ``N=100`` member's, and a
            repeated size would name the same directory twice.
        salt: Mixed into each subject's rank key. It fixes which subjects are chosen and which shard each
            lands in, so it is recorded in the family manifest and must not change within a family.
        n_subjects_per_shard: Subjects per output shard. Shard membership is ``rank // this``, which is
            why every full shard is identical across members and shareable; only each member's final,
            partial shard is its own. It must therefore also be held constant within a family.
        link_mode: How subset roots reference the shared shard store. See :class:`LinkMode`.
        code_metadata: What to write as the subset's code metadata. See :class:`CodeMetadataPolicy`.
        non_train_policy: What to do with the non-train splits. See :class:`NonTrainPolicy`.
        train_split: The split that is actually subsetted. ``"train"`` matches ``meds.train_split``, but
            any string is a legal MEDS split label, so this is configurable.
        name_template: ``str.format`` template for member directory names, rendered with ``n_subjects``.
            It must consume ``{n_subjects}`` and render a distinct name for every configured size, since
            two members sharing a directory name would silently overwrite each other. The zero-padded
            default makes lexicographic order agree with sweep order.

    Examples:
        >>> cfg = SubsetConfig(n_subjects=(100, 10))
        >>> cfg.n_subjects
        (10, 100)
        >>> cfg.salt, cfg.n_subjects_per_shard, cfg.train_split
        ('meds_subsetter', 10000, 'train')
        >>> cfg.link_mode, cfg.code_metadata
        (<LinkMode.SYMLINK: 'symlink'>, <CodeMetadataPolicy.PRUNE: 'prune'>)
        >>> cfg.non_train_policy
        <NonTrainPolicy.PASSTHROUGH: 'passthrough'>

        Policy fields accept their string spellings, so a mapping parsed from YAML needs no
        pre-processing, and an unrecognized spelling fails at construction rather than at write time:

        >>> SubsetConfig(n_subjects=(10,), link_mode="hardlink").link_mode
        <LinkMode.HARDLINK: 'hardlink'>
        >>> SubsetConfig(n_subjects=(10,), code_metadata="copy").code_metadata
        <CodeMetadataPolicy.COPY: 'copy'>
        >>> SubsetConfig(n_subjects=(10,), link_mode="hard-link")
        Traceback (most recent call last):
            ...
        ValueError: 'hard-link' is not a valid LinkMode

        Configs are frozen, so no stage can retune the salt or the shard size mid-family and silently
        break the disk sharing that the family depends on:

        >>> cfg.salt = "other"
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'salt'

        The size list must be a non-empty set of distinct positive integers:

        >>> SubsetConfig(n_subjects=())
        Traceback (most recent call last):
            ...
        ValueError: n_subjects must be non-empty; got ()
        >>> SubsetConfig(n_subjects=(10, 0))
        Traceback (most recent call last):
            ...
        ValueError: n_subjects entries must be positive; got 0
        >>> SubsetConfig(n_subjects=(10, 100, 10))
        Traceback (most recent call last):
            ...
        ValueError: n_subjects must be strictly increasing once sorted; got (10, 10, 100)

        Sizes must be *integers*, not merely integer-valued. A size is a row count, a slice bound and a
        zero-padded directory name, so a float or a string that survived construction would fail far
        downstream instead of here -- and YAML makes both easy to write by accident (``1000.0`` parses as
        a float, ``1.0e3`` as a bare string):

        >>> SubsetConfig(n_subjects=(1000.0,))
        Traceback (most recent call last):
            ...
        ValueError: n_subjects entries must be an integer; got 1000.0
        >>> SubsetConfig(n_subjects=("1.0e3",))
        Traceback (most recent call last):
            ...
        ValueError: n_subjects entries must be an integer; got '1.0e3'
        >>> SubsetConfig(n_subjects=(True,))
        Traceback (most recent call last):
            ...
        ValueError: n_subjects entries must be an integer; got True

        Anything that is losslessly an integer is accepted and stored as a builtin ``int`` -- notably a
        ``numpy`` scalar, since ``np.logspace(...).astype(int)`` is the obvious way to write a log sweep
        -- so :meth:`to_dict` stays JSON- and YAML-native whatever the caller passed:

        >>> class Int64:  # a stand-in for a numpy integer scalar
        ...     def __index__(self) -> int:
        ...         return 100
        >>> cfg2 = SubsetConfig(n_subjects=[Int64()])
        >>> cfg2.n_subjects, [type(n).__name__ for n in cfg2.n_subjects]
        ((100,), ['int'])

        A bare scalar where a sequence belongs is the likeliest YAML typo of all, and it names itself
        rather than failing as ``'int' object is not iterable``:

        >>> SubsetConfig(n_subjects=1000)
        Traceback (most recent call last):
            ...
        ValueError: n_subjects must be a sequence of positive integers; got 1000

        The remaining scalars are checked one by one:

        >>> SubsetConfig(n_subjects=(10,), salt="")
        Traceback (most recent call last):
            ...
        ValueError: salt must be non-empty; got ''
        >>> SubsetConfig(n_subjects=(10,), n_subjects_per_shard=0)
        Traceback (most recent call last):
            ...
        ValueError: n_subjects_per_shard must be positive; got 0
        >>> SubsetConfig(n_subjects=(10,), n_subjects_per_shard=2.5)
        Traceback (most recent call last):
            ...
        ValueError: n_subjects_per_shard must be an integer; got 2.5
        >>> SubsetConfig(n_subjects=(10,), train_split="")
        Traceback (most recent call last):
            ...
        ValueError: train_split must be non-empty; got ''

        A template that does not consume ``n_subjects`` would write every member of the family to the
        same directory, so it is rejected. The check parses the template's replacement fields rather than
        looking for a substring, so an escaped ``{{n_subjects}}`` -- which renders literally, and
        collides just as badly -- is caught too:

        >>> SubsetConfig(n_subjects=(10,), name_template="subset")
        Traceback (most recent call last):
            ...
        ValueError: name_template must contain '{n_subjects'; got 'subset'
        >>> SubsetConfig(n_subjects=(10, 100), name_template="N{{n_subjects}}")
        Traceback (most recent call last):
            ...
        ValueError: name_template must contain '{n_subjects'; got 'N{{n_subjects}}'

        The template is then validated by actually rendering every configured size, which catches a
        malformed template, and also a well-formed but *lossy* one that would collapse two members of
        this particular family onto one directory -- both before any data is written:

        >>> SubsetConfig(n_subjects=(10,), name_template="N{n_subjects}-{fold}")
        Traceback (most recent call last):
            ...
        ValueError: name_template must be a valid format string; got 'N{n_subjects}-{fold}' (KeyError: 'fold')
        >>> SubsetConfig(n_subjects=(10,), name_template="N{n_subjects")
        Traceback (most recent call last):
            ...
        ValueError: name_template must be a valid format string; got 'N{n_subjects' (ValueError: ...)
        >>> SubsetConfig(n_subjects=(10,), name_template="N{n_subjects[0]}")
        Traceback (most recent call last):
            ...
        ValueError: name_template must be a valid format string; got 'N{n_subjects[0]}' (TypeError: ...)
        >>> SubsetConfig(n_subjects=(100, 110), name_template="N{n_subjects:.1g}")
        Traceback (most recent call last):
            ...
        ValueError: name_template must render a distinct name per n_subjects;
        'N{n_subjects:.1g}' renders both 100 and 110 as 'N1e+02'
    """

    n_subjects: tuple[int, ...]
    salt: str = DEFAULT_SALT
    n_subjects_per_shard: int = 10000
    link_mode: LinkMode = LinkMode.SYMLINK
    code_metadata: CodeMetadataPolicy = CodeMetadataPolicy.PRUNE
    non_train_policy: NonTrainPolicy = NonTrainPolicy.PASSTHROUGH
    train_split: str = "train"
    name_template: str = "N{n_subjects:07d}"

    def __post_init__(self) -> None:
        """Validate the config, coercing policy strings to enum members and ``n_subjects`` to a tuple."""
        if isinstance(self.n_subjects, str | bytes) or not isinstance(self.n_subjects, Iterable):
            raise ValueError(f"n_subjects must be a sequence of positive integers; got {self.n_subjects!r}")
        n_subjects = tuple(self.n_subjects)
        if not n_subjects:
            raise ValueError(f"n_subjects must be non-empty; got {n_subjects!r}")
        ordered = tuple(sorted(_as_positive_int("n_subjects entries", n) for n in n_subjects))
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"n_subjects must be strictly increasing once sorted; got {ordered!r}")
        _require_non_empty("salt", self.salt)
        n_per_shard = _as_positive_int("n_subjects_per_shard", self.n_subjects_per_shard)
        _require_non_empty("train_split", self.train_split)
        _require_template("name_template", self.name_template, "n_subjects", ordered)

        object.__setattr__(self, "n_subjects", ordered)
        object.__setattr__(self, "n_subjects_per_shard", n_per_shard)
        object.__setattr__(self, "link_mode", LinkMode(self.link_mode))
        object.__setattr__(self, "code_metadata", CodeMetadataPolicy(self.code_metadata))
        object.__setattr__(self, "non_train_policy", NonTrainPolicy(self.non_train_policy))

    def subset_name(self, n: int) -> str:
        """Render the directory name of the ``n``-subject member.

        Args:
            n: The member size, in subjects.

        Returns:
            The rendered directory name.

        Examples:
            >>> cfg = SubsetConfig(n_subjects=(10, 1000))
            >>> cfg.subset_name(10)
            'N0000010'
            >>> [cfg.subset_name(n) for n in cfg.n_subjects]
            ['N0000010', 'N0001000']

            The zero-padded default sorts lexicographically in sweep order:

            >>> sorted(cfg.subset_name(n) for n in (10, 1000, 200))
            ['N0000010', 'N0000200', 'N0001000']

            Any size renders, not only the configured ones, since the same template is used to locate an
            existing member when extending a family:

            >>> SubsetConfig(n_subjects=(10,), name_template="train_{n_subjects}").subset_name(42)
            'train_42'
        """
        return self.name_template.format(n_subjects=n)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain-mapping form embedded in the family manifest.

        Returns:
            A dict with sorted keys, tuples rendered as lists and enum members as their string values.

        Examples:
            >>> from pprint import pprint
            >>> pprint(SubsetConfig(n_subjects=(10, 100), code_metadata="copy").to_dict())
            {'code_metadata': 'copy',
             'link_mode': 'symlink',
             'n_subjects': [10, 100],
             'n_subjects_per_shard': 10000,
             'name_template': 'N{n_subjects:07d}',
             'non_train_policy': 'passthrough',
             'salt': 'meds_subsetter',
             'train_split': 'train'}

            ``pprint`` sorts keys itself, so the ordering above proves nothing on its own; the mapping
            really is emitted in sorted order, which keeps a manifest's diff stable across versions:

            >>> d = SubsetConfig(n_subjects=(10, 100), code_metadata="copy").to_dict()
            >>> list(d) == sorted(d)
            True

            Enum members are flattened to plain strings, so the result survives ``yaml.safe_dump``,
            which refuses a bare ``StrEnum``:

            >>> import yaml
            >>> d = SubsetConfig(n_subjects=(10,)).to_dict()
            >>> yaml.safe_load(yaml.safe_dump(d)) == d
            True

            The round trip is exact, which is what lets a recorded family be rebuilt:

            >>> cfg = SubsetConfig(n_subjects=(10, 100), link_mode="hardlink", salt="s")
            >>> SubsetConfig.from_dict(cfg.to_dict()) == cfg
            True
        """
        return _as_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SubsetConfig:
        """Build a config from a plain mapping, rejecting unknown keys.

        Args:
            raw: The mapping to read. ``n_subjects`` may be any sequence; policy fields may be given as
                strings.

        Returns:
            The parsed config.

        Raises:
            ValueError: If ``raw`` is not a mapping, carries a key this version does not define, or
                carries values that do not validate.
            TypeError: If a required key is absent.

        Examples:
            >>> SubsetConfig.from_dict({"n_subjects": [100, 10]}).n_subjects
            (10, 100)
            >>> SubsetConfig.from_dict({"n_subjects": [10], "link_mode": "copy"}).link_mode
            <LinkMode.COPY: 'copy'>

            A typo fails loudly rather than silently leaving the default in place -- a quietly ignored
            ``n_subject_per_shard`` would mean building a family whose shards nothing else can share:

            >>> SubsetConfig.from_dict({"n_subjects": [10], "n_subject_per_shard": 5})
            Traceback (most recent call last):
                ...
            ValueError: Unknown SubsetConfig keys: ['n_subject_per_shard']

            YAML mappings may carry non-string keys, so the report is built from the ``repr`` of each
            key and cannot itself fail on an unorderable mix:

            >>> SubsetConfig.from_dict({"n_subjects": [10], 1: "oops", "bogus": 2})
            Traceback (most recent call last):
                ...
            ValueError: Unknown SubsetConfig keys: ['bogus', 1]

            Something that is not a mapping at all is named as such, rather than being iterated into a
            nonsense key report:

            >>> SubsetConfig.from_dict([10, 100])
            Traceback (most recent call last):
                ...
            ValueError: SubsetConfig must be built from a mapping; got list

            Required keys stay required:

            >>> SubsetConfig.from_dict({})
            Traceback (most recent call last):
                ...
            TypeError: ...missing 1 required keyword-only argument: 'n_subjects'
        """
        return cls(**_checked_kwargs(cls, raw))

    @classmethod
    def from_yaml(cls, path: str | Path) -> SubsetConfig:
        r"""Load a config from a YAML file.

        Args:
            path: The YAML file to read.

        Returns:
            The parsed config.

        Raises:
            ValueError: If the document is not a mapping, carries unknown keys, or carries values that
                do not validate.
            TypeError: If a required key is absent.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / "subset.yaml"
            ...     _ = path.write_text("n_subjects: [100, 10]\nlink_mode: hardlink\n")
            ...     cfg = SubsetConfig.from_yaml(path)
            >>> cfg.n_subjects
            (10, 100)
            >>> cfg.link_mode
            <LinkMode.HARDLINK: 'hardlink'>

            The file is decoded as UTF-8 regardless of the machine's locale. ``salt`` fixes which
            subjects are chosen and which shard each lands in, so the same YAML bytes must yield the same
            family everywhere they are read:

            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / "subset.yaml"
            ...     _ = path.write_text("n_subjects: [10]\nsalt: café\n", encoding="utf-8")
            ...     SubsetConfig.from_yaml(path).salt
            'café'

            A document that is not a mapping is reported as such, instead of being iterated
            character-by-character into a nonsense list of unknown keys:

            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / "subset.yaml"
            ...     _ = path.write_text("a stray string\n")
            ...     SubsetConfig.from_yaml(path)
            Traceback (most recent call last):
                ...
            ValueError: ...subset.yaml must contain a YAML mapping; got str

            An empty document is an empty mapping, so the missing required key is what gets reported:

            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / "subset.yaml"
            ...     _ = path.write_text("")
            ...     SubsetConfig.from_yaml(path)
            Traceback (most recent call last):
                ...
            TypeError: ...missing 1 required keyword-only argument: 'n_subjects'
        """
        return cls.from_dict(_load_yaml(path))


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ResampleConfig:
    """Defines a family of resampled evaluation manifests drawn from a parent split.

    A resample is emitted as a *manifest* over the unmodified parent, never as a MEDS root: multiplicity
    cannot survive one, since ``SubjectSplitSchema`` is closed and every consumer deduplicates. The
    manifest carries the repeated subject ids, so the draws stay faithful.

    Attributes:
        n_resamples: How many replicates to emit. Each becomes one manifest directory.
        size: Subjects drawn per replicate. Without replacement this may not exceed the eligible pool,
            but that comparison needs the pool, so it happens at draw time rather than here.
        replace: Whether to draw with replacement -- the classic bootstrap. Draws come from a sha256
            counter stream with rejection sampling rather than ``numpy.random``, whose ``Generator``
            streams are not promised stable across numpy versions.
        salt: Mixed into that counter stream. Fixing it fixes the draws exactly.
        allow_overlap: Whether a replicate may draw subjects that are also in the training subset. Off by
            default: nothing in the MEDS ecosystem enforces train/test disjointness, so an evaluation
            resample that quietly overlaps training is an easy and invisible mistake.
        source_split: The parent split the pool is drawn from.
        name_template: ``str.format`` template for replicate directory names, rendered with ``index``.
            It must consume ``{index}`` and render a distinct name for each of the ``n_resamples``
            replicates.

    Examples:
        >>> cfg = ResampleConfig(n_resamples=100, size=500)
        >>> cfg.replace, cfg.allow_overlap
        (True, False)
        >>> cfg.salt, cfg.source_split
        ('meds_subsetter', 'train')
        >>> [cfg.resample_name(i) for i in range(3)]
        ['boot_000', 'boot_001', 'boot_002']

        Frozen, like :class:`SubsetConfig`:

        >>> cfg.size = 10
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'size'

        Every scalar is checked at construction:

        >>> ResampleConfig(n_resamples=0, size=10)
        Traceback (most recent call last):
            ...
        ValueError: n_resamples must be positive; got 0
        >>> ResampleConfig(n_resamples=10, size=0)
        Traceback (most recent call last):
            ...
        ValueError: size must be positive; got 0
        >>> ResampleConfig(n_resamples=10, size=10, salt="")
        Traceback (most recent call last):
            ...
        ValueError: salt must be non-empty; got ''
        >>> ResampleConfig(n_resamples=10, size=10, source_split="")
        Traceback (most recent call last):
            ...
        ValueError: source_split must be non-empty; got ''

        Both counts must be integers, not merely integer-valued -- they index a counter stream and bound
        a draw loop, and a float would fail only once sampling had started:

        >>> ResampleConfig(n_resamples=3.0, size=10)
        Traceback (most recent call last):
            ...
        ValueError: n_resamples must be an integer; got 3.0
        >>> ResampleConfig(n_resamples=3, size="50")
        Traceback (most recent call last):
            ...
        ValueError: size must be an integer; got '50'

        As is the name template, which must vary with the replicate index -- checked by parsing its
        replacement fields, so an escaped ``{{index}}`` does not slip through, and then by rendering
        every replicate:

        >>> ResampleConfig(n_resamples=10, size=10, name_template="boot")
        Traceback (most recent call last):
            ...
        ValueError: name_template must contain '{index'; got 'boot'
        >>> ResampleConfig(n_resamples=3, size=10, name_template="boot_{{index}}")
        Traceback (most recent call last):
            ...
        ValueError: name_template must contain '{index'; got 'boot_{{index}}'
        >>> ResampleConfig(n_resamples=10, size=10, name_template="boot_{index}_{seed}")
        Traceback (most recent call last):
            ...
        ValueError: name_template must be a valid format string; got 'boot_{index}_{seed}' (KeyError: 'seed')
        >>> ResampleConfig(n_resamples=20, size=10, name_template="boot_{index:.1g}")
        Traceback (most recent call last):
            ...
        ValueError: name_template must render a distinct name per index;
        'boot_{index:.1g}' renders both 10 and 11 as 'boot_1e+01'
    """

    n_resamples: int
    size: int
    replace: bool = True
    salt: str = DEFAULT_SALT
    allow_overlap: bool = False
    source_split: str = "train"
    name_template: str = "boot_{index:03d}"

    def __post_init__(self) -> None:
        """Validate the config, narrowing the two counts to builtin ``int``."""
        n_resamples = _as_positive_int("n_resamples", self.n_resamples)
        size = _as_positive_int("size", self.size)
        _require_non_empty("salt", self.salt)
        _require_non_empty("source_split", self.source_split)
        _require_template("name_template", self.name_template, "index", range(n_resamples))

        object.__setattr__(self, "n_resamples", n_resamples)
        object.__setattr__(self, "size", size)

    def resample_name(self, i: int) -> str:
        """Render the directory name of replicate ``i``.

        Args:
            i: The zero-based replicate index.

        Returns:
            The rendered directory name.

        Examples:
            >>> cfg = ResampleConfig(n_resamples=1000, size=10)
            >>> cfg.resample_name(0)
            'boot_000'

            The default pads to three digits, so replicate counts above 1000 widen rather than collide:

            >>> cfg.resample_name(7), cfg.resample_name(1234)
            ('boot_007', 'boot_1234')
            >>> ResampleConfig(n_resamples=5, size=10, name_template="rep{index}").resample_name(4)
            'rep4'
        """
        return self.name_template.format(index=i)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain-mapping form embedded in the resample manifest.

        Returns:
            A dict with sorted keys and JSON/YAML-native values.

        Examples:
            >>> from pprint import pprint
            >>> pprint(ResampleConfig(n_resamples=3, size=50, replace=False).to_dict())
            {'allow_overlap': False,
             'n_resamples': 3,
             'name_template': 'boot_{index:03d}',
             'replace': False,
             'salt': 'meds_subsetter',
             'size': 50,
             'source_split': 'train'}

            ``pprint`` sorts keys itself, so the ordering above is not evidence; the mapping really is
            emitted sorted:

            >>> d = ResampleConfig(n_resamples=3, size=50, replace=False).to_dict()
            >>> list(d) == sorted(d)
            True

            >>> cfg = ResampleConfig(n_resamples=3, size=50, salt="s", allow_overlap=True)
            >>> ResampleConfig.from_dict(cfg.to_dict()) == cfg
            True
        """
        return _as_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResampleConfig:
        """Build a config from a plain mapping, rejecting unknown keys.

        Args:
            raw: The mapping to read.

        Returns:
            The parsed config.

        Raises:
            ValueError: If ``raw`` is not a mapping, carries a key this version does not define, or
                carries values that do not validate.
            TypeError: If a required key is absent.

        Examples:
            >>> ResampleConfig.from_dict({"n_resamples": 3, "size": 50}).n_resamples
            3
            >>> ResampleConfig.from_dict({"n_resamples": 3, "size": 50, "n_draws": 3})
            Traceback (most recent call last):
                ...
            ValueError: Unknown ResampleConfig keys: ['n_draws']
            >>> ResampleConfig.from_dict("n_resamples: 3")
            Traceback (most recent call last):
                ...
            ValueError: ResampleConfig must be built from a mapping; got str
        """
        return cls(**_checked_kwargs(cls, raw))

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResampleConfig:
        r"""Load a config from a YAML file.

        Args:
            path: The YAML file to read.

        Returns:
            The parsed config.

        Raises:
            ValueError: If the document is not a mapping, carries unknown keys, or carries values that
                do not validate.
            TypeError: If a required key is absent.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / "resample.yaml"
            ...     _ = path.write_text("n_resamples: 200\nsize: 64\nreplace: false\n")
            ...     cfg = ResampleConfig.from_yaml(path)
            >>> cfg.n_resamples, cfg.size, cfg.replace
            (200, 64, False)

            As for :meth:`SubsetConfig.from_yaml`, the file is decoded as UTF-8 and the document must be
            a mapping:

            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / "resample.yaml"
            ...     _ = path.write_text("- 200\n- 64\n")
            ...     ResampleConfig.from_yaml(path)
            Traceback (most recent call last):
                ...
            ValueError: ...resample.yaml must contain a YAML mapping; got list
        """
        return cls.from_dict(_load_yaml(path))
