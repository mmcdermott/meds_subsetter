"""Command-line entry points.

Four commands, registered as console scripts in ``pyproject.toml`` and each also reachable as
``python -m meds_subsetter <subcommand>``:

* ``meds-subset`` / ``subset`` -- build a nested family of subject subsets, sharing shards on disk.
* ``meds-resample`` / ``resample`` -- draw resampled (bootstrap) evaluation manifests.
* ``meds-size`` / ``size`` -- report exact dataset sizes as a tidy parquet plus a JSON sidecar.
* ``meds-fingerprint`` / ``fingerprint`` -- report a dataset's and/or an index frame's content ids.

Every ``*_main`` takes an explicit argument list and returns an exit code rather than calling
``sys.exit``, which is what lets the whole CLI surface be exercised from doctests. An input that
cannot be used prints ``error: <message>`` to stderr and returns :data:`EXIT_INPUT_ERROR` -- never a
traceback; a malformed command line is argparse's own exit code ``2``, and its usage text goes to
stderr, as argparse's does. "Cannot be used" is :data:`_INPUT_ERRORS`, and it deliberately includes
``OSError``: a ``--config`` that is a directory or an ``out_root`` that is an existing file are as
much user mistakes as a path that is not there at all.

The two commands that build things take ``--config``, the YAML form of the frozen config they are
driven by (:class:`~meds_subsetter.config.SubsetConfig`,
:class:`~meds_subsetter.config.ResampleConfig`), and one flag per field of it. The file supplies a
*partial* mapping and the flags override the fields they name, so only the merged result has to be a
complete, valid config -- which is what makes "pin the salt and the shard size in a file, vary the
sweep on the command line" work. A flag counts as an override when it is *present*, never when its
value merely differs from the default: every such flag parses to ``None`` when absent, so passing one
its own default value is still an explicit choice and still wins over the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import meds
import polars as pl
import yaml

from . import __package_name__, __version__

# `_load_yaml` is imported deliberately: `--config` must be readable as a *partial* mapping, since the
# flags are what complete it, and only the merged mapping is a whole config. Reading it here rather
# than through `SubsetConfig.from_yaml` is the difference between "the file may set what it likes" and
# "the file must already be complete"; doing it with the config module's own reader is what keeps the
# UTF-8 decoding, the empty-document rule, and the not-a-mapping message identical either way. See
# `needs_from_others`: config wants a public reader for a partial mapping.
from .config import (
    CodeMetadataPolicy,
    LinkMode,
    NonTrainPolicy,
    ResampleConfig,
    SubsetConfig,
    _load_yaml,
)
from .digests import frame_digest, pinned_schema, short_id
from .materialize import atomic_write_json, atomic_write_parquet
from .resample import RESAMPLES_SUBDIR, build_resamples

# `meds-fingerprint` must report the ids `meds-subset` *records*, so it calls the builder's own
# `subset.fingerprint` rather than reimplementing the computation, which could drift from it.
from .sizes import TENSORIZED_SCHEMAS_SUBDIR, _parquet_shard_paths, size_meds, size_tensorized
from .subset import (
    FAMILY_MANIFEST_NAME,
    NotAMEDSDatasetError,
    build_family,
    fingerprint,
)
from .tensorized import build_tensorized_family, is_tensorized_cohort

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .resample import ResampleFamilyManifest
    from .sizes import SizeReport
    from .subset import FamilyManifest

#: Exit code for an input that could not be used: a directory that is not a MEDS dataset, a config
#: that does not resolve, a draw that cannot be made. Distinct from ``2``, which argparse uses for a
#: malformed command line.
EXIT_INPUT_ERROR = 3

#: What "an input that cannot be used" looks like as an exception, and therefore what turns into
#: ``error: <message>`` and :data:`EXIT_INPUT_ERROR` instead of a traceback.
#: :class:`~meds_subsetter.subset.NotAMEDSDatasetError` is itself a ``ValueError`` and is named only
#: because it is the one the user meets most. ``OSError`` is the whole "the path is there but is not
#: what it has to be" family -- ``IsADirectoryError`` for a ``--config`` that names a directory,
#: ``NotADirectoryError`` for an ``out_root`` that is an existing file, ``PermissionError``,
#: ``FileNotFoundError`` -- every one of which is a user mistake and none of which is a bug to report
#: with a stack trace.
_INPUT_ERRORS = (NotAMEDSDatasetError, OSError, ValueError)

#: Defaults shown in ``--help``. They are *not* installed as argparse defaults: an override-able flag
#: parses to ``None`` when absent so that presence, rather than a value comparison, decides whether it
#: overrides ``--config``. A ``slots=True`` dataclass replaces its class attributes with slot
#: descriptors, so the defaults have to be read off an instance. ``SubsetConfig`` has no default for
#: ``n_subjects`` and ``ResampleConfig`` none for ``n_resamples``/``size``; those placeholder values
#: are never read.
_SUBSET_DEFAULTS = SubsetConfig(n_subjects=(1,))
_RESAMPLE_DEFAULTS = ResampleConfig(n_resamples=1, size=1)

#: The subcommand names :func:`main` dispatches on.
_COMMANDS = ("subset", "resample", "size", "fingerprint")

#: The columns the human-readable size table shows. The full eleven are in the parquet and the JSON
#: sidecar; a terminal summary that wraps is no summary.
_SUMMARY_COLUMNS = ("split", "n_subjects", "n_events", "n_measurements", "n_shards", "n_bytes")


def _report_input_error(e: Exception) -> int:
    """Print an unusable input as ``error: <message>`` on stderr, and return the exit code for it.

    Every command reports bad input the same way, through here, so none of them can drift into a
    traceback or into a different exit code for the same class of mistake.

    Args:
        e: The exception, one of :data:`_INPUT_ERRORS`.

    Returns:
        :data:`EXIT_INPUT_ERROR`, so a caller can say ``return _report_input_error(e)``.

    Examples:
        >>> import contextlib, io
        >>> err = io.StringIO()
        >>> with contextlib.redirect_stderr(err):
        ...     _report_input_error(FileNotFoundError("/tmp/nope does not exist"))
        3
        >>> print(err.getvalue(), end="")
        error: /tmp/nope does not exist

        A message the builders wrote for a Python caller is re-worded for one holding a command line.
        There is no ``--do-overwrite`` flag, and the one that does that job is ``--force``:

        The builders phrase their refusals so that both callers are served, so nothing is rewritten
        on the way out:

        >>> err = io.StringIO()
        >>> with contextlib.redirect_stderr(err):
        ...     _ = _report_input_error(
        ...         ValueError("Rebuild with overwrite enabled (--force, or do_overwrite=True).")
        ...     )
        >>> print(err.getvalue(), end="")
        error: Rebuild with overwrite enabled (--force, or do_overwrite=True).
    """
    print(f"error: {e}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _config_mapping(path: Path | None) -> dict[str, Any]:
    """Read a ``--config`` file as a raw mapping, without requiring it to be a whole config on its own.

    The flags are what complete it, so validation waits until the two have been merged; all this does
    is get the YAML off disk and say whose fault it is when that fails.

    Args:
        path: The file ``--config`` named, or ``None`` when the flag was not passed.

    Returns:
        The mapping the file holds; an empty one when there is no file.

    Raises:
        ValueError: If the file is not valid YAML, or does not hold a mapping.
        OSError: If it cannot be read at all -- it is missing, or it is a directory.

    Examples:
        >>> _config_mapping(None)
        {}
        >>> with yaml_disk({"c.yaml": {"salt": "s", "n_subjects": [2]}}) as root:
        ...     sorted(_config_mapping(root / "c.yaml").items())
        [('n_subjects', [2]), ('salt', 's')]

        Unknown keys are deliberately *not* rejected here. That is the merged config's job, and doing
        it here would report a key as unknown before the merge that defines the key set has happened:

        >>> with yaml_disk({"c.yaml": {"bogus": 1}}) as root:
        ...     _config_mapping(root / "c.yaml")
        {'bogus': 1}

        A YAML error names the file. PyYAML's own message calls it ``<unicode string>``, which is no
        help to someone holding a directory of configs:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "c.yaml"
        ...     _ = path.write_text("salt: [oops")
        ...     _config_mapping(path)
        Traceback (most recent call last):
            ...
        ValueError: ...c.yaml is not valid YAML: while parsing a flow sequence...
    """
    if path is None:
        return {}
    try:
        return _load_yaml(path)
    except yaml.YAMLError as e:
        raise ValueError(f"{path} is not valid YAML: {e}") from e


def _passed_overrides(args: argparse.Namespace, fields: Sequence[str]) -> dict[str, Any]:
    """Collect the config fields the user actually passed on the command line.

    Presence is what "passed" means: every override-able flag parses to ``None`` when absent, so a
    flag given its own default value is still an override and still beats the config file. Deciding
    it by comparing against the default instead would silently discard ``--salt meds_subsetter``, and
    a discarded salt is a different set of subjects -- a wrong answer, quietly.

    Args:
        args: The parsed command line.
        fields: The config field names to look for. Each is also the ``dest`` of its flag.

    Returns:
        A mapping of only the fields that were passed.

    Examples:
        >>> args = _subset_parser().parse_args(["/d", "/o", "-n", "2", "--salt", "meds_subsetter"])
        >>> _passed_overrides(args, ("n_subjects", "salt", "train_split"))
        {'n_subjects': [2], 'salt': 'meds_subsetter'}
        >>> _passed_overrides(_subset_parser().parse_args(["/d", "/o"]), ("salt", "link_mode"))
        {}
    """
    return {field: getattr(args, field) for field in fields if getattr(args, field) is not None}


def _subset_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the subset command.

    Examples:
        >>> args = _subset_parser().parse_args(["/data/mimic", "/out/sweep", "-n", "10", "100"])
        >>> args.parent, args.out_root, args.n_subjects
        (PosixPath('/data/mimic'), PosixPath('/out/sweep'), [10, 100])
        >>> args.workers, args.worker, args.force, args.index_dir
        (1, 0, False, None)

        Every flag that ``--config`` can also set parses to ``None`` when it is absent. That absence,
        not a comparison against the default value, is what makes "the user passed this" detectable --
        so passing a flag its own default value is still an override:

        >>> args.salt, args.link_mode, args.code_metadata, args.non_train_policy
        (None, None, None, None)
        >>> args.n_subjects_per_shard, args.train_split, args.name_template
        (None, None, None)
        >>> _subset_parser().parse_args(["/d", "/o", "--salt", "meds_subsetter"]).salt
        'meds_subsetter'

        The defaults still live in the help text, which is where a user reads them:

        >>> help_text = " ".join(_subset_parser().format_help().split())
        >>> [d in help_text for d in ("(default: meds_subsetter)", "(default: 10000)",
        ...                           "(default: symlink)", "(default: prune)",
        ...                           "(default: passthrough)", "(default: train)",
        ...                           "(default: N{n_subjects:07d})")]
        [True, True, True, True, True, True, True]

        Sizes are optional on the command line, because ``--config`` may carry them:

        >>> _subset_parser().parse_args(["/d", "/o"]).n_subjects is None
        True
    """
    parser = argparse.ArgumentParser(
        prog="meds-subset",
        description="Build a nested family of MEDS subject subsets that share their shard files.",
    )
    parser.add_argument("parent", type=Path, help="Root of the parent MEDS dataset or tensorized cohort.")
    parser.add_argument(
        "--kind",
        choices=["auto", "meds", "tensorized"],
        default="auto",
        help=(
            "Which kind of parent this is. 'tensorized' subsets a MEDS-Torch-Data cohort in place by "
            "slicing its .nrt tensors, with no preprocessing run; 'meds' subsets a raw MEDS dataset. "
            "'auto' (the default) picks 'tensorized' when the parent has tokenization/schemas."
        ),
    )
    parser.add_argument(
        "out_root", type=Path, help="Directory to build the family in; family.json is written here."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="YAML SubsetConfig. The flags below override the fields they name.",
    )
    parser.add_argument(
        "-n",
        "--n-subjects",
        type=int,
        nargs="+",
        metavar="N",
        default=None,
        help="Train-split sizes to build, in subjects. Required unless --config sets n_subjects.",
    )
    parser.add_argument(
        "--salt",
        default=None,
        help=(
            "Mixed into each subject's rank key; fixes which subjects are chosen and which shard each "
            f"lands in (default: {_SUBSET_DEFAULTS.salt})."
        ),
    )
    parser.add_argument(
        "--n-subjects-per-shard",
        type=int,
        default=None,
        help=(
            "Subjects per output shard. Every full shard is then identical across the family, which is "
            f"what makes it shareable on disk (default: {_SUBSET_DEFAULTS.n_subjects_per_shard})."
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=tuple(m.value for m in LinkMode),
        default=None,
        help=(
            "How each member's shards reference the shared store "
            f"(default: {_SUBSET_DEFAULTS.link_mode.value})."
        ),
    )
    parser.add_argument(
        "--code-metadata",
        choices=tuple(m.value for m in CodeMetadataPolicy),
        default=None,
        help=(
            "What to write as each member's metadata/codes.parquet. 'prune' makes the vocabulary a "
            "function of N; 'copy' shares the parent's across the family "
            f"(default: {_SUBSET_DEFAULTS.code_metadata.value})."
        ),
    )
    parser.add_argument(
        "--non-train-policy",
        choices=tuple(m.value for m in NonTrainPolicy),
        default=None,
        help=(
            "What to do with the splits that are not being subsetted "
            f"(default: {_SUBSET_DEFAULTS.non_train_policy.value})."
        ),
    )
    parser.add_argument(
        "--train-split",
        default=None,
        help=f"The split that is actually subsetted (default: {_SUBSET_DEFAULTS.train_split}).",
    )
    parser.add_argument(
        "--name-template",
        default=None,
        help="str.format template for member directory names, rendered with n_subjects "
        f"(default: {_SUBSET_DEFAULTS.name_template}).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=(
            "Directory of task/index label parquet files. Each member gets its own slice under "
            "tasks/, resharded to match its data (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Name of the tasks/ subdirectory (default: the --index-dir directory's own name).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="How many processes are cooperating on this output root (default: %(default)s).",
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=0,
        help="This process's index in [0, workers) (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard and rebuild a family whose recorded parent no longer matches this one.",
    )
    return parser


def _resolve_subset_config(args: argparse.Namespace) -> SubsetConfig:
    """Merge the YAML config, if any, with command-line overrides.

    The file is read as a *partial* mapping and the flags are layered on top; only the merge is
    validated, as a whole, by :meth:`~meds_subsetter.config.SubsetConfig.from_dict`. So a file may set
    exactly the fields it wants pinned and leave ``n_subjects`` to ``-n``, and a missing required
    field is reported as the actionable message below rather than as a ``TypeError`` from the
    dataclass. Flags win over the file, but only where the user actually passed one: presence, not a
    comparison against the default, is what makes a flag an override (see :func:`_passed_overrides`).

    Args:
        args: The parsed command line.

    Returns:
        The resolved config.

    Raises:
        ValueError: If no size list is available from either source, if the file is not valid YAML, or
            if the merged mapping is not a valid config.
        OSError: If the ``--config`` file cannot be read.

    Examples:
        >>> parser = _subset_parser()
        >>> cfg = _resolve_subset_config(parser.parse_args(["/d", "/o", "-n", "100", "10"]))
        >>> cfg.n_subjects, cfg.salt
        ((10, 100), 'meds_subsetter')

        With a config file and no conflicting flag, the file wins:

        >>> spec = {"c.yaml": {"n_subjects": [7], "salt": "from_file", "code_metadata": "copy"}}
        >>> with yaml_disk(spec) as root:
        ...     cfg = _resolve_subset_config(parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml")]))
        >>> cfg.n_subjects, cfg.salt, cfg.code_metadata
        ((7,), 'from_file', <CodeMetadataPolicy.COPY: 'copy'>)

        With both, the explicit flag wins -- and only the fields actually passed change:

        >>> with yaml_disk(spec) as root:
        ...     cfg = _resolve_subset_config(
        ...         parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml"), "--salt", "cli"])
        ...     )
        >>> cfg.n_subjects, cfg.salt, cfg.code_metadata
        ((7,), 'cli', <CodeMetadataPolicy.COPY: 'copy'>)

        A flag overrides the file because it was *passed*, not because its value differs from the
        default. Passing a flag its own default value is still an explicit choice, and silently
        dropping it would change which subjects the family selects:

        >>> with yaml_disk(spec) as root:
        ...     cfg = _resolve_subset_config(parser.parse_args(
        ...         ["/d", "/o", "-c", str(root / "c.yaml"), "--salt", "meds_subsetter",
        ...          "--code-metadata", "prune"]))
        >>> cfg.salt, cfg.code_metadata
        ('meds_subsetter', <CodeMetadataPolicy.PRUNE: 'prune'>)

        The file does not have to be a whole config on its own -- only the merge does. That is the
        natural use of ``--config``: pin the settings that must not drift in a file, vary the sweep on
        the command line:

        >>> partial = {"c.yaml": {"salt": "from_file", "n_subjects_per_shard": 2}}
        >>> with yaml_disk(partial) as root:
        ...     cfg = _resolve_subset_config(
        ...         parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml"), "-n", "2"]))
        >>> cfg.n_subjects, cfg.salt, cfg.n_subjects_per_shard
        ((2,), 'from_file', 2)

        A partial file with nothing to fill the gap still reports the gap, not a ``TypeError``:

        >>> with yaml_disk(partial) as root:
        ...     _resolve_subset_config(parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml")]))
        Traceback (most recent call last):
            ...
        ValueError: No subset sizes: pass --n-subjects, or a --config that sets n_subjects.

        Only the merged mapping is validated, but it *is* validated: a typo in the file is still
        refused rather than silently ignored:

        >>> with yaml_disk({"c.yaml": {"n_subjects": [2], "n_subject_per_shard": 5}}) as root:
        ...     _resolve_subset_config(parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml")]))
        Traceback (most recent call last):
            ...
        ValueError: Unknown SubsetConfig keys: ['n_subject_per_shard']

        A file that is not YAML at all is named, which the parser's own message does not do:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "c.yaml"
        ...     _ = path.write_text("n_subjects: [2")
        ...     _resolve_subset_config(parser.parse_args(["/d", "/o", "-c", str(path)]))
        Traceback (most recent call last):
            ...
        ValueError: ...c.yaml is not valid YAML: while parsing a flow sequence...

        Every flag overrides independently:

        >>> cfg = _resolve_subset_config(parser.parse_args(
        ...     ["/d", "/o", "-n", "5", "--n-subjects-per-shard", "3", "--link-mode", "hardlink",
        ...      "--non-train-policy", "drop", "--train-split", "training",
        ...      "--name-template", "n{n_subjects}"]))
        >>> cfg.n_subjects_per_shard, cfg.link_mode, cfg.non_train_policy
        (3, <LinkMode.HARDLINK: 'hardlink'>, <NonTrainPolicy.DROP: 'drop'>)
        >>> cfg.train_split, cfg.subset_name(5)
        ('training', 'n5')

        Without sizes from either source there is nothing to build, and saying so here keeps the
        message actionable rather than a ``TypeError`` out of the dataclass:

        >>> _resolve_subset_config(parser.parse_args(["/d", "/o"]))
        Traceback (most recent call last):
            ...
        ValueError: No subset sizes: pass --n-subjects, or a --config that sets n_subjects.

        An invalid value is refused by the config itself, so the CLI never builds against one:

        >>> _resolve_subset_config(parser.parse_args(["/d", "/o", "-n", "0"]))
        Traceback (most recent call last):
            ...
        ValueError: n_subjects entries must be positive; got 0
    """
    base = _config_mapping(args.config)
    overrides = _passed_overrides(
        args,
        (
            "n_subjects",
            "salt",
            "n_subjects_per_shard",
            "link_mode",
            "code_metadata",
            "non_train_policy",
            "train_split",
            "name_template",
        ),
    )
    merged = {**base, **overrides}
    if "n_subjects" not in merged:
        raise ValueError("No subset sizes: pass --n-subjects, or a --config that sets n_subjects.")
    return SubsetConfig.from_dict(merged)


def _format_family(family: FamilyManifest) -> str:
    """Render a built family as a short human-readable block.

    Args:
        family: The manifest :func:`~meds_subsetter.subset.build_family` returned.

    Returns:
        The block, without a trailing newline.

    Examples:
        >>> from meds_subsetter.digests import digest
        >>> from meds_subsetter.subset import FamilyManifest, SubsetManifest
        >>> member = SubsetManifest(
        ...     name="N0000010", root="/out/N0000010", n_subjects=10,
        ...     subject_set_id=digest(["1"]), train_set_id=digest(["2"]),
        ...     content_ids={"train": digest(["2"])}, dataset_id=digest(["3"]),
        ...     codes_id=digest([]), observed_code_set_id=digest(["A"]), index_id=None,
        ...     n_index_rows=None, task_name=None, shards={"train/0": "train/s/r0-10.parquet"},
        ...     sizes={},
        ... )
        >>> family = FamilyManifest(
        ...     out_root="/out", parent="/parent", parent_ids={}, config={}, members=(member,),
        ...     disk={"n_bytes_on_disk": 40, "n_bytes_independent_copies": 100, "n_bytes_saved": 60,
        ...           "n_store_files": 3, "n_store_bytes": 40, "n_bytes_apparent": 100,
        ...           "n_bytes_cache": 7},
        ... )
        >>> print(_format_family(family))
        Built 1 member from /parent into /out
          N0000010  n_subjects=10  shards=1  train_set_id=673aeeb08cfbb00b
          disk: 40 bytes, vs 100 as independent copies (saved 60); store 3 files, cache 7 bytes
          manifest: /out/family.json
    """
    n_members = len(family.members)
    lines = [
        f"Built {n_members} member{'' if n_members == 1 else 's'} from {family.parent} into {family.out_root}"
    ]
    lines += [
        f"  {m.name}  n_subjects={m.n_subjects}  shards={len(m.shards)}  "
        f"train_set_id={short_id(m.train_set_id)}"
        for m in family.members
    ]
    disk = family.disk
    lines.append(
        f"  disk: {disk['n_bytes_on_disk']} bytes, vs {disk['n_bytes_independent_copies']} as "
        f"independent copies (saved {disk['n_bytes_saved']}); store {disk['n_store_files']} files, "
        f"cache {disk['n_bytes_cache']} bytes"
    )
    lines.append(f"  manifest: {Path(family.out_root) / FAMILY_MANIFEST_NAME}")
    return "\n".join(lines)


def subset_main(argv: list[str] | None = None) -> int:
    r"""Run the subset command.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.

    Examples:
        >>> import contextlib, io
        >>> def run(*args):
        ...     out, err = io.StringIO(), io.StringIO()
        ...     with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        ...         code = subset_main(list(args))
        ...     return code, out.getvalue(), err.getvalue()

        A two-member family, built into a fresh directory:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "sweep"
        >>> code, stdout, stderr = run(
        ...     str(simple_static_MEDS), str(out), "-n", "2", "4",
        ...     "--n-subjects-per-shard", "2", "--code-metadata", "copy",
        ... )
        >>> code
        0
        >>> print(stdout.replace(str(simple_static_MEDS), "<parent>").replace(str(out), "<out>"))
        Built 2 members from <parent> into <out>
          N0000002  n_subjects=2  shards=3  train_set_id=...
          N0000004  n_subjects=4  shards=4  train_set_id=...
          disk: ... bytes, vs ... as independent copies (saved ...); store 5 files, cache ... bytes
          manifest: <out>/family.json
        >>> len(stdout.splitlines())
        5

        The family manifest is on disk, and the members are real MEDS roots whose train splits nest:

        >>> manifest = json.loads((out / "family.json").read_text())
        >>> [m["name"] for m in manifest["members"]]
        ['N0000002', 'N0000004']
        >>> splits = [pl.read_parquet(out / n / "metadata" / "subject_splits.parquet")
        ...           for n in ("N0000002", "N0000004")]
        >>> [sorted(df.filter(pl.col("split") == "train")["subject_id"]) for df in splits]
        [[239684, 814703], [68729, 239684, 814703, 1195293]]

        A directory that is not a MEDS dataset exits with the input-error code and a message, not a
        traceback:

        >>> code, stdout, stderr = run(str(Path(tmp.name) / "nope"), str(out), "-n", "2")
        >>> code, stdout
        (3, '')
        >>> print(stderr, end="")
        error: ...nope is not a MEDS dataset: no data directory at ...nope/data

        So does a family with no sizes, and a size the parent cannot supply:

        >>> run(str(simple_static_MEDS), str(out))[::2]
        (3, 'error: No subset sizes: pass --n-subjects, or a --config that sets n_subjects.\n')
        >>> code, _, stderr = run(str(simple_static_MEDS), str(out), "-n", "9")
        >>> code, stderr
        (3, "error: Cannot build a 9-subject subset: the parent's 'train' split has only 4 subjects.\n")

        A ``--config`` file carries the settings that must not drift between runs and the command line
        carries the sweep, so the file is not required to be a whole config on its own. A flag passed
        its own default value still wins over the file, because it was passed:

        >>> cfg = Path(tmp.name) / "pin.yaml"
        >>> _ = cfg.write_text("salt: from_file\ncode_metadata: copy\nn_subjects_per_shard: 2\n")
        >>> pinned = Path(tmp.name) / "pinned"
        >>> run(str(simple_static_MEDS), str(pinned), "-c", str(cfg), "-n", "2")[0]
        0
        >>> json.loads((pinned / "family.json").read_text())["config"]["salt"]
        'from_file'
        >>> overridden = Path(tmp.name) / "overridden"
        >>> run(str(simple_static_MEDS), str(overridden), "-c", str(cfg), "-n", "2",
        ...     "--salt", "meds_subsetter")[0]
        0
        >>> json.loads((overridden / "family.json").read_text())["config"]["salt"]
        'meds_subsetter'

        The two families select different subjects, which is exactly what a dropped ``--salt`` would
        have silently changed:

        >>> def train_of(root):
        ...     df = pl.read_parquet(root / "N0000002" / "metadata" / "subject_splits.parquet")
        ...     return sorted(df.filter(pl.col("split") == "train")["subject_id"])
        >>> train_of(pinned), train_of(overridden)
        ([68729, 239684], [239684, 814703])

        Bad inputs stay on the ``error:`` path whatever kind of bad they are -- a config that is not
        YAML, a config that is a directory, an output root that is an existing file:

        >>> bad = Path(tmp.name) / "bad.yaml"
        >>> _ = bad.write_text("n_subjects: [2")
        >>> code, _, stderr = run(str(simple_static_MEDS), str(out), "-c", str(bad))
        >>> code, stderr.splitlines()[0]
        (3, 'error: ...bad.yaml is not valid YAML: while parsing a flow sequence')
        >>> code, _, stderr = run(str(simple_static_MEDS), str(out), "-c", str(simple_static_MEDS))
        >>> code, stderr.startswith("error: [Errno 21] Is a directory: ")
        (3, True)
        >>> notadir = Path(tmp.name) / "notadir"
        >>> _ = notadir.write_text("")
        >>> code, _, stderr = run(str(simple_static_MEDS), str(notadir), "-n", "2")
        >>> code, stderr.startswith("error: [Errno 20] Not a directory: ")
        (3, True)

        A family remembers which parent it was built from and refuses a different one. The refusal is
        worded for whoever is reading it: the builders name the command-line flag alongside the
        keyword the library API takes, so nothing has to be rewritten on the way to stderr:

        >>> import shutil
        >>> other = Path(tmp.name) / "other"
        >>> _ = shutil.copytree(simple_static_MEDS, other)
        >>> splits = other / "metadata" / "subject_splits.parquet"
        >>> _ = pl.read_parquet(splits).head(5).write_parquet(splits)
        >>> code, _, stderr = run(str(other), str(pinned), "-c", str(cfg), "-n", "2")
        >>> code, "--force" in stderr
        (3, True)
        >>> print(stderr.split("; ")[-1], end="")
        its shards are not interchangeable. Rebuild with overwrite enabled (--force, or
        do_overwrite=True) to discard and rebuild.
        >>> run(str(other), str(pinned), "-c", str(cfg), "-n", "2", "--force")[0]
        0
        >>> tmp.cleanup()

        With ``--index-dir``, each member also gets its task labels, resharded to match its data:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "task"
        >>> parent = simple_static_MEDS_dataset_with_task
        >>> run(str(parent), str(out), "-n", "2", "--n-subjects-per-shard", "2",
        ...     "--code-metadata", "copy",
        ...     "--index-dir", str(parent / "task_labels" / "boolean_value_task"))[0]
        0
        >>> print_directory(out / "N0000002" / "tasks")
        └── boolean_value_task
            ├── held_out
            │   └── 0.parquet
            ├── train
            │   └── 0.parquet
            └── tuning
                └── 0.parquet
        >>> tmp.cleanup()
    """
    args = _subset_parser().parse_args(argv)
    try:
        cfg = _resolve_subset_config(args)
        kind = args.kind
        if kind == "auto":
            kind = "tensorized" if is_tensorized_cohort(args.parent) else "meds"
        if kind == "tensorized":
            if args.index_dir is not None:
                raise ValueError(
                    "--index-dir applies to a raw MEDS parent only. A tensorized cohort's task labels "
                    "live outside it (meds_torchdata reads them from its own task_labels_dir), so "
                    "subset them against the raw dataset instead."
                )
            family = build_tensorized_family(args.parent, args.out_root, cfg, do_overwrite=args.force)
        else:
            family = build_family(
                args.parent,
                args.out_root,
                cfg,
                index_dir=args.index_dir,
                task_name=args.task_name,
                workers=args.workers,
                worker=args.worker,
                do_overwrite=args.force,
            )
    except _INPUT_ERRORS as e:
        return _report_input_error(e)
    except ImportError as e:
        return _report_input_error(e)

    print(_format_family(family))
    return 0


def _resample_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the resample command.

    Examples:
        >>> args = _resample_parser().parse_args(["/data/mimic", "/out", "--n-resamples", "100",
        ...                                       "--size", "50"])
        >>> args.parent, args.out_root, args.n_resamples, args.size
        (PosixPath('/data/mimic'), PosixPath('/out'), 100, 50)
        >>> args.exclude_from, args.index_dir, args.task_name
        (None, None, None)

        As for :func:`_subset_parser`, the flags a ``--config`` file can also set parse to ``None``
        when absent, so passing one its own default value still overrides the file. The two switches
        are exposed only in their non-default direction, which makes them self-describing:

        >>> args.salt, args.source_split, args.name_template
        (None, None, None)
        >>> args.no_replace, args.allow_overlap
        (False, False)
        >>> _resample_parser().parse_args(["/d", "/o", "--source-split", "train"]).source_split
        'train'
        >>> help_text = " ".join(_resample_parser().format_help().split())
        >>> [d in help_text for d in ("(default: meds_subsetter)", "(default: train)",
        ...                           "(default: boot_{index:03d})")]
        [True, True, True]
    """
    parser = argparse.ArgumentParser(
        prog="meds-resample",
        description="Draw a family of resampled evaluation manifests over an unmodified MEDS dataset.",
    )
    parser.add_argument("parent", type=Path, help="Root of the parent MEDS dataset. Never written to.")
    parser.add_argument("out_root", type=Path, help="Directory to write <out_root>/resamples/ under.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="YAML ResampleConfig. The flags below override the fields they name.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=None,
        metavar="N",
        help="How many replicates to draw. Required unless --config sets n_resamples.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        metavar="N",
        help="Subjects drawn per replicate. Required unless --config sets size.",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Draw distinct subjects, without replacement. The default is the classic bootstrap, "
        "which draws with replacement.",
    )
    parser.add_argument(
        "--salt",
        default=None,
        help=(
            f"Mixed into the sha256 counter stream the draws come from (default: {_RESAMPLE_DEFAULTS.salt})."
        ),
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Let replicates draw subjects that --exclude-from named. Off by default, so an "
        "evaluation resample cannot quietly overlap the training subset.",
    )
    parser.add_argument(
        "--source-split",
        default=None,
        help=f"The parent split the pool is drawn from (default: {_RESAMPLE_DEFAULTS.source_split}).",
    )
    parser.add_argument(
        "--name-template",
        default=None,
        help="str.format template for replicate directory names, rendered with index "
        f"(default: {_RESAMPLE_DEFAULTS.name_template}).",
    )
    parser.add_argument(
        "--exclude-from",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "A MEDS root -- normally the training subset -- or a subject_splits.parquet file, whose "
            "subjects no replicate may draw (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=(
            "Directory of the parent's task/index label shards. Each replicate then gets its own "
            "labels/ copy, rows repeated per draw (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="The task name recorded in the manifests (default: the --index-dir directory's own name).",
    )
    return parser


def _resolve_resample_config(args: argparse.Namespace) -> ResampleConfig:
    """Merge the YAML config, if any, with command-line overrides.

    As for :func:`_resolve_subset_config`, the file is a partial mapping, only the merge is validated,
    and a flag overrides the file when it was *passed* rather than when its value differs from the
    default. The two booleans are exposed only in their non-default direction (``--no-replace``,
    ``--allow-overlap``), which is what makes "passed" and "not passed" distinguishable for them.

    Args:
        args: The parsed command line.

    Returns:
        The resolved config.

    Raises:
        ValueError: If ``n_resamples`` or ``size`` is available from neither source, if the file is
            not valid YAML, or if the merged mapping is not a valid config.
        OSError: If the ``--config`` file cannot be read.

    Examples:
        >>> parser = _resample_parser()
        >>> cfg = _resolve_resample_config(
        ...     parser.parse_args(["/d", "/o", "--n-resamples", "10", "--size", "50"]))
        >>> cfg.n_resamples, cfg.size, cfg.replace, cfg.allow_overlap
        (10, 50, True, False)

        The file supplies what the flags do not:

        >>> spec = {"c.yaml": {"n_resamples": 4, "size": 9, "salt": "from_file", "replace": False}}
        >>> with yaml_disk(spec) as root:
        ...     cfg = _resolve_resample_config(
        ...         parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml")]))
        >>> cfg.n_resamples, cfg.size, cfg.salt, cfg.replace
        (4, 9, 'from_file', False)

        A passed flag overrides it:

        >>> with yaml_disk(spec) as root:
        ...     cfg = _resolve_resample_config(
        ...         parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml"), "--size", "3"]))
        >>> cfg.size, cfg.salt
        (3, 'from_file')

        A flag passed its own default value is still an override, since "passed" is detected by
        absence and not by comparison against the default:

        >>> with yaml_disk(spec) as root:
        ...     cfg = _resolve_resample_config(parser.parse_args(
        ...         ["/d", "/o", "-c", str(root / "c.yaml"), "--salt", "meds_subsetter"]))
        >>> cfg.salt
        'meds_subsetter'

        The file need only be a *part* of a config; the flags fill the rest in:

        >>> with yaml_disk({"c.yaml": {"salt": "from_file", "source_split": "held_out"}}) as root:
        ...     cfg = _resolve_resample_config(parser.parse_args(
        ...         ["/d", "/o", "-c", str(root / "c.yaml"), "--n-resamples", "2", "--size", "2"]))
        >>> cfg.n_resamples, cfg.size, cfg.salt, cfg.source_split
        (2, 2, 'from_file', 'held_out')

        Unknown keys are still refused, and a file that is not YAML is named:

        >>> with yaml_disk({"c.yaml": {"n_resamples": 2, "size": 2, "n_resample": 3}}) as root:
        ...     _resolve_resample_config(parser.parse_args(["/d", "/o", "-c", str(root / "c.yaml")]))
        Traceback (most recent call last):
            ...
        ValueError: Unknown ResampleConfig keys: ['n_resample']
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "c.yaml"
        ...     _ = path.write_text("size: [2")
        ...     _resolve_resample_config(parser.parse_args(["/d", "/o", "-c", str(path)]))
        Traceback (most recent call last):
            ...
        ValueError: ...c.yaml is not valid YAML: while parsing a flow sequence...

        The switches and the remaining fields:

        >>> cfg = _resolve_resample_config(parser.parse_args(
        ...     ["/d", "/o", "--n-resamples", "2", "--size", "3", "--no-replace", "--allow-overlap",
        ...      "--salt", "boot", "--source-split", "held_out", "--name-template", "r{index}"]))
        >>> cfg.replace, cfg.allow_overlap, cfg.salt, cfg.source_split
        (False, True, 'boot', 'held_out')
        >>> cfg.resample_name(1)
        'r1'

        Both counts are required from one source or the other:

        >>> _resolve_resample_config(parser.parse_args(["/d", "/o", "--size", "3"]))
        Traceback (most recent call last):
            ...
        ValueError: Missing resample settings: n_resamples. Pass --n-resamples, or a --config that
        sets it.
        >>> _resolve_resample_config(parser.parse_args(["/d", "/o"]))
        Traceback (most recent call last):
            ...
        ValueError: Missing resample settings: n_resamples, size. Pass --n-resamples/--size, or a
        --config that sets them.
    """
    base = _config_mapping(args.config)
    overrides = _passed_overrides(args, ("n_resamples", "size", "salt", "source_split", "name_template"))
    if args.no_replace:
        overrides["replace"] = False
    if args.allow_overlap:
        overrides["allow_overlap"] = True

    merged = {**base, **overrides}
    missing = [field for field in ("n_resamples", "size") if field not in merged]
    if missing:
        flags = "/".join(f"--{field.replace('_', '-')}" for field in missing)
        raise ValueError(
            f"Missing resample settings: {', '.join(missing)}. Pass {flags}, or a --config that "
            f"sets {'them' if len(missing) > 1 else 'it'}."
        )
    return ResampleConfig.from_dict(merged)


def _excluded_subjects(path: Path) -> list[int]:
    """Read the subject ids a resample family must not draw.

    Args:
        path: A MEDS root -- normally the training subset the evaluation sets must stay disjoint from
            -- or a ``subject_splits.parquet`` file directly.

    Returns:
        Its subject ids, sorted and deduplicated.

    Raises:
        FileNotFoundError: If no splits file is there.

    Examples:
        >>> _excluded_subjects(simple_static_MEDS)
        [68729, 239684, 754281, 814703, 1195293, 1500733]
        >>> _excluded_subjects(simple_static_MEDS / "metadata" / "subject_splits.parquet")
        [68729, 239684, 754281, 814703, 1195293, 1500733]
        >>> _excluded_subjects(Path("/nonexistent"))
        Traceback (most recent call last):
            ...
        FileNotFoundError: No metadata/subject_splits.parquet under /nonexistent
    """
    splits_path = path if path.is_file() else path / meds.subject_splits_filepath
    if not splits_path.is_file():
        raise FileNotFoundError(f"No {meds.subject_splits_filepath} under {path}")
    column = meds.SubjectSplitSchema.subject_id_name
    return sorted(set(pl.read_parquet(splits_path, glob=False)[column].to_list()))


def _format_resamples(family: ResampleFamilyManifest, out_root: Path) -> str:
    """Render a built resample family as a short human-readable block.

    The replicates are summarized rather than listed: a family is routinely hundreds of them, and each
    one's manifest is on disk anyway.

    Args:
        family: The manifest :func:`~meds_subsetter.resample.build_resamples` returned.
        out_root: The output root it was written under.

    Returns:
        The block, without a trailing newline.

    Examples:
        >>> from meds_subsetter.digests import digest
        >>> from meds_subsetter.resample import ResampleFamilyManifest
        >>> family = ResampleFamilyManifest(
        ...     config=ResampleConfig(n_resamples=100, size=50, salt="boot"), parent="/parent",
        ...     parent_subject_set_id=digest(["1"]), pool_subject_set_id=digest(["2"]), pool_size=80,
        ...     n_excluded=20, excluded_training_subset=True, task_name="mortality", resamples=(),
        ... )
        >>> print(_format_resamples(family, Path("/out")))
        Drew 100 resamples from /parent into /out/resamples
          pool: 80 eligible subjects in 'train', 20 excluded
          draws: 50 per replicate, with replacement, salt='boot', task='mortality'
          manifest: /out/resamples/family.json

        Without replacement, and with no exclusion, it says so:

        >>> family = ResampleFamilyManifest(
        ...     config=ResampleConfig(n_resamples=2, size=3, replace=False), parent="/parent",
        ...     parent_subject_set_id=digest(["1"]), pool_subject_set_id=digest(["2"]), pool_size=4,
        ...     n_excluded=0, excluded_training_subset=False, task_name=None, resamples=(),
        ... )
        >>> print(_format_resamples(family, Path("/out")))
        Drew 2 resamples from /parent into /out/resamples
          pool: 4 eligible subjects in 'train', 0 excluded
          draws: 3 per replicate, without replacement, salt='meds_subsetter', task=None
          manifest: /out/resamples/family.json
    """
    cfg = family.config
    family_dir = out_root / RESAMPLES_SUBDIR
    return "\n".join(
        [
            f"Drew {cfg.n_resamples} resamples from {family.parent} into {family_dir}",
            f"  pool: {family.pool_size} eligible subjects in {cfg.source_split!r}, "
            f"{family.n_excluded} excluded",
            f"  draws: {cfg.size} per replicate, "
            f"{'with' if cfg.replace else 'without'} replacement, salt={cfg.salt!r}, "
            f"task={family.task_name!r}",
            f"  manifest: {family_dir / FAMILY_MANIFEST_NAME}",
        ]
    )


def resample_main(argv: list[str] | None = None) -> int:
    r"""Run the resample command.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.

    Examples:
        >>> import contextlib, io
        >>> def run(*args):
        ...     out, err = io.StringIO(), io.StringIO()
        ...     with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        ...         code = resample_main(list(args))
        ...     return code, out.getvalue(), err.getvalue()

        Three bootstrap replicates of the parent's train split, with their labels:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "boot"
        >>> parent = simple_static_MEDS_dataset_with_task
        >>> labels = parent / "task_labels" / "boolean_value_task"
        >>> code, stdout, stderr = run(str(parent), str(out), "--n-resamples", "3", "--size", "4",
        ...                            "--salt", "boot", "--index-dir", str(labels))
        >>> code
        0
        >>> print(stdout.replace(str(parent), "<parent>").replace(str(out), "<out>"))
        Drew 3 resamples from <parent> into <out>/resamples
          pool: 4 eligible subjects in 'train', 0 excluded
          draws: 4 per replicate, with replacement, salt='boot', task='boolean_value_task'
          manifest: <out>/resamples/family.json

        Each replicate is a manifest over the untouched parent -- repeated subject ids and repeated
        label rows, never a MEDS root:

        >>> sorted(p.name for p in (out / "resamples").iterdir())
        ['boot_000', 'boot_001', 'boot_002', 'family.json']
        >>> pl.read_parquet(out / "resamples" / "boot_002" / "subjects.parquet")["subject_id"].to_list()
        [239684, 239684, 239684, 814703]
        >>> print_directory(out / "resamples" / "boot_002")
        ├── labels
        │   ├── labels_A.parquet.parquet
        │   └── labels_B.parquet.parquet
        ├── manifest.json
        └── subjects.parquet

        ``--exclude-from`` takes the training subset out of the pool, which is what keeps an
        evaluation resample disjoint from what was trained on:

        >>> sweep = Path(tmp.name) / "sweep"
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     _ = subset_main([str(parent), str(sweep), "-n", "2", "--code-metadata", "copy"])
        >>> code, stdout, _ = run(str(parent), str(Path(tmp.name) / "disjoint"), "--n-resamples", "1",
        ...                       "--size", "2", "--exclude-from", str(sweep / "N0000002"))
        >>> code
        0
        >>> print(stdout.splitlines()[1])
          pool: 2 eligible subjects in 'train', 2 excluded

        Asking for more distinct subjects than the pool holds is an input error, not a short draw:

        >>> code, _, stderr = run(str(parent), str(out), "--n-resamples", "1", "--size", "9",
        ...                       "--no-replace")
        >>> code
        3
        >>> print(stderr, end="")
        error: Cannot draw 9 distinct subjects without replacement from a pool of 4 ('train' split
        of ..., 0 excluded)

        As is a parent with no splits file to draw from:

        >>> run(str(Path(tmp.name) / "nope"), str(out), "--n-resamples", "1", "--size", "1")[::2]
        (3, 'error: ...nope/metadata/subject_splits.parquet does not exist\n')

        A partial ``--config`` is completed by the flags, and a config that is not YAML -- or is a
        directory -- is an input error rather than a traceback:

        >>> cfg = Path(tmp.name) / "pin.yaml"
        >>> _ = cfg.write_text("salt: from_file\nsource_split: train\n")
        >>> code, stdout, _ = run(str(parent), str(Path(tmp.name) / "partial"), "-c", str(cfg),
        ...                       "--n-resamples", "2", "--size", "2")
        >>> code, stdout.splitlines()[2]
        (0, "  draws: 2 per replicate, with replacement, salt='from_file', task=None")
        >>> bad = Path(tmp.name) / "bad.yaml"
        >>> _ = bad.write_text("size: [2")
        >>> code, _, stderr = run(str(parent), str(out), "-c", str(bad))
        >>> code, stderr.splitlines()[0]
        (3, 'error: ...bad.yaml is not valid YAML: while parsing a flow sequence')
        >>> code, _, stderr = run(str(parent), str(out), "-c", str(parent))
        >>> code, stderr.startswith("error: [Errno 21] Is a directory: ")
        (3, True)
        >>> tmp.cleanup()
    """
    args = _resample_parser().parse_args(argv)
    try:
        cfg = _resolve_resample_config(args)
        exclude = None if args.exclude_from is None else _excluded_subjects(args.exclude_from)
        family = build_resamples(
            args.parent,
            args.out_root,
            cfg,
            exclude_subjects=exclude,
            index_dir=args.index_dir,
            task_name=args.task_name,
        )
    except _INPUT_ERRORS as e:
        return _report_input_error(e)

    print(_format_resamples(family, args.out_root))
    return 0


def _size_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the size command.

    Examples:
        >>> args = _size_parser().parse_args(["/data/a", "/data/b", "-o", "sizes.parquet"])
        >>> args.datasets, args.output, args.kind
        ([PosixPath('/data/a'), PosixPath('/data/b')], PosixPath('sizes.parquet'), 'auto')
        >>> _size_parser().parse_args(["/data/a", "--kind", "tensorized"]).kind
        'tensorized'
        >>> _size_parser().parse_args(["/data/a"]).output, _size_parser().parse_args(["/d"]).json_output
        (None, None)
    """
    parser = argparse.ArgumentParser(
        prog="meds-size",
        description="Report exact size statistics for MEDS datasets and MTD tensorized cohorts.",
    )
    parser.add_argument(
        "datasets", type=Path, nargs="+", help="Dataset roots to size. Each becomes its own rows."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the tidy parquet -- one row per (dataset, split), plus a total row per "
            "dataset. A JSON sidecar is written beside it. Default: a summary on stdout."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        default=None,
        help="Where to write the JSON sidecar (default: --output with a .json suffix).",
    )
    parser.add_argument(
        "--kind",
        choices=("meds", "tensorized", "auto"),
        default="auto",
        help=(
            "Whether each dataset is a raw MEDS root or an MTD tensorized cohort; 'auto' decides per "
            f"dataset by looking for {TENSORIZED_SCHEMAS_SUBDIR}/ (default: %(default)s)."
        ),
    )
    return parser


def _size_dataset(root: Path, kind: str) -> SizeReport:
    """Size one dataset, resolving ``auto`` to the kind the directory actually is.

    Args:
        root: The dataset root.
        kind: ``"meds"``, ``"tensorized"``, or ``"auto"``.

    Returns:
        The report.

    Raises:
        FileNotFoundError: If the dataset holds nothing of the requested kind.

    Examples:
        >>> _size_dataset(simple_static_MEDS, "auto").kind
        'meds'
        >>> _size_dataset(simple_static_MEDS, "meds").total.n_subjects
        6

        ``auto`` is decided by the presence of ``tokenization/schemas``, which only a tensorized
        cohort has:

        >>> spec = {"tokenization/schemas/train/0.parquet": {"subject_id": [1], "static_code": [[7]]},
        ...         "metadata/codes.parquet": {"code": ["A"], "vocab_index": [7]}}
        >>> with yaml_disk(spec) as root:
        ...     header = json.dumps({"dim1/time_delta_days": {"dtype": "I64", "shape": [2]}}).encode()
        ...     (root / "data" / "train").mkdir(parents=True)
        ...     _ = (root / "data" / "train" / "0.nrt").write_bytes(
        ...         len(header).to_bytes(8, "little") + header)
        ...     report = _size_dataset(root, "auto")
        >>> report.kind, report.total.n_subjects, report.total.n_events
        ('tensorized', 1, 2)

        Asking for the wrong kind fails rather than reporting zeros:

        >>> _size_dataset(simple_static_MEDS, "tensorized")
        Traceback (most recent call last):
            ...
        FileNotFoundError: No tensorized schema shards found under ...tokenization/schemas
    """
    if kind == "auto":
        kind = "tensorized" if (root / TENSORIZED_SCHEMAS_SUBDIR).is_dir() else "meds"
    return size_tensorized(root) if kind == "tensorized" else size_meds(root)


def _sidecar_path(output: Path | None, json_output: Path | None) -> Path | None:
    """Resolve where the JSON sidecar goes, refusing a destination that would clobber the parquet.

    The sidecar defaults to ``--output`` with its suffix swapped for ``.json``, which collides with
    ``--output`` itself the moment ``--output`` already ends in ``.json`` -- and the sidecar is
    written second, so the tidy parquet the flag promises would be silently replaced by JSON while the
    command reported success. ``--json`` naming the same file is the same collision, spelled out.

    Args:
        output: The ``--output`` path, or ``None``.
        json_output: The ``--json`` path, or ``None``.

    Returns:
        Where to write the sidecar, or ``None`` when there is none to write.

    Raises:
        ValueError: If the sidecar and ``--output`` would be the same file.

    Examples:
        >>> _sidecar_path(Path("/out/sizes.parquet"), None)
        PosixPath('/out/sizes.json')
        >>> _sidecar_path(None, Path("/out/only.json"))
        PosixPath('/out/only.json')
        >>> _sidecar_path(None, None) is None
        True

        A ``.json`` ``--output`` has no distinct sidecar name left, so it is refused rather than
        overwritten:

        >>> _sidecar_path(Path("/out/sizes.json"), None)
        Traceback (most recent call last):
            ...
        ValueError: --output and the JSON sidecar would both be written to /out/sizes.json; pass a
        distinct --json path.

        So is the same collision written out in full, and one spelled differently on the way to the
        same file:

        >>> _sidecar_path(Path("/out/both.parquet"), Path("/out/both.parquet"))
        Traceback (most recent call last):
            ...
        ValueError: --output and the JSON sidecar would both be written to /out/both.parquet; pass a
        distinct --json path.
        >>> _sidecar_path(Path("/out/sizes.parquet"), Path("/out/../out/sizes.parquet"))
        Traceback (most recent call last):
            ...
        ValueError: ...pass a distinct --json path.
    """
    if json_output is not None:
        json_path = json_output
    elif output is not None:
        json_path = output.with_suffix(".json")
    else:
        return None
    if output is not None and json_path.resolve() == output.resolve():
        raise ValueError(
            f"--output and the JSON sidecar would both be written to {json_path}; pass a distinct "
            f"--json path."
        )
    return json_path


def _format_report(report: SizeReport) -> str:
    """Render one size report as a small fixed-width table.

    Args:
        report: The report to render.

    Returns:
        The table, without a trailing newline.

    Examples:
        >>> from meds_subsetter.digests import digest
        >>> from meds_subsetter.sizes import SizeReport, SplitSizes
        >>> train = SplitSizes(
        ...     split="train", n_subjects=4, n_events=20, n_measurements=44,
        ...     n_dynamic_measurements=36, n_static_measurements=8, n_subjects_with_static=4,
        ...     n_codes_observed=11, observed_code_set_id=digest(["A"]), n_shards=2, n_bytes=3238,
        ... )
        >>> report = SizeReport(
        ...     source="/data/cohort", kind="meds", splits=(train,), n_codes_observed=11,
        ...     observed_code_set_id=digest(["A"]), n_shards=2, n_bytes=3238,
        ... )
        >>> print(_format_report(report))
        /data/cohort (meds)
          split      n_subjects  n_events  n_measurements  n_shards  n_bytes
          train               4        20              44         2     3238
          __total__           4        20              44         2     3238
          n_codes_observed=11  observed_code_set_id=3addd7d5a0987fa7
    """
    label, *counts = _SUMMARY_COLUMNS
    rows = [split.to_dict() for split in (*report.splits, report.total)]
    widths = {c: max(len(c), *(len(str(row[c])) for row in rows)) for c in _SUMMARY_COLUMNS}

    def render(values: dict[str, Any]) -> str:
        cells = [f"{values[label]:<{widths[label]}}"]
        cells += [f"{values[c]:>{widths[c]}}" for c in counts]
        return "  " + "  ".join(cells)

    return "\n".join(
        [
            f"{report.source} ({report.kind})",
            render({c: c for c in _SUMMARY_COLUMNS}),
            *(render(row) for row in rows),
            f"  n_codes_observed={report.n_codes_observed}  "
            f"observed_code_set_id={short_id(report.observed_code_set_id)}",
        ]
    )


def size_main(argv: list[str] | None = None) -> int:
    r"""Run the size command.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.

    Examples:
        >>> import contextlib, io
        >>> def run(*args):
        ...     out, err = io.StringIO(), io.StringIO()
        ...     with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        ...         code = size_main(list(args))
        ...     return code, out.getvalue(), err.getvalue()

        With no ``-o``, the report is a short table on stdout:

        >>> code, stdout, stderr = run(str(simple_static_MEDS))
        >>> code
        0
        >>> print(stdout.replace(str(simple_static_MEDS), "<root>"))
        <root> (meds)
          split      n_subjects  n_events  n_measurements  n_shards  n_bytes
          held_out            1         5              11         1  ...
          train               4        20              44         2  ...
          tuning              1         3               7         1  ...
          __total__           6        28              62         4  ...
          n_codes_observed=11  observed_code_set_id=4e3cf86133348078
        >>> len(stdout.splitlines())
        7

        With ``-o``, the tidy parquet is the deliverable -- one row per (dataset, split), plus a total
        row per dataset -- and a JSON sidecar lands beside it:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "sizes.parquet"
        >>> code, stdout, stderr = run(str(simple_static_MEDS), "-o", str(out))
        >>> code, stdout
        (0, '')
        >>> print(stderr.replace(str(out.parent), "<dir>"), end="")
        Wrote sizes for 1 dataset to <dir>/sizes.parquet and <dir>/sizes.json
        >>> pl.read_parquet(out).select("kind", "split", "n_subjects", "n_measurements")
        shape: (4, 4)
        ┌──────┬───────────┬────────────┬────────────────┐
        │ kind ┆ split     ┆ n_subjects ┆ n_measurements │
        │ ---  ┆ ---       ┆ ---        ┆ ---            │
        │ str  ┆ str       ┆ i64        ┆ i64            │
        ╞══════╪═══════════╪════════════╪════════════════╡
        │ meds ┆ held_out  ┆ 1          ┆ 11             │
        │ meds ┆ train     ┆ 4          ┆ 44             │
        │ meds ┆ tuning    ┆ 1          ┆ 7              │
        │ meds ┆ __total__ ┆ 6          ┆ 62             │
        └──────┴───────────┴────────────┴────────────────┘
        >>> sidecar = json.loads((out.parent / "sizes.json").read_text())
        >>> sorted(sidecar), len(sidecar["datasets"])
        (['datasets', 'tool', 'version'], 1)
        >>> sidecar["datasets"][0]["total"]["n_events"]
        28

        ``--json`` on its own writes only the sidecar:

        >>> only = Path(tmp.name) / "only.json"
        >>> code, stdout, stderr = run(str(simple_static_MEDS), "--json", str(only))
        >>> code, stdout, sorted(json.loads(only.read_text()))
        (0, '', ['datasets', 'tool', 'version'])

        Several datasets go into one frame, which is what a scaling sweep wants; the ``source`` column
        is what separates them:

        >>> sweep = Path(tmp.name) / "sweep"
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     _ = subset_main([str(simple_static_MEDS), str(sweep), "-n", "2", "4",
        ...                      "--n-subjects-per-shard", "2", "--code-metadata", "copy"])
        >>> code, _, _ = run(str(sweep / "N0000002"), str(sweep / "N0000004"), "-o", str(out))
        >>> code
        0
        >>> pl.read_parquet(out).filter(pl.col("split") == "__total__").select(
        ...     pl.col("source").str.split("/").list.last(), "n_subjects", "n_measurements")
        shape: (2, 3)
        ┌──────────┬────────────┬────────────────┐
        │ source   ┆ n_subjects ┆ n_measurements │
        │ ---      ┆ ---        ┆ ---            │
        │ str      ┆ i64        ┆ i64            │
        ╞══════════╪════════════╪════════════════╡
        │ N0000002 ┆ 4          ┆ 38             │
        │ N0000004 ┆ 6          ┆ 62             │
        └──────────┴────────────┴────────────────┘

        A directory with no shards is an input error:

        >>> run(str(Path(tmp.name) / "nope"))[::2]
        (3, 'error: No MEDS shards found under ...nope/data\n')

        So is a pair of destinations that would land on the same file: the sidecar is derived from
        ``--output`` by suffix, so a ``.json`` ``--output`` would otherwise have its parquet written
        and then silently overwritten by the JSON, and the command would still claim success:

        >>> clash = Path(tmp.name) / "clash.json"
        >>> code, _, stderr = run(str(simple_static_MEDS), "-o", str(clash))
        >>> code, clash.exists()
        (3, False)
        >>> print(stderr.replace(str(clash), "<path>"), end="")
        error: --output and the JSON sidecar would both be written to <path>; pass a distinct
        --json path.
        >>> both = Path(tmp.name) / "both.parquet"
        >>> run(str(simple_static_MEDS), "-o", str(both), "--json", str(both))[0]
        3
        >>> both.exists()
        False

        A destination that cannot be written to is reported the same way, rather than as a traceback
        out of the writer:

        >>> notadir = Path(tmp.name) / "notadir"
        >>> _ = notadir.write_text("")
        >>> code, _, stderr = run(str(simple_static_MEDS), "-o", str(notadir / "sizes.parquet"))
        >>> code, stderr.startswith("error: [Errno 17] File exists: ")
        (3, True)
        >>> tmp.cleanup()
    """
    args = _size_parser().parse_args(argv)
    written: list[Path] = []
    try:
        json_path = _sidecar_path(args.output, args.json_output)
        reports = [_size_dataset(dataset, args.kind) for dataset in args.datasets]
        if args.output is not None:
            atomic_write_parquet(pl.concat([report.to_frame() for report in reports]), args.output)
            written.append(args.output)
        if json_path is not None:
            sidecar = {
                "tool": __package_name__,
                "version": __version__,
                "datasets": [report.to_dict() for report in reports],
            }
            atomic_write_json(sidecar, json_path)
            written.append(json_path)
    except _INPUT_ERRORS as e:
        return _report_input_error(e)

    if not written:
        print("\n".join(_format_report(report) for report in reports))
        return 0

    n_datasets = len(reports)
    print(
        f"Wrote sizes for {n_datasets} dataset{'' if n_datasets == 1 else 's'} to "
        f"{' and '.join(str(path) for path in written)}",
        file=sys.stderr,
    )
    return 0


def _fingerprint_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the fingerprint command.

    Examples:
        >>> args = _fingerprint_parser().parse_args(["/data/mimic"])
        >>> args.dataset, args.index_dir, args.train_split, args.output
        (PosixPath('/data/mimic'), None, 'train', None)
        >>> _fingerprint_parser().parse_args(["--index-dir", "/labels"]).dataset is None
        True
    """
    parser = argparse.ArgumentParser(
        prog="meds-fingerprint",
        description="Report the content ids of a MEDS dataset and/or a task/index label frame.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=None,
        help="Root of the MEDS dataset to fingerprint (default: %(default)s).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory of task/index label parquet files to fingerprint (default: %(default)s).",
    )
    parser.add_argument(
        "--train-split",
        default=_SUBSET_DEFAULTS.train_split,
        help="Which split's content id is also reported as train_set_id (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON. Default: stdout.",
    )
    return parser


def _index_fingerprint(index_dir: Path) -> dict[str, Any]:
    """Compute the content id of a task/index label frame.

    The id is over the label *rows*, rendered at the union of the shards' column sets, so it does not
    move when the labels are resharded -- which is exactly what
    :func:`~meds_subsetter.subset.build_family` does to them, and why a member's ``tasks/`` directory
    fingerprints to the ``index_id`` its manifest records.

    ``meds.LabelSchema.align`` is deliberately not called: real ACES output leaves the value columns
    entirely null while the closed schema declares them non-nullable, so aligning raises on genuine
    data. The columns are taken as they are found, exactly as the builder takes them.

    Args:
        index_dir: Directory of label parquet files, read recursively, dotfiles skipped.

    Returns:
        The id and the row count, JSON-ready.

    Raises:
        FileNotFoundError: If ``index_dir`` holds no parquet files.

    Examples:
        >>> labels = simple_static_MEDS_dataset_with_task / "task_labels" / "boolean_value_task"
        >>> ids = _index_fingerprint(labels)
        >>> sorted(ids)
        ['index_id', 'n_index_rows', 'path']
        >>> ids["n_index_rows"], short_id(ids["index_id"])
        (21, 'c79f91f0bcb16f46')
        >>> with tempfile.TemporaryDirectory() as empty:
        ...     _index_fingerprint(Path(empty))
        Traceback (most recent call last):
            ...
        FileNotFoundError: No parquet label files under /tmp/...
    """
    paths = _parquet_shard_paths(index_dir)
    if not paths:
        raise FileNotFoundError(f"No parquet label files under {index_dir}")
    schema = pinned_schema(paths)
    labels = pl.scan_parquet(
        paths, glob=False, schema=schema, missing_columns="insert", extra_columns="ignore"
    ).collect()
    return {
        "path": str(index_dir),
        "index_id": frame_digest(labels.lazy(), schema=schema),
        "n_index_rows": labels.height,
    }


def fingerprint_main(argv: list[str] | None = None) -> int:
    r"""Run the fingerprint command.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.

    Examples:
        >>> import contextlib, io
        >>> def run(*args):
        ...     out, err = io.StringIO(), io.StringIO()
        ...     with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        ...         code = fingerprint_main(list(args))
        ...     return code, out.getvalue(), err.getvalue()

        With no ``-o`` the ids go to stdout as JSON:

        >>> code, stdout, stderr = run(str(simple_static_MEDS))
        >>> code
        0
        >>> ids = json.loads(stdout)
        >>> sorted(ids)
        ['dataset', 'tool', 'version']
        >>> sorted(ids["dataset"]["content_ids"])
        ['held_out', 'train', 'tuning']

        **The point of the command**: these are the ids a subset family records, so a member's
        manifest can be checked against the bytes actually on disk:

        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = Path(tmp.name) / "sweep"
        >>> parent = simple_static_MEDS_dataset_with_task
        >>> labels = parent / "task_labels" / "boolean_value_task"
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     _ = subset_main([str(parent), str(out), "-n", "2", "--n-subjects-per-shard", "2",
        ...                      "--code-metadata", "copy", "--index-dir", str(labels)])
        >>> member = json.loads((out / "family.json").read_text())["members"][0]
        >>> code, stdout, _ = run(str(out / "N0000002"),
        ...                       "--index-dir", str(out / "N0000002" / "tasks" / "boolean_value_task"))
        >>> code
        0
        >>> ids = json.loads(stdout)
        >>> [ids["dataset"][k] == member[k] for k in
        ...  ("subject_set_id", "train_set_id", "content_ids", "codes_id", "dataset_id")]
        [True, True, True, True, True]
        >>> ids["index"]["index_id"] == member["index_id"], ids["index"]["n_index_rows"]
        (True, 14)

        An index frame can be fingerprinted on its own, with no dataset at all:

        >>> code, stdout, _ = run("--index-dir", str(labels))
        >>> code, sorted(json.loads(stdout))
        (0, ['index', 'tool', 'version'])

        ``-o`` writes the same JSON to a file instead:

        >>> path = Path(tmp.name) / "ids.json"
        >>> code, stdout, stderr = run(str(simple_static_MEDS), "-o", str(path))
        >>> code, stdout
        (0, '')
        >>> print(stderr.replace(str(path), "<path>"), end="")
        Wrote content ids to <path>
        >>> sorted(json.loads(path.read_text()))
        ['dataset', 'tool', 'version']

        With neither a dataset nor an index there is nothing to fingerprint:

        >>> run()[::2]
        (3, 'error: Nothing to fingerprint: pass a dataset, --index-dir, or both.\n')

        A directory that is not a MEDS dataset is an input error, not a traceback -- and so is an
        ``-o`` that cannot be written to:

        >>> run(str(Path(tmp.name) / "nope"))[::2]
        (3, 'error: ...nope is not a MEDS dataset: no data directory at ...nope/data\n')
        >>> notadir = Path(tmp.name) / "notadir"
        >>> _ = notadir.write_text("")
        >>> code, _, stderr = run(str(simple_static_MEDS), "-o", str(notadir / "ids.json"))
        >>> code, stderr.startswith("error: [Errno 17] File exists: ")
        (3, True)
        >>> tmp.cleanup()
    """
    args = _fingerprint_parser().parse_args(argv)
    report: dict[str, Any] = {"tool": __package_name__, "version": __version__}
    try:
        if args.dataset is None and args.index_dir is None:
            raise ValueError("Nothing to fingerprint: pass a dataset, --index-dir, or both.")
        if args.dataset is not None:
            report["dataset"] = fingerprint(args.dataset, args.train_split)
        if args.index_dir is not None:
            report["index"] = _index_fingerprint(args.index_dir)
        if args.output is not None:
            atomic_write_json(report, args.output)
    except _INPUT_ERRORS as e:
        return _report_input_error(e)

    if args.output is None:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"Wrote content ids to {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    r"""Dispatch ``python -m meds_subsetter <subset|resample|size|fingerprint> ...``.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        A process exit code.

    Examples:
        >>> import contextlib, io
        >>> def run(*args):
        ...     out, err = io.StringIO(), io.StringIO()
        ...     with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        ...         code = main(list(args))
        ...     return code, out.getvalue(), err.getvalue()

        No subcommand, or one this dispatcher does not have, is a malformed command line: exit code
        ``2`` and the usage on **stderr**, exactly where argparse puts every other bad command line in
        this CLI. A run whose stdout is a results file must not get usage text in the results file:

        >>> run()
        (2, '', 'usage: python -m meds_subsetter {subset,resample,size,fingerprint} ...\n')
        >>> code, stdout, stderr = run("frobnicate")
        >>> code, stdout
        (2, '')
        >>> print(stderr, end="")
        usage: python -m meds_subsetter {subset,resample,size,fingerprint} ...
        python -m meds_subsetter: error: unknown subcommand: frobnicate

        Each name reaches the matching command:

        >>> code, stdout, _ = run("size", str(simple_static_MEDS))
        >>> code
        0
        >>> stdout.splitlines()[0].endswith("(meds)")
        True
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _COMMANDS:
        prog = f"python -m {__package_name__}"
        print(f"usage: {prog} {{{','.join(_COMMANDS)}}} ...", file=sys.stderr)
        if argv:
            print(f"{prog}: error: unknown subcommand: {argv[0]}", file=sys.stderr)
        return 2
    command, *rest = argv
    runners = {
        "subset": subset_main,
        "resample": resample_main,
        "size": size_main,
        "fingerprint": fingerprint_main,
    }
    return runners[command](rest)
