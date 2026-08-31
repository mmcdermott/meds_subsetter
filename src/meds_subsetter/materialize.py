r"""On-disk placement for a subset family: atomic writes, link modes, and the shard store.

A family of nested subsets shares nearly all of its bytes. By design decision D2 every *full* shard is
byte-identical across every member of the family, so a shard is written exactly once into a store
keyed by shard identity (``<out_root>/.shard_store``) and then *placed* into each subset root as a
symlink, a hardlink, or a copy.

Sharing bytes is only safe if nothing ever writes *through* a placed link, and two rules -- both
implemented here -- are what keep it safe:

1. **Every write stages.** Content is written to ``<path>.tmp`` and moved into place with
   :func:`os.replace`. Writing in place would mutate the *inode*, and under ``hardlink`` mode that is
   the very inode every sibling subset points at; staging swaps the *directory entry* instead, so the
   old inode keeps its bytes and any other link to it still sees the data it was placed with. This is
   the MEDS-Transforms ``write_df`` idiom, and it is what makes hardlink mode survivable.
2. **Every placement stages too.** :func:`place` builds the new entry at ``<dst>.tmp`` and
   :func:`os.replace`\ s it onto ``dst``, which buys both halves of the same rule at once: a copy
   writes a fresh file rather than following an existing symlink down into the store and silently
   rewriting the shard of every sibling subset, and a concurrent reader of ``dst`` sees either the old
   placement or the new one -- never a half-copied parquet.

The two sizing helpers at the end of the module, :func:`dir_size` and :func:`unique_inode_size`, are
how a caller demonstrates the resulting disk saving honestly; they disagree exactly where hardlinks
are involved, which is the point.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import LinkMode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import polars as pl

logger = logging.getLogger(__name__)

#: Suffix of the staging file every write in this module goes through before :func:`os.replace`.
TMP_SUFFIX = ".tmp"

#: The four magic bytes that both open and close a parquet file.
PARQUET_MAGIC = b"PAR1"


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` as parquet, via a staging file and :func:`os.replace`.

    Parent directories are created as needed. The staged file is ``<path>.tmp``; on success it is
    renamed over ``path`` in a single filesystem operation, so a reader never observes a partial shard
    and -- crucially for ``hardlink`` placement -- the previous contents of ``path`` are never mutated
    in place. See the module docstring for why that matters.

    Args:
        df: The frame to write.
        path: Final destination. Its parent directories are created if absent.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> path = root / "data" / "train" / "0.parquet"
        >>> atomic_write_parquet(pl.DataFrame({"subject_id": [1, 2]}), path)
        >>> pl.read_parquet(path)["subject_id"].to_list()
        [1, 2]

        No staging file survives a successful write:

        >>> sorted(p.name for p in path.parent.iterdir())
        ['0.parquet']

        Rewriting ``path`` replaces the directory entry rather than the bytes of the file, so the same
        shard hardlinked into a sibling subset keeps what it was placed with:

        >>> sibling = root / "other" / "0.parquet"
        >>> place(path, sibling, "hardlink")
        >>> atomic_write_parquet(pl.DataFrame({"subject_id": [3]}), path)
        >>> pl.read_parquet(path)["subject_id"].to_list()
        [3]
        >>> pl.read_parquet(sibling)["subject_id"].to_list()
        [1, 2]
        >>> tmp.cleanup()
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    df.write_parquet(tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    r"""Write ``text`` to ``path`` as UTF-8, via a staging file and :func:`os.replace`.

    ``text`` is written verbatim: no trailing newline is added. Use :func:`atomic_write_json` for
    JSON, which pins a canonical rendering.

    Args:
        text: The exact contents to write.
        path: Final destination. Its parent directories are created if absent.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> path = root / "nested" / "note.txt"
        >>> atomic_write_text("hello", path)
        >>> path.read_text()
        'hello'
        >>> sorted(p.name for p in path.parent.iterdir())
        ['note.txt']

        Rewriting is a replace, not an in-place edit, so the file is never observed half-written:

        >>> atomic_write_text("goodbye\n", path)
        >>> path.read_text()
        'goodbye\n'
        >>> tmp.cleanup()
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(obj: Any, path: Path) -> None:
    r"""Write ``obj`` to ``path`` as canonical JSON, via a staging file and :func:`os.replace`.

    The rendering is pinned: ``indent=2``, ``sort_keys=True``, and a trailing newline. Sorted keys
    make the bytes a function of the *content* rather than of dict insertion order, so two runs that
    computed the same manifest produce the same file (and therefore the same digest); the indentation
    and trailing newline make the files diff and ``cat`` cleanly.

    Args:
        obj: Any JSON-serializable object.
        path: Final destination. Its parent directories are created if absent.

    Raises:
        TypeError: If ``obj`` is not JSON-serializable. Serialization happens before anything touches
            the filesystem, so a failure leaves no file and no staging file behind.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> path = root / "family.json"
        >>> atomic_write_json({"salt": "abc", "members": [10, 20]}, path)
        >>> print(path.read_text(), end="")
        {
          "members": [
            10,
            20
          ],
          "salt": "abc"
        }
        >>> path.read_text().endswith("\n")
        True

        The indent is two spaces per level. (Checked as leading-space counts rather than as the block
        above, because the suite runs with ``NORMALIZE_WHITESPACE``, under which the pretty-printed
        block matches *any* indent.)

        >>> [len(ln) - len(ln.lstrip()) for ln in path.read_text().splitlines()]
        [0, 2, 4, 4, 2, 2, 0]

        Keys are sorted, so insertion order cannot change the bytes:

        >>> other = root / "family2.json"
        >>> atomic_write_json({"members": [10, 20], "salt": "abc"}, other)
        >>> path.read_bytes() == other.read_bytes()
        True

        A non-serializable object fails before any file is created:

        >>> atomic_write_json({"members": {10, 20}}, root / "bad.json")
        Traceback (most recent call last):
            ...
        TypeError: Object of type set is not JSON serializable
        >>> sorted(p.name for p in root.iterdir())
        ['family.json', 'family2.json']
        >>> tmp.cleanup()
    """
    atomic_write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", path)


def place(src: Path, dst: Path, mode: LinkMode | str) -> None:
    """Place the file ``src`` at ``dst`` under the given link ``mode``.

    The new entry is built at ``<dst>.tmp`` and moved onto ``dst`` with :func:`os.replace`, so an
    existing placement is replaced rather than written through. That staging is what stops a ``copy``
    over an existing symlink from following the link into the shard store and rewriting a shard that
    every sibling subset shares, and it is why ``dst`` is never observed half-written. Two processes
    placing the *same* ``dst`` share that staging name, so -- exactly as for
    :meth:`ShardStore.put` -- concurrent writers of one path must be serialized by the caller.

    The three modes trade off differently, and none dominates:

    - ``symlink`` (the family default): cheapest, and ``ls -l`` shows provenance. An accidental
      in-place overwrite by a downstream tool visibly lands in the store rather than silently
      diverging two subsets. The link is **relative**, so the whole output tree stays movable.
      ``cp`` without ``-a`` dereferences it, quietly turning a shared tree into a full copy.
    - ``hardlink``: no dangling-link risk and no ``cp`` dereference surprise, but the sharing is
      invisible in the tree and both failure modes are silent -- a downstream tool that rewrites a
      shard in place changes it for *every* sibling, and one that stages (as everything here does)
      detaches this member from the family. Hardlinks also cannot cross a filesystem boundary.
    - ``copy``: fully independent bytes, full cost.

    Args:
        src: Existing file to place. Must exist.
        dst: Where to place it. Parent directories are created; an existing ``dst`` is replaced.
        mode: A :class:`~meds_subsetter.config.LinkMode`, or its string value.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        ValueError: If ``mode`` is not a valid link mode, or if ``src`` and ``dst`` name the same
            file (which would delete the source). Naming the same file means either the same
            directory entry -- which a ``dst`` that is itself a symlink does not excuse -- or, for a
            non-symlink ``dst``, the same file after resolution.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> tree = root / "family"
        >>> src = tree / ".shard_store" / "train" / "r0-50.parquet"
        >>> src.parent.mkdir(parents=True)
        >>> _ = src.write_text("shard A")
        >>> dst = tree / "n50" / "data" / "train" / "0.parquet"

        ``symlink`` writes a relative link:

        >>> place(src, dst, "symlink")
        >>> dst.is_symlink(), os.readlink(dst)
        (True, '../../../.shard_store/train/r0-50.parquet')
        >>> dst.read_text()
        'shard A'

        Relative means movable -- renaming the family root does not break the placement:

        >>> os.rename(tree, root / "moved")
        >>> (root / "moved" / "n50" / "data" / "train" / "0.parquet").read_text()
        'shard A'
        >>> os.rename(root / "moved", tree)

        ``hardlink`` shares the inode; ``copy`` does not:

        >>> hard, copied = tree / "n50" / "hard.parquet", tree / "n50" / "copied.parquet"
        >>> place(src, hard, "hardlink")
        >>> hard.stat().st_ino == src.stat().st_ino
        True
        >>> place(src, copied, LinkMode.COPY)
        >>> copied.read_text(), copied.is_symlink(), copied.stat().st_ino == src.stat().st_ino
        ('shard A', False, False)

        Overwriting an existing placement never touches the previous target's bytes. This is the
        hazard the staging exists for: written straight to ``dst``, the copy below would follow the
        symlink and overwrite ``shard A`` inside the store.

        >>> other = tree / ".shard_store" / "train" / "r50-100.parquet"
        >>> _ = other.write_text("shard B")
        >>> place(other, dst, "copy")
        >>> dst.read_text(), dst.is_symlink()
        ('shard B', False)
        >>> src.read_text()
        'shard A'
        >>> place(other, hard, "hardlink")
        >>> hard.read_text(), src.read_text()
        ('shard B', 'shard A')

        Because the new entry is staged, ``dst`` is only ever replaced by a *complete* file: a copy
        that fails partway leaves the previous placement -- and no staging file -- behind.

        >>> from unittest.mock import patch
        >>> with patch("shutil.copy2", side_effect=OSError("No space left on device")):
        ...     place(src, dst, "copy")
        Traceback (most recent call last):
            ...
        OSError: No space left on device
        >>> dst.read_text()
        'shard B'
        >>> sorted(p.name for p in dst.parent.iterdir())
        ['0.parquet']

        A placement is never a source. ``place(dst, dst, ...)`` would delete the very file it was
        asked to place, so it raises -- whether ``dst`` is a regular file or, as every ``symlink``
        placement is, a link:

        >>> place(src, dst, "symlink")
        >>> place(dst, dst, "symlink")
        Traceback (most recent call last):
            ...
        ValueError: Cannot place a file onto itself: /...0.parquet
        >>> place(dst, dst, "copy")
        Traceback (most recent call last):
            ...
        ValueError: Cannot place a file onto itself: /...0.parquet
        >>> dst.read_text(), dst.is_symlink()
        ('shard A', True)

        Re-placing the same source onto an existing link to it is *not* self-placement, though --
        rebuilding a family has to stay idempotent:

        >>> place(src, dst, "symlink")
        >>> dst.read_text(), os.readlink(dst)
        ('shard A', '../../../.shard_store/train/r0-50.parquet')

        A filesystem that refuses hardlinks (or a cross-device ``dst``) degrades to a copy rather than
        failing the run:

        >>> with patch("os.link", side_effect=OSError("Invalid cross-device link")):
        ...     place(src, hard, "hardlink")
        >>> hard.read_text(), hard.stat().st_ino == src.stat().st_ino
        ('shard A', False)

        Errors name the offender:

        >>> place(tree / "nope.parquet", dst, "copy")
        Traceback (most recent call last):
            ...
        FileNotFoundError: Cannot place a source that does not exist: /...nope.parquet
        >>> place(src, src, "copy")
        Traceback (most recent call last):
            ...
        ValueError: Cannot place a file onto itself: /...r0-50.parquet
        >>> place(src, dst, "hard link")
        Traceback (most recent call last):
            ...
        ValueError: 'hard link' is not a valid LinkMode
        >>> tmp.cleanup()
    """
    link_mode = LinkMode(mode)
    if not src.exists():
        raise FileNotFoundError(f"Cannot place a source that does not exist: {src}")
    # The entry comparison resolves the parents but not the final component, so a `dst` that is itself
    # a symlink -- which every placement made in the default mode is -- cannot slip past the check;
    # the resolved comparison then catches the rest, but only when `dst` is not a link, since a link
    # pointing at `src` is precisely the idempotent re-placement case.
    same_entry = dst.parent.resolve() / dst.name == src.parent.resolve() / src.name
    if same_entry or (not dst.is_symlink() and dst.resolve() == src.resolve()):
        raise ValueError(f"Cannot place a file onto itself: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + TMP_SUFFIX)
    tmp.unlink(missing_ok=True)
    try:
        match link_mode:
            case LinkMode.SYMLINK:
                tmp.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
            case LinkMode.HARDLINK:
                try:
                    os.link(src, tmp)
                except OSError as e:
                    logger.warning("Hardlinking %s to %s failed (%s); copying instead.", src, dst, e)
                    shutil.copy2(src, tmp)
            case LinkMode.COPY:
                shutil.copy2(src, tmp)
            case _:  # pragma: no cover - unreachable while LinkMode has exactly these three members
                raise ValueError(f"Unsupported link mode: {link_mode!r}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)


def _check_component(name: str, value: str) -> None:
    """Reject a path component that could escape the store.

    Keys and kinds are derived from user-supplied config and from dataset contents, so the store must
    not be reachable from outside itself. ``..`` is rejected outright rather than resolved, which is
    both simpler to reason about and stricter: no legitimate shard key contains it.

    Args:
        name: The parameter name, for the error message.
        value: The component to check. May be empty (it is simply skipped by the caller) and may
            contain ``/``, since MEDS shard names nest.

    Raises:
        ValueError: If ``value`` contains ``..`` or is absolute.

    Examples:
        >>> _check_component("key", "held_out/deep/nest/1")
        >>> _check_component("key", "a/../../etc/passwd")
        Traceback (most recent call last):
            ...
        ValueError: key must not contain '..'; got 'a/../../etc/passwd'
        >>> _check_component("kind", "/etc")
        Traceback (most recent call last):
            ...
        ValueError: kind must be a relative path; got '/etc'
    """
    if ".." in value:
        raise ValueError(f"{name} must not contain '..'; got {value!r}")
    if os.path.isabs(value):
        raise ValueError(f"{name} must be a relative path; got {value!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class ShardStore:
    """A parquet store keyed by shard identity, shared by every member of a subset family.

    The store lives at ``<out_root>/.shard_store`` and is addressed by *what a shard is* -- its kind,
    an optional subdirectory (typically the salt prefix or the split), and a key naming the rank range
    it covers -- never by which subset asked for it. Two members of a family that need the same full
    shard therefore resolve to the same stored file, and the second one to ask does no work at all.

    The address is that name, not a digest of the bytes: :meth:`put` never inspects content, so a key
    is the caller's promise that everything asking for it wants the same shard. Under design decision
    D2 the promise holds by construction -- a key names a rank range under one salt, and one store
    belongs to one family, whose manifest binds the parent's ``subject_set_id``. Point two different
    parents at a single ``out_root`` without checking that binding and the second one is silently
    handed the first one's shards.

    Attributes:
        root: The ``.shard_store`` directory. Created lazily, on first :meth:`put`.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> store = ShardStore(root / ".shard_store")

        A shard is written through a callback, so the caller keeps control of *how* it is produced
        (streaming ``sink_parquet``, say) while the store keeps control of *where* and of atomicity:

        >>> calls = []
        >>> def write(p: Path) -> None:
        ...     calls.append(p.name)
        ...     pl.DataFrame({"subject_id": [1, 2]}).write_parquet(p)
        >>> stored = store.put("train", "r0000000000-0000000050", write, subdir="a1b2c3d4")
        >>> stored.relative_to(root).as_posix()
        '.shard_store/train/a1b2c3d4/r0000000000-0000000050.parquet'
        >>> calls
        ['r0000000000-0000000050.parquet.tmp']

        Asking again is free -- this is what makes rebuilding a family idempotent and cheap:

        >>> store.put("train", "r0000000000-0000000050", write, subdir="a1b2c3d4") == stored
        True
        >>> calls
        ['r0000000000-0000000050.parquet.tmp']

        Stored shards are then linked into each subset root:

        >>> store.link_into(stored, root / "n50" / "data" / "train" / "0.parquet", "symlink")
        >>> store.link_into(stored, root / "n100" / "data" / "train" / "0.parquet", "symlink")
        >>> pl.read_parquet(root / "n100" / "data" / "train" / "0.parquet")["subject_id"].to_list()
        [1, 2]
        >>> store.stat()["n_files"]
        1
        >>> tmp.cleanup()
    """

    root: Path

    def __post_init__(self) -> None:
        """Coerce ``root`` to a :class:`~pathlib.Path`, so a string root cannot silently misbehave.

        Examples:
            >>> ShardStore("/tmp/x/.shard_store").root
            PosixPath('/tmp/x/.shard_store')
        """
        object.__setattr__(self, "root", Path(self.root))

    def path(self, kind: str, key: str, *, subdir: str = "") -> Path:
        """Resolve ``<root>/<kind>/<subdir>/<key>.parquet``, skipping empty components.

        ``key`` may contain ``/`` -- a passed-through MEDS shard name nests arbitrarily
        (``deep/nest/1``) -- which is exactly why the traversal check below is not optional: keys and
        kinds are derived from user-supplied config and dataset contents.

        Args:
            kind: Top-level bucket, e.g. ``"train"`` or ``"passthrough"``. May be empty.
            key: The shard key, without the ``.parquet`` extension. May contain ``/``.
            subdir: Optional middle component, e.g. the salt prefix or a split name.

        Returns:
            The resolved path. Nothing is created and nothing needs to exist.

        Raises:
            ValueError: If any component is absolute or contains ``..``, or if ``key`` is empty.

        Examples:
            >>> store = ShardStore(Path("/out/.shard_store"))
            >>> store.path("train", "r0000000000-0000000050", subdir="a1b2c3d4").as_posix()
            '/out/.shard_store/train/a1b2c3d4/r0000000000-0000000050.parquet'

            Empty components drop out, and a nested shard name is preserved:

            >>> store.path("passthrough", "held_out/deep/nest/1").as_posix()
            '/out/.shard_store/passthrough/held_out/deep/nest/1.parquet'
            >>> store.path("", "0").as_posix()
            '/out/.shard_store/0.parquet'

            Traversal is impossible by construction:

            >>> store.path("train", "../../../etc/passwd")
            Traceback (most recent call last):
                ...
            ValueError: key must not contain '..'; got '../../../etc/passwd'
            >>> store.path("../train", "0")
            Traceback (most recent call last):
                ...
            ValueError: kind must not contain '..'; got '../train'
            >>> store.path("train", "0", subdir="/etc")
            Traceback (most recent call last):
                ...
            ValueError: subdir must be a relative path; got '/etc'
            >>> store.path("train", "")
            Traceback (most recent call last):
                ...
            ValueError: key must not be empty
        """
        _check_component("kind", kind)
        _check_component("subdir", subdir)
        _check_component("key", key)
        if not key:
            raise ValueError("key must not be empty")
        return self.root.joinpath(*(c for c in (kind, subdir, f"{key}.parquet") if c))

    def has(self, kind: str, key: str, *, subdir: str = "") -> bool:
        """Return whether this shard is already stored *and* carries complete parquet framing.

        The check reads the ``PAR1`` magic that both opens and closes a parquet file rather than
        trusting the name, so a shard left truncated by a killed process (or a file that merely ends
        in ``.parquet``) is reported missing and gets rewritten. The store's own writes are atomic, so
        this only ever sees a finished file or nothing -- but the store is on a shared filesystem
        alongside other tools, and the check costs two ``seek`` calls.

        It is deliberately a *framing* check, not a parse: reading the footer of every shard on every
        run would cost the very I/O the store exists to avoid, and a file that is correctly framed at
        both ends is not something a crash produces.

        Args:
            kind: As :meth:`path`.
            key: As :meth:`path`.
            subdir: As :meth:`path`.

        Returns:
            ``True`` if the resolved path is readable and carries the parquet magic at both ends.

        Raises:
            ValueError: If the components are unsafe, as :meth:`path`.

        Examples:
            >>> tmp = tempfile.TemporaryDirectory()
            >>> root = Path(tmp.name)
            >>> store = ShardStore(root)
            >>> store.has("train", "0")
            False
            >>> _ = store.put("train", "0", lambda p: pl.DataFrame({"subject_id": [1]}).write_parquet(p))
            >>> store.has("train", "0")
            True

            A file that is not parquet, or one truncated mid-write, does not count as stored:

            >>> atomic_write_text("not parquet", store.path("train", "1"))
            >>> store.has("train", "1")
            False
            >>> good = store.path("train", "0")
            >>> _ = good.write_bytes(good.read_bytes()[:-1])
            >>> store.has("train", "0")
            False

            It is a framing check, not a parse: a file that carries the magic at *both* ends is
            trusted, and one that carries it at only one end is not.

            >>> _ = store.path("train", "3").write_bytes(b"PAR1 no schema, no footer PAR1")
            >>> store.has("train", "3")
            True
            >>> _ = store.path("train", "4").write_bytes(b"no header, but ends right PAR1")
            >>> store.has("train", "4")
            False
            >>> tmp.cleanup()
        """
        stored = self.path(kind, key, subdir=subdir)
        n = len(PARQUET_MAGIC)
        try:
            with stored.open("rb") as f:
                if f.seek(0, os.SEEK_END) < 2 * n:
                    return False
                f.seek(-n, os.SEEK_END)
                if f.read(n) != PARQUET_MAGIC:
                    return False
                f.seek(0)
                return f.read(n) == PARQUET_MAGIC
        except OSError:  # missing, a directory, or unreadable -- all mean "not stored"
            return False

    def put(self, kind: str, key: str, write: Callable[[Path], None], *, subdir: str = "") -> Path:
        """Store a shard if it is not already stored, and return its path.

        When :meth:`has` is already true, ``write`` is **not called**: rebuilding a family, or adding a
        member to one, then costs only the shards that member actually introduces. Otherwise ``write``
        is handed a staging path (``<final>.parquet.tmp``) and its output is moved into place with
        :func:`os.replace`, so a concurrent reader sees either nothing or a complete shard. Any
        staging file an earlier, interrupted attempt left at that path is discarded *before* ``write``
        is called, so what :func:`os.replace` promotes is always the bytes this call produced.

        ``write`` is trusted to emit parquet; :meth:`has` rejects anything that is not framed as one,
        so such a shard is rebuilt on every run. Concurrent *writers* of
        the same key share one staging path and must be serialized by the caller -- with
        ``MEDS_transforms.mapreduce.rwlock.rwlock_wrap`` per output shard, as the rest of the ecosystem
        does. :meth:`put` is a skip-if-exists cache, not a lock.

        Args:
            kind: As :meth:`path`.
            key: As :meth:`path`.
            write: Callback that writes a complete parquet file to the path it is given.
            subdir: As :meth:`path`.

        Returns:
            The path of the stored shard.

        Raises:
            ValueError: If the components are unsafe, as :meth:`path`.
            FileNotFoundError: If ``write`` returned without creating the staging file. A ``write``
                that raises propagates; in both cases the staging file, never a partial shard, is what
                is left behind.

        Examples:
            >>> tmp = tempfile.TemporaryDirectory()
            >>> root = Path(tmp.name)
            >>> store = ShardStore(root)
            >>> stored = store.put("train", "0", lambda p: pl.DataFrame({"subject_id": [7]}).write_parquet(p))
            >>> pl.read_parquet(stored)["subject_id"].to_list()
            [7]

            A callback that writes nothing is a programming error, and is reported as one rather than
            leaving the store looking populated:

            >>> store.put("train", "1", lambda p: None)
            Traceback (most recent call last):
                ...
            FileNotFoundError: write callback did not create /...1.parquet.tmp
            >>> store.has("train", "1")
            False

            A callback that raises leaves the staging file, never a shard:

            >>> def boom(p: Path) -> None:
            ...     p.write_text("partial")
            ...     raise RuntimeError("interrupted")
            >>> store.put("train", "2", boom)
            Traceback (most recent call last):
                ...
            RuntimeError: interrupted
            >>> store.has("train", "2"), store.path("train", "2").with_suffix(".parquet.tmp").exists()
            (False, True)

            The next attempt discards that file rather than promoting its bytes, so an interrupted
            run can never be mistaken for a finished shard:

            >>> store.put("train", "2", lambda p: None)
            Traceback (most recent call last):
                ...
            FileNotFoundError: write callback did not create /...2.parquet.tmp
            >>> store.has("train", "2")
            False
            >>> _ = store.put("train", "2", lambda p: pl.DataFrame({"subject_id": [8]}).write_parquet(p))
            >>> pl.read_parquet(store.path("train", "2"))["subject_id"].to_list()
            [8]
            >>> tmp.cleanup()
        """
        stored = self.path(kind, key, subdir=subdir)
        if self.has(kind, key, subdir=subdir):
            return stored

        stored.parent.mkdir(parents=True, exist_ok=True)
        tmp = stored.with_name(stored.name + TMP_SUFFIX)
        tmp.unlink(missing_ok=True)  # never promote what an earlier, interrupted attempt staged
        write(tmp)
        if not tmp.exists():
            raise FileNotFoundError(f"write callback did not create {tmp}")
        os.replace(tmp, stored)
        return stored

    def link_into(self, stored: Path, dst: Path, mode: LinkMode | str) -> None:
        """Place a stored shard into a subset root. Thin delegate to :func:`place`.

        Args:
            stored: A path returned by :meth:`put` (or :meth:`path`).
            dst: Where the subset expects the shard, e.g. ``<subset>/data/train/0.parquet``.
            mode: A :class:`~meds_subsetter.config.LinkMode`, or its string value.

        Raises:
            FileNotFoundError: If ``stored`` does not exist.

        Examples:
            >>> tmp = tempfile.TemporaryDirectory()
            >>> root = Path(tmp.name)
            >>> store = ShardStore(root / ".shard_store")
            >>> stored = store.put("train", "0", lambda p: pl.DataFrame({"subject_id": [1]}).write_parquet(p))
            >>> store.link_into(stored, root / "n1" / "data" / "train" / "0.parquet", "symlink")
            >>> os.readlink(root / "n1" / "data" / "train" / "0.parquet")
            '../../../.shard_store/train/0.parquet'
            >>> missing = store.path("train", "9")
            >>> store.link_into(missing, root / "n1" / "data" / "train" / "1.parquet", "copy")
            Traceback (most recent call last):
                ...
            FileNotFoundError: Cannot place a source that does not exist: /...9.parquet
            >>> tmp.cleanup()
        """
        place(stored, dst, mode)

    def stat(self) -> dict[str, int]:
        """Return ``{"n_files": ..., "n_bytes": ...}`` over the whole store.

        This is the family's *true* cost: the store holds one copy of every distinct shard, however
        many subsets link to it. Compare it against the sum of :func:`dir_size` over the subset roots
        to quantify the saving.

        Returns:
            A dict with ``n_files`` (stored shards under :attr:`root`) and ``n_bytes`` (their apparent
            sizes). Staging files left behind by an interrupted write are not stored shards --
            :meth:`has` reports them missing and :meth:`put` overwrites them -- so they are not
            counted here either, and a crashed earlier run cannot change what a family reports. A
            store that does not exist yet reports zeros.

        Examples:
            >>> tmp = tempfile.TemporaryDirectory()
            >>> root = Path(tmp.name)
            >>> store = ShardStore(root / ".shard_store")
            >>> store.stat()
            {'n_files': 0, 'n_bytes': 0}
            >>> for key in ("0", "1"):
            ...     _ = store.put("train", key, lambda p: pl.DataFrame({"subject_id": [1]}).write_parquet(p))
            >>> store.stat()["n_files"]
            2
            >>> store.stat()["n_bytes"] == sum(p.stat().st_size for p in store.root.rglob("*.parquet"))
            True

            A staging file abandoned by a killed write is not a stored shard -- :meth:`has` says so --
            and is not counted as one either:

            >>> def boom(p: Path) -> None:
            ...     p.write_text("partial" * 100)
            ...     raise RuntimeError("interrupted")
            >>> store.put("train", "2", boom)
            Traceback (most recent call last):
                ...
            RuntimeError: interrupted
            >>> store.has("train", "2"), store.stat()["n_files"]
            (False, 2)
            >>> tmp.cleanup()
        """
        if not self.root.exists():
            return {"n_files": 0, "n_bytes": 0}
        files = [f for f in _iter_files(self.root) if not f.name.endswith(TMP_SUFFIX)]
        return {"n_files": len(files), "n_bytes": sum(f.lstat().st_size for f in files)}


def _iter_files(path: Path) -> Iterator[Path]:
    """Yield every non-directory entry at or under ``path``, without following directory symlinks.

    Args:
        path: A file or directory. A file yields itself.

    Yields:
        Paths of regular files and symlinks. Symlinks to directories are *not* descended into (they
        are not yielded either), which keeps the walk cycle-free.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> (root / "a").mkdir()
        >>> _ = (root / "a" / "x.txt").write_text("x")
        >>> _ = (root / "y.txt").write_text("y")
        >>> sorted(p.relative_to(root).as_posix() for p in _iter_files(root))
        ['a/x.txt', 'y.txt']

        A symlink to a *directory* is neither yielded nor descended into, so a linked-in tree is never
        counted twice and even a cycle terminates:

        >>> (root / "loop").symlink_to(root, target_is_directory=True)
        >>> (root / "a" / "up").symlink_to(root / "a", target_is_directory=True)
        >>> sorted(p.relative_to(root).as_posix() for p in _iter_files(root))
        ['a/x.txt', 'y.txt']

        A symlink to a *file* is yielded, since that is what every placed shard is:

        >>> (root / "z.txt").symlink_to(root / "y.txt")
        >>> sorted(p.relative_to(root).as_posix() for p in _iter_files(root))
        ['a/x.txt', 'y.txt', 'z.txt']
        >>> [p.name for p in _iter_files(root / "y.txt")]
        ['y.txt']
        >>> list(_iter_files(root / "nope"))
        Traceback (most recent call last):
            ...
        FileNotFoundError: No such file or directory: /...nope
        >>> tmp.cleanup()
    """
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(f"No such file or directory: {path}")
    if not path.is_dir():
        yield path
        return
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for name in sorted(filenames):
            yield Path(dirpath) / name


def dir_size(path: Path, *, follow_links: bool = False) -> int:
    """Return the apparent bytes of the file tree rooted at ``path``.

    With ``follow_links=False`` (the default) a symlink contributes only its own tiny size -- the
    length of the target path -- which is what lets a caller show the disk saving of a symlinked
    family. With ``follow_links=True`` each symlink contributes the size of its target instead, which
    is what the family would have cost as independent copies.

    Hardlinks cannot be told apart this way: a hardlinked shard is an ordinary directory entry and is
    counted in full at every path it appears at. The honest measure there is ``du``, or
    :func:`unique_inode_size`, which counts each inode once.

    Args:
        path: File or directory to measure.
        follow_links: Whether a symlink is measured by its target's size instead of its own. Broken
            symlinks contribute 0 in that mode. Directory symlinks are never descended into.

    Returns:
        Total apparent bytes.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> store, subset = root / "store", root / "subset"
        >>> store.mkdir(), subset.mkdir()
        (None, None)
        >>> shard = store / "shard.parquet"
        >>> _ = shard.write_bytes(b"x" * 4096)

        A symlinked placement costs the link, not the shard:

        >>> place(shard, subset / "0.parquet", "symlink")
        >>> dir_size(root) == 4096 + len(os.readlink(subset / "0.parquet"))
        True
        >>> dir_size(root, follow_links=True)
        8192
        >>> dir_size(shard)
        4096

        A broken link contributes nothing rather than raising, so a half-built tree still measures:

        >>> (subset / "broken.parquet").symlink_to("nowhere.parquet")
        >>> dir_size(root, follow_links=True)
        8192
        >>> dir_size(root / "nope")
        Traceback (most recent call last):
            ...
        FileNotFoundError: No such file or directory: /...nope
        >>> tmp.cleanup()
    """
    total = 0
    for f in _iter_files(path):
        try:
            total += (f.stat() if follow_links else f.lstat()).st_size
        except OSError:  # a broken symlink, under follow_links=True
            continue
    return total


def unique_inode_size(path: Path) -> int:
    """Return the apparent bytes under ``path``, counting each ``(st_dev, st_ino)`` exactly once.

    This is the measure that tells the truth about hardlinks, and it matches ``du --apparent-size``:
    a shard hardlinked into ten subsets occupies one inode and is charged once, whereas
    :func:`dir_size` charges it ten times.

    Args:
        path: File or directory to measure.

    Returns:
        Total apparent bytes over distinct inodes. Symlinks have inodes of their own and so are
        counted at their own (tiny) size, exactly as in ``dir_size(path, follow_links=False)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Examples:
        >>> tmp = tempfile.TemporaryDirectory()
        >>> root = Path(tmp.name)
        >>> store, subset = root / "store", root / "subset"
        >>> store.mkdir(), subset.mkdir()
        (None, None)
        >>> shard = store / "shard.parquet"
        >>> _ = shard.write_bytes(b"x" * 4096)
        >>> place(shard, subset / "0.parquet", "hardlink")

        The hardlinked shard looks like 8 KiB of files but occupies 4 KiB of storage:

        >>> dir_size(root)
        8192
        >>> unique_inode_size(root)
        4096

        A copied placement really does cost twice:

        >>> place(shard, subset / "1.parquet", "copy")
        >>> dir_size(root), unique_inode_size(root)
        (12288, 8192)
        >>> tmp.cleanup()
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    for f in _iter_files(path):
        st = f.lstat()
        if (st.st_dev, st.st_ino) in seen:
            continue
        seen.add((st.st_dev, st.st_ino))
        total += st.st_size
    return total
