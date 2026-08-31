"""``python -m meds_subsetter`` dispatcher.

Deliberately logic-free: pytest excludes ``__main__.py`` from doctest collection, so anything here
would be untested. All dispatch lives in :mod:`meds_subsetter.cli`.
"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
