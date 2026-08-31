# meds_subsetter

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/meds-subsetter)](https://pypi.org/project/meds-subsetter/)
[![Tests](https://github.com/mmcdermott/meds_subsetter/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/meds_subsetter/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/mmcdermott/meds_subsetter/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/meds_subsetter/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/mmcdermott/meds_subsetter.svg)](https://github.com/mmcdermott/meds_subsetter/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/meds_subsetter/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

Nested, hash-based subject subsetting, resharding, sizing, and content fingerprinting of MEDS datasets for scaling-law and variance experiments.

## Quick start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. To get set up:

```bash
git clone https://github.com/mmcdermott/meds_subsetter.git
cd meds_subsetter
uv sync
uv run pre-commit install
```

Run the tests:

```bash
uv run pytest -v
```

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the full development guide (build system, testing
conventions, code style, PR workflow).

> [!WARNING]
> There is no folder in this repository for `data`. Datasets — public or private — must be stored
> outside the repository to avoid leaking sensitive data, bloating the repo, or over-fitting the
> project to a single data resource. The same applies to API keys, tokens, and credentials. Note
> that anything committed to `git` history can be recovered through the published repository even
> if removed from the current tree or pushed to a non-main branch; if something sensitive is
> accidentally committed, you need to
> [purge it from history](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Repository management

This repository lives within the
['mmcdermott'](https://github.com/mmcdermott) GitHub organization, which sets default
issue labels that propagate to new repos. All changes to `main` go through pull requests; see the
"Pull Request Workflow" section of [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the expected flow.
Versioning follows semantic versioning, managed through `git` tags (e.g., `git tag 0.0.1`) —
`setuptools-scm` reads the tag and stamps the package version automatically.
