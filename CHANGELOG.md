# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.26] - [Unreleased]
### Added
- PR #47, which adds function to `juliet.utils` to read AstroImageJ tables.
- PR #22, which started a unit test script (`test_utils.py`); also added a `.gitignore`.
- Created `CHANGELOG.md` (this file).
- Now `dynesty` runs have the optional `dynesty_n_effective` flag, which receives an `int` and defines the maximum number of effective samples to be outputted. Default is `None`. This saves `dynesty` (and thus `juliet`) from going overboard with memory on expensive runs (i.e., runs with large priors or very large number of parameters).
### Fixed
- Fixed bug that was making `juliet` runs with `dynesty` always go to Dynamic Nested Sampling by default. If `dynamic` is not `True`, the default goes to `dynesty`'s "vanilla" nested sampling.
