# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.26] - [Unreleased]
### Added
- PR #47, which adds function to `juliet.utils` to read AstroImageJ tables.
- PR #22, which started a unit test script (`test_utils.py`); also added a `.gitignore`.
- Created `CHANGELOG.md` (this file).
- Deprecated the use of several flags (e.g., `use_dynesty`, `use_ultranest`, `dynamic`, etc.); now samplers can be selected using the `sampler` string via `juliet.fit`. Options for each sampler can be directly ingested to `juliet.fit` via `**kwargs`.

### Fixed
- Fixed bug that was making `juliet` runs with `dynesty` always go to Dynamic Nested Sampling by default. If `dynamic` is not `True`, the default goes to `dynesty`'s "vanilla" nested sampling.
- Bug that made the `juliet.utils.get_all_TESS_data()` call to download _all_ files. Now by default only lightcurves are downloaded.
- Removed the `delta_z_lim` flag which didn't do anything; the `delta_z` limit can be inputted through the `kwargs` via `juliet.fit`.
