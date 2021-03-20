# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.2] - 2021-03-20
### Removed
- Files for pink noise in deference of building a new package.

### Fixed
- Error on importing `__init__.py` after new versioning file.
- All `np.int` calls have been changed for `int` calls.

## [2.1.1] - 2021-03-19
### Added
- Support for multithreading with `emcee`.
- Added `stepsampler_ultranest` to use Ultranest's stepsampler (see https://johannesbuchner.github.io/UltraNest/example-sine-highd.html).
- `setup.py` automatically installs `ultranest` and `emcee` now, as both are supported by `juliet`.
- Added canonical `_version.py` file to store versions of the code, following user Zooko in stackoverflow question 458550. Modified `__init__.py` and `setup.py` accordingly.
- Added `test_juliet.py` script to give `juliet` a test run with all the samplers.

### Fixed
- `juliet.fit` docstring parsing.
- Bug when both `input_folder` and `out_folder` are given, introduced by the addition of support to `emcee`.

## [2.1.0] - 2021-02-23
### Added
- PR #47, which adds function to `juliet.utils` to read AstroImageJ tables.
- PR #22, which started a unit test script (`test_utils.py`); also added a `.gitignore`.
- Created `CHANGELOG.md` (this file).
- Deprecated the use of several flags (e.g., `use_dynesty`, `use_ultranest`, `dynamic`, etc.); now samplers can be selected using the `sampler` string via `juliet.fit`. Options for each sampler can be directly ingested to `juliet.fit` via `**kwargs`.
- Outputs from multinest are kept in the output folder now (were removed in previous juliet versions).
- Support for `emcee`.

### Fixed
- Fixed bug that was making `juliet` runs with `dynesty` always go to Dynamic Nested Sampling by default. 
- Bug that made the `juliet.utils.get_all_TESS_data()` call to download _all_ files. Now by default only lightcurves are downloaded.
- Refactored the `juliet.fit` class of the code; much easier to read, much easier to pass arguments around, much easier to implement new samplers.
- Removed the `delta_z_lim` flag which didn't do anything; the `delta_z` limit can be inputted through the `kwargs` via `juliet.fit`.
