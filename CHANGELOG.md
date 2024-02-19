# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.2.5] - 
### Fixed
- Bug of multi-instrument fits not updating posteriors by @Jayshil (PR #112).
- Bug of multi-planet TTV fits (issues #110 and #97) by @melissa-hobson (PR #111).
- Bug on joint transit and eclipse models; now remove fp component to out-of-eclipse component --- joint 
  transit and eclipse models have thus, by default, and out-of-eclipse and out-of-transit value equal to 1.

### Added
- Now `get_all_TESS_lightcurves` has an extra flag to save lightcurves (`save_data = True`; PR #106 by @melissa-hobson).
- Tests for transit, eclipse and joint transit and eclipse fits in `tests`.
- Implemented phase-curve toy model (simple sinusoid with phase-offset; amplitude set by secondary depth).
- Added phase-curve test suite under `tests`.
- Support for light-travel time delay in eclipse and transit+eclipse fits (PR #XXX by Taylor Bell). Activate by `dataset.fit(..., light_travel_delay = True)`

## [2.2.4] - 2023-11-14 
### Fixed
- Homogeneized instrument namings on code, which fixed a photometry-only fit bug (fixes #104).
- Removed the `posteriors.dat` file printing out only `omega` values larger than 0 (fixes #90).

## [2.2.3] - 2023-10-30 
### Added
- Support to instrument-dependant `fp`, `p`, `p1` and `p2` thanks to @Jayshil (#85).
- Matern+SHO kernel thanks to @Jayshil (#85).
- Support to instrument-dependant `mflux`, `sigma_w`, and linear models (e.g., `theta0_inst1_inst2`).
- Added `method` to `utils.bin_data`.
- Upgraded list of contributors.

### Fixed
- Made changes in #85 back-compatible with non-instrument dependant `fp`, `p`, `p1` and `p2`.
- Issue with new version of `dynesty` thanks to @andres-jordan (#98).
- Fixed bug when evaluating `catwoman` models thanks to @Jayshil (#96).
- Fixed deprecated `is` comparisons with `==` on samplers thanks to @rosteen (#86).
- Fixed issue #81 thanks to @ssagear (#82).

## [2.2.1] - 2022-08-17
### Fixed
- Changed `ac` for `t_secondary` as the fitting parameter for time of eclipses.

## [2.2.0] - 2022-08-02
### Added
- Upgraded list of contributors.
- `tests` folder where code tests will be saved.
- `zeus` sampler thanks to contribution from Rachel Cooper.
- Secondary eclipse fitting thanks to contribution from Jayshil Patel.
- Additional GP kernels thanks to contribution from Jonas Kemmer.
- Added lowercase priors (thanks to @rosteen; #80).
- Option to activate/deactivate HODLR solver when using `george` GP kernels.

### Fixed
- Bug when multiple planets using and not `efficient_bp` thanks to @tronsgaard.
- Keywords that made `juliet` incompatible with `dynesty` 1.2.2 (this makes `juliet` incompatible with pervious `dynesty` versions)
- `exponential` prior which was not working.
- Problem with `dynesty`, whose newest version crashed with `juliet`.

### Removed
- Removed the `juliet.py` (and associated `utils.py`, `FLAGS.md`) which hosted the original version of the code, to not confuse contributors.

### Fixed
- Supersampling bug when evaluating models.

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
