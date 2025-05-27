# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.8] - 2025-05-27 
### Added
- Users can now fit either for the Kipping (2013) parametrization (q1, q2) or fit directly for coefficients (u1, u2). The former allows for linear, quadratic, square-root and logarithmic laws. The latter for the same plus exponential and power2.
- Users can now also fit for the non-linear law by using the (c1, c2, c3, c4) coefficients.
- Users can now add their own non-linear (additive) functions by passing `non_linear_functions` to `juliet.load`. It has to contain the actual function to be evaluated under `non_linear_functions['function']` and the regressor it will be evaluated on `non_linear_functions['regressor']`.
 
### Fixed
- Now user can print posteriors of fits with eccentricity free and omega fixed.
- Bug on circular orbits with travel time light-delay (see PR #124) thanks to @JeffValenti.

## [2.2.7] - 2024-05-31
### Fixed
- Now dynesty inputs can be given to `juliet` for both, the general properties (e.g., `bound`) _and_ the sampler properties (e.g., nlive).

## [2.2.6] - 2024-05-30
### Fixed
- New dynesty version not using all cores with how multiprocessing is incorporated in `juliet`. Now it works, but one has to add the pesky `if __name__ == '__main__':` to the start of scripts using this.
- Fixed `Gamma` parameter on the Exp-sine-squared kernel, for which `juliet` was actually fitting log(`Gamma`) and not `Gamma`, as pointed out in #118.
- Fixed bug on binning with given errorbars pointed out in #117.
- Fixed bug that didn't let `dynesty` args to be passed with newest `dynesty` version (as noted in #109).

### Added
- Exception if user tries to fit `a_p1` and `rho` (covering issue #116).


## [2.2.5] - 2024-03-01 
### Fixed
- Bug of multi-instrument fits not updating posteriors by @Jayshil (PR #112).
- Bug of multi-planet TTV fits (issues #110 and #97) by @melissa-hobson (PR #111).
- Bug on joint transit and eclipse models; now remove fp component to out-of-eclipse component --- joint 
  transit and eclipse models have thus, by default, and out-of-eclipse and out-of-transit value equal to 1.
- Bug on setup file that didn't install h5py by default (needed for radvel) (PR #113).

### Added
- Now `get_all_TESS_lightcurves` has an extra flag to save lightcurves (`save_data = True`; PR #106 by @melissa-hobson).
- Tests for transit, eclipse and joint transit and eclipse fits in `tests`.
- Implemented phase-curve toy model (simple sinusoid with phase-offset; amplitude set by secondary depth).
- Added phase-curve test suite under `tests`.
- Support for light-travel time delay in eclipse and transit+eclipse fits (PR #113 by Taylor Bell). Activate by `dataset.fit(..., light_travel_delay = True, stellar_radius = your_value)`. Note this applies the correction in this PR only on the eclipses via comparing radial distances from the eclipses to the time-of-transit center, which generates time delays which are subtracted iteratively to the measured time-stamps. This means the transits (or, really, the time-of-transit t0) are used as the references for the correction. In general practice this has little impact, except for (a) distortions that might be injected in phase-curves and (b) for comparisons with codes that apply this to the entire orbit (e.g., starry). On this latter ones, the reference to measure times of the orbit is typically at the center of mass of the system (or the star); in `juliet`, the reference for the time-stamps is the mid-transit point. As a practical example: for Earth, starry would measure a transit 8 mins earlier and an eclipse 8 minutes late. With `juliet`, you would measure the transit at t0, and the eclipse 16 minutes late to the no-delay case. The impact on the orbital parameters is the same in both cases.

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
