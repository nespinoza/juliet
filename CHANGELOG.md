# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.26] - [Unreleased]
### Added
- PR #47, which adds function to `juliet.utils` to read AstroImageJ tables.
- PR #22, which started a unit test script (`test_utils.py`); also added a `.gitignore`.
- Created `CHANGELOG.md` (this file).
- Now `dynesty` runs have the optional `dynesty_n_effective`, `dynesty_use_stop` and `dynesty_use_pool` flags. The first, receives an `int` and defines the maximum number of effective samples to be outputted; default is `np.inf`. The second defines whether the stopping criterion in `dynesty` is used or simply the run is kept going until `dynesty_n_effective` is reached. Default is `True`. Finally, `dynesty_use_pool` defines which steps should use multiprocessing. For details, see the `dynesty` API on the `n_effective`, `use_stop` and `use_pool` flags: https://dynesty.readthedocs.io/en/latest/api.html. 
### Fixed
- Fixed bug that was making `juliet` runs with `dynesty` always go to Dynamic Nested Sampling by default. If `dynamic` is not `True`, the default goes to `dynesty`'s "vanilla" nested sampling.
- Bug that made the `juliet.utils.get_all_TESS_data()` call to download _all_ files. Now by default only lightcurves are downloaded.
