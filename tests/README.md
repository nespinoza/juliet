# juliet's tests

    pip install pytest pytest-cov
    pytest                            # everything fast: no sampler, no plots
    pytest --cov                      # with coverage
    pytest tests/sampler --run-sampler         # small real fits, in minutes
    pytest tests/scientific --run-scientific   # the paper's fits, in hours

The old scripts that did real fits at import time are now in `examples/validation/`. They are useful to look at, but aren't tests.

## The layers

| Folder | What it checks | Runs |
|---|---|---|
| `unit/` | Single functions against scipy and closed forms. `test_gp_dense.py` checks every GP kernel against a dense calculation. | Always |
| `golden/` | Models, likelihoods and `evaluate` output at fixed parameters, against stored baselines. | Always |
| `contract/` | The evaluation matrix, state handling, public signatures, reading results. Known bugs live here. | Always |
| `sampler/` | Each installed sampler recovers injected values. | `--run-sampler` |
| `scientific/` | K2-140 and K2-32, against the paper's tables and a stored run. | `--run-scientific` |

## What to do when a test fails

**A golden test fails.** Your change altered results. Unless that is what you meant to do, the change is wrong and the baseline is right.

**A strict xfail reports `XPASS(strict)`.** A known bug no longer happens. If you fixed it on purpose, delete the marker. If you didn't, find out why it changed.

**`test_public_signatures_are_unchanged` fails.** If it's on purpose, run `python tests/contract/test_signatures.py` to update the snapshot. Adding a keyword at the end is safe. Removing, renaming or reordering breaks user scripts.

## Every PR has a category

| Category | Meaning | Rule |
|---|---|---|
| **A: pure refactor** | No numerical change | Baselines and xfails untouched. |
| **B: crash to working** | Something that raised now works | Remove the matching xfail. No previously valid output changes. |
| **C: science change** | Valid inputs give different numbers | Separate PR, maintainer's approval, a "Changed results" entry in `CHANGELOG.md`, and a regenerated baseline with the reason. |

## Updating baselines

Only for category C, or for a configuration you just added:

    JULIET_GOLDEN_REASON="Why these numbers changed" pytest tests/golden --update-golden -k "test_against_baseline and <id>"

Without a reason, this refuses to run. The reason, commit, date and dependency versions are added to `golden/CHANGES.md`, and CI fails if `golden/data/` changes without an entry there. Review the diff of what you regenerated, and limit it with `-k` to what should change.

## Adding a capability

1. Add a configuration to `golden/configs.py`. It's only data: an id, a data builder, parameters and options.
2. Generate its baseline as above. Check by some independent means that the numbers are right, because from then on they define correct.
3. If it has a closed form, add a unit test too.
4. If some `evaluate` route doesn't work, leave it out of the configuration's `ops`, and add a strict xfail in `contract/test_known_bugs.py`.

## Tolerances

They're all in `golden/tolerances.py`: `1e-8` absolute on log-likelihoods, `1e-10` relative plus `1e-12` absolute on arrays.

They were measured, comparing all baselines with fresh runs on one macOS machine:

| Environment | Largest log-likelihood difference | Largest relative array difference |
|---|---:|---:|
| Same environment, repeated | 0 | 0 |
| Python 3.13, radvel 1.6.6 | 0 | 0 |
| Python 3.9, numpy 1.24, scipy 1.10, batman 2.4.9, radvel 1.4.11, george 0.4.0, celerite 0.4.2 | `2e-8` (on a value of `-8e6`) | `3e-13` |

The relative tolerance of `1e-13` on log-likelihoods exists for that `-8e6` case. They haven't been measured on Linux, where the first CI run will show whether they need adjusting.

## Samplers

There's no seed for dynesty yet (issue #138), so these tests allow 5 posterior standard deviations from the true values. Measured on this problem:

| Sampler | Runs | Failures | Largest offset |
|---|---:|---:|---:|
| `dynesty` | 40 | 0 | 0.87σ |
| `dynamic_dynesty` | 5 | 0 | 0.72σ |
| `ultranest` | 3 | 0 | 0.72σ |
| `emcee` | 5 | 0 | 0.83σ |
| `multinest` | 5 | 0 | 0.83σ |

MultiNest has a limit on the length of its paths, and going over it stops the whole process with a Fortran error, so its test uses a short relative folder.

## Optional packages

catwoman and kelp configurations are skipped when the package is missing. A kelp baseline hasn't been generated yet, because no installable kelp was available when these were made: run the update command above with kelp installed for the four `lc_kelp_*` configurations. Their parameters haven't been tried, so they may need adjusting.
