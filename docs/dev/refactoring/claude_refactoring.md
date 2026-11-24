# juliet refactoring report

Baseline: `juliet` 2.2.11, commit `986724e`, branch `claude-refactoring`, reviewed 2026-11-03.
Updated 2026-11-04, after Priority 1 and two kernel fixes were made on branch `tests/framework`. That work is committed locally as `69f2634` (tests) and `8399ef6` (kernel fixes), and isn't pushed. Line numbers in section 2 refer to `986724e`.

## Where things stand

| Item | Status |
|---|---|
| **Priority 1** (tests, baselines, CI) | Built. `pytest`: 448 passed, 69 strict xfails, 17 skipped, in about 21 s. The CI workflows have never run. |
| **Kernel fixes** (`GP_Gamma`, Double SHO `omega2`) | Done, with docs and `CHANGELOG`. 8 of the 132 baselines were regenerated, and no others changed. |
| **Global `evaluate` shape** | Decided (section 6). The tests expect it, and the code comes with Prompt 2B. |
| **Priorities 2 to 5** | Not started. |
| **Your decisions** | Section 6 lists what's still open, including new findings from Priority 1. |

## Summary

**The GP mathematics in juliet is mostly sound, but the code around it is fragile.** Two of the 10 kernels had errors, now fixed. For RV fits with a celerite Matern GP, in both global and per-instrument modes, juliet's log-likelihoods and GP predictions match an independent dense-matrix calculation to about 10⁻¹⁰. The mess you described is in how evaluation manages state. `evaluate_model` temporarily rewrites the model object, and each combination of global/per-instrument, LC/RV and `t`/no `t` restores that state differently.

**Before Priority 1, there was almost no safety net.** The 20 old `tests/test_*.py` scripts contain no assertions, none of them uses RVs, GPs or TTVs, and there was no CI. Refactoring would have changed results without anyone noticing. The new suite in `tests/` closes this gap.

**A safe and very fast test layer is available.** Nearly all of the science is in `generate_*_model` + `get_log_likelihood`, which are deterministic given fixed parameters. 1000 transit likelihood evaluations on 210 points took 0.064 s here. Thousands of frozen reference values can therefore run on every PR without a sampler.

**Five things you may want to know before reading further:**

1. **Multi-planet TTV fits are silently wrong (#135).** I reproduced this and found the root cause. Fixing it changes results, so it needs your sign-off.
2. **dynesty keyword arguments are dropped already on dynesty 2.1.5, not only on v3 (#138).** This includes `rstate`, so seeded dynesty runs are currently impossible.
3. **Global photometric GPs can fit but can't be evaluated.** Every evaluation route I tried raises an error on a fresh model.
4. **Two kernels were wrong, and are now fixed.** The exp-sine-squared kernel used `ln(GP_Gamma)` from 2.2.6 to 2.2.10, and the double SHO kernel used `Q1` in `omega2`. See section 2.3.
5. **The K2-32 tutorial may not reproduce the paper's Table 10.** The evidence points to different GP conventions. See section 6.

### The five priorities

| # | Priority | Main goal | Why at this position |
|---|---|---|---|
| 1 | **Testing suite, frozen baselines and CI** | Monitor changes | Everything else depends on it |
| 2 | **One noise model and stateless evaluation** | Understanding, new capabilities | Your largest pain point, and it clears the most open issues |
| 3 | **Composable forward models** | New capabilities | About 45% of `generate_lc_model`'s lines are four copies of one block, and the TTV bug lives here |
| 4 | **One parameter system and a validated config** | Understanding | The rules for prior names are duplicated and differ between LC and RV |
| 5 | **Sampler adapters, module split, packaging and developer docs** | All three | Lowest risk once the rest is protected |

---

## 1. How juliet works today

juliet (Espinoza, Kossakowski & Brahm 2019) models photometry and RVs jointly and compares models using Bayesian evidence.

| Concept | Paper | Code |
|---|---|---|
| Photometric model: `[T·D + (1−D)] / (1 + D·M)` | Eq. 5–6 | `generate_lc_model`, `fit.py:4523-4528` |
| Linear model, added to the above | Eq. 1–2 | `fit.py:4531-4543` |
| RV model: Keplerians + μ + trend | Sec. 2.2 | `generate_rv_model`, `fit.py:2531-2634` |
| Per-instrument likelihood | Eq. 3 | `get_log_likelihood`, `fit.py:4594-4616` |
| Global likelihood (one shared GP realization) | Eq. 4 | `get_log_likelihood`, `fit.py:4581-4593` |
| GP on residuals | Eq. 7 | `gaussian_process`, `fit.py:5245` |

Call path:

```text
juliet.load    parses prior names, sets flags in lc_options/rv_options, builds GP objects
  └─ .fit() → juliet.fit
       ├─ model('lc'), model('rv')    long-lived objects holding mutable state
       ├─ sampler → loglike(cube) → generate_*_model → get_log_likelihood
       └─ results.lc / results.rv .evaluate(...)    temporarily rewrites the model object
```

Most of the behavior lives in a few very large functions. The five largest make up 66% of `fit.py` (3838 of 5781 lines):

| Function | Lines | Branches | Max nesting |
|---|---:|---:|---:|
| `model.generate_lc_model` | 1052 | 114 | 11 |
| `model.evaluate_model` | 852 | 162 | 11 |
| `fit.__init__` | 783 | 110 | 8 |
| `model.__init__` | 607 | 110 | 8 |
| `load.generate_datadict` | 544 | 135 | 7 |

---

## 2. Evidence

Findings are labeled by how I established them:

- **[ran]**: reproduced by running this checkout.
- **[read]**: from reading the code only, not executed.
- **[issue]**: as reported on GitHub.

Probes ran on Python 3.12, numpy 2.2.6, batman 2.5.1, radvel 1.6.4, george 0.4.4, celerite 0.4.3 and dynesty 2.1.5, using small synthetic datasets. Line numbers refer to `986724e`.

### 2.1 What works, and must stay as it is

- **[ran]** RV log-likelihood and GP prediction match a dense calculation, using celerite's Matern with two instruments and differing jitters:

  | Mode | Log-likelihood difference | Max prediction difference |
  |---|---:|---:|
  | Global | 1.7×10⁻¹⁰ | 8.6×10⁻¹⁰ |
  | Per-instrument | −1.8×10⁻¹⁰ | 8.4×10⁻¹⁰ |

- **[ran]** George's SE kernel also matches a dense calculation through the default HODLR solver (agreement to the 6 decimals compared), under the convention described in section 2.4.
- **[ran]** Per-instrument evaluation at `t` = the data times reproduces evaluation with `t=None` exactly.

`tests/unit/test_gp_dense.py` now makes this comparison for all 10 kernels, for LC and RV, global and per-instrument. This is what exposed the two kernel bugs in section 2.3. After the fixes, all 10 match their closed forms, and the double SHO closed form matches celerite2's own `RotationTerm` to about 3×10⁻¹⁵.

### 2.2 The GP evaluation matrix [ran]

Each cell uses a fresh model with posterior samples loaded and a celerite Matern GP. I also confirmed three global RV cells through a real `dataset.fit()` with dynesty: `evaluate(inst)`, the `t=` form, and `evaluate(inst)` after a `t=` call.

| Call | RV per-instrument | RV global | LC per-instrument | LC global |
|---|---|---|---|---|
| `evaluate(inst)` | OK | ❌ `AttributeError`, `fit.py:2648` | OK | ❌ `AttributeError`, `fit.py:2648` |
| `evaluate()` | not applicable | ❌ `AttributeError`, `fit.py:2648` | not applicable | ❌ `KeyError: None`, `fit.py:2754` |
| `evaluate(inst, t=, GPregressors=)` | OK | OK | OK | ❌ `KeyError`, `fit.py:3083` (2 instruments); `UnboundLocalError`, `fit.py:3094` (1 instrument) |
| `evaluate(t=, GPregressors=)` | not applicable | ❌ `KeyError: None`, `fit.py:2613` | not applicable | not run |
| `evaluate(inst, t=, GPregressors=<dict>)`, as the docstring instructs | not run | ❌ `TypeError`, `fit.py:2650` | not run | not run |
| `evaluate(inst, parameter_values=<scalars>)` | not run without `t` | ❌ `AttributeError`, `fit.py:2648` (#125) | not run without `t` | ❌ `AttributeError`, `fit.py:2648` |
| `evaluate(inst, parameter_values=<scalars>, t=...)` | ⚠️ **`t` ignored**: returned 30 points, not 77 | not run | ⚠️ **`t` ignored**: returned 120 points, not 55 | not run |
| `evaluate(inst)` after a `t=` call | ❌ `ValueError`, `fit.py:2664` | ⚠️ returns all 50 points, not `inst`'s 30 | ❌ `ValueError`, `fit.py:2664` | not reachable |

**Why the RV tutorials work anyway.** They call the `t=` form first, and that call happens to set `self.original_GPregressors` (`fit.py:3027`). Right after a global RV fit, `results.rv.evaluate('A')` raises. After one `t=` call, the same line succeeds.

**Causes [read]:**

- `original_GPregressors` is only assigned inside the `t is not None` branch, but `get_GP_plus_deterministic_model` always uses it for global GPs.
- Non-global `noise_model.X` is overwritten (`fit.py:3038`) and never restored. Only the global path restores it (`fit.py:3486-3489`). This is why a second `evaluate` fails.
- `supersample_params` and `sample_params` are reset inside the instrument loop (`fit.py:2864-2865`), so only the last instrument survives.
- `dummy_indexes` and `original_instrument_indexes` are only defined in the RV branch (`fit.py:2944-2950`), but the global LC code uses them (`fit.py:3094`, `3233`).
- The RV `t=` path doesn't update `ndatapoints_per_instrument`, so a linear model has the wrong length (next section). The LC `t=` path doesn't update `self.times`, which is the cause of #132 (open fix in PR #137).

**RV and photometric GPs use different mechanisms for the same job [read]:**

| Concern | Per-instrument | Global |
|---|---|---|
| Jitter | A kernel term (`JitterTerm`/`white_noise`) | Added to `yerr`, then `compute()` is called |
| When GP is recomputed | Implicitly, when parameters change | Explicitly, in both likelihood and predict |
| Likelihood failure | Caught, returns `-inf` (`fit.py:4604-4609`) | Not caught, so it stops the run (`fit.py:4588-4590`) |
| Shape returned for `inst` | That instrument | All instruments stacked (to change: section 6) |
| Data order | As given | May be re-sorted by `sort_GP`; `self.t` then differs from `self.times` |

This matrix is now `tests/contract/test_evaluate_matrix.py`, which adds `return_err`, `return_samples`, `return_components` and repeated calls: 34 of its cells are strict xfails.

### 2.3 Other verified defects

**Fixed on `tests/framework`, in 2.2.11**

| Finding | Evidence and effect |
|---|---|
| **`GP_Gamma` entered the exp-sine-squared kernel as `ln(Γ)`, from 2.2.6 to 2.2.10.** The 2.2.6 change for #118 assumed that george's parameter vector holds a logarithm, but `gamma` is the exception. A value below 1 gave a negative coefficient. | **[ran]** george 0.4.4 reports `('gamma', 'log_period')`, and a kernel value that matches `exp(−γ sin²)` with `γ` linear. Versions up to 2.2.5 had it right, so this reverts the 2.2.6 change. On the test LC datasets, ln L changed by about +1000 and on the RV datasets by about −1 (at `Γ = 2`). |
| **`CeleriteDoubleSHOKernel` used `Q1` in `omega2`,** while the `RotationTerm` it follows uses `Q2`. | **[ran]** The old `omega2` is 0.58 of the correct one at `Q0 = 0.1, dQ = 1`; 0.96 at `Q0 = 1, dQ = 1`; identical at `dQ = 0`. On the test datasets, ln L changed by about +45 for LC and −0.2 for RV. `GP_Q0` must now be above 0, as `omega2` diverges at 0. |

Fits with either kernel have to be redone or rechecked. No other kernel or configuration changed.

**Changes science results or is silently wrong**

| Finding | Evidence |
|---|---|
| **Multi-planet TTVs use the wrong time stamps (#135).** `dummy_time` is reassigned per planet in one loop (`fit.py:3649`) and used for every planet in a later loop (`fit.py:3726` onward), so the last planet's values are used for all of them. | **[ran]** Two planets: moving the inner planet's `dt` by 0.02 d changed the model by exactly 0. Moving the outer planet's `dt` changed the **inner** planet's light curve by up to 0.0098. |
| **RV `non_linear_functions` is ignored.** | **[ran]** The function was called 0 times and the likelihood was identical with and without it. |
| **Scalar `parameter_values` with `t=` ignores `t`.** | **[ran]** See the matrix above. |
| **dynesty keyword arguments are silently dropped (#138, also on 2.1.5).** The code looks up `vars(DynestySampler)['__init__']`, which doesn't exist (`fit.py:2043`, `2104`). | **[ran]** Passing `bound='single', sample='unif', rstate=...`, the sampler received `bound='multi', sample='rwalk'` and no `rstate`. Same-seed runs gave lnZ −161.33 and −162.52. |
| **Global GP regressor order is not validated.** | **[ran]** A misaligned george regressor gave log-likelihood −131.7 instead of −91.6, with no warning. No test covers this yet. |
| **`phaseoffset` is ignored for eclipse-only models** (`fit.py:3974`). | **[ran]** |
| **`evaluate_beta` tests the hyperparameters instead of `x`** (`utils.py:661`), so any Beta prior with `b ≥ 1` gives `-inf` under MCMC samplers. | **[ran]** |
| **`get_quantiles` gives a median half a sample off for even-sized input** (`utils.py:995`). | **[ran]** |
| **`bin_data` drops the last point of each bin** (`utils.py:1017`). | **[ran]** |

**Crashes or breaks a documented feature**

| Finding | Evidence |
|---|---|
| A global model with no GP returns `None` from `get_log_likelihood`, because of a missing `return` (`fit.py:4592`). | **[ran]** |
| `extra_loglikelihood` raises `NameError` when used (`fit.py:1693`, missing `self.`). | **[ran]** |
| RV linear model with `t=` raises `ValueError` (`fit.py:2623`). The 2.2.9 fix covers evaluation at the data times only (#128). | **[ran]** |
| Shared `sigma_w_A_B` works for LC without a GP, but raises `KeyError` with a per-instrument GP (`fit.py:5411`) or a global GP (`fit.py:4573`). | **[ran]** |
| RVs don't support shared `mu_A_B` or `sigma_w_A_B` (`KeyError`, `fit.py:2614`, `2617`). | **[ran]** |
| A 1-D unsorted global regressor raises `IndexError` in `sort_GP` (`fit.py:479`). | **[ran]** |
| Unknown GP parameter names give `KeyError: None` (`fit.py:5624`) instead of a useful message. | **[ran]** |
| `t=` crashes with light-travel delay (`fit.py:4104`), `dt` TTVs (`fit.py:4394`) and non-linear functions (`fit.py:4560`). | **[ran]** |
| `return_components` crashes on multi-instrument RVs (`fit.py:3204`) and on instruments with no transit (`fit.py:3194`). | **[ran]** |
| An instrument named like a GP parameter, such as `B` or `C`, raises `KeyError`, because the instrument is looked for anywhere in the prior name (`fit.py:5624`). | **[ran]** |

**State leaks between calls**

| Finding | Evidence |
|---|---|
| `load` keeps your input dictionaries by reference (`d.times_rv is T`), and `evaluate` writes into them. | **[ran]** After a failed evaluate, the user's own `T['A']` had 77 points instead of 30. This also broke later `load` calls in my probe. |
| A failed `evaluate` leaves `self.inames` narrowed to one instrument. | **[ran]** |
| `load` sorts your input arrays in place for global celerite GPs. | **[ran]** |
| `set_posterior_samples` hides failures behind a bare `except` (`fit.py:4630-4635`). There are 19 bare `except:` statements in `fit.py` and `utils.py`. | **[read]** |

**Packaging and infrastructure [read]**

- `tests/framework` adds CI workflows and a `pyproject.toml` holding only test configuration. Packaging itself still uses `setup.py`.
- `setup.py` has `extras_requires` (should be `extras_require`), so the extras are silently ignored, and it declares `python_requires='>=2.7'`.
- `juliet/plots.py` imports `mpl_toolkits.axes_grid`, which modern matplotlib has removed.
- `juliet/__main__.py` is a stub.
- `from .fit import *` makes `juliet.fit` the class, hiding the module of the same name.
- `sort_GP` is still O(N²) (`fit.py:454`), so #120's slow loading likely remains for global models with unsorted data.
- `CeleriteTripleSHOKernel` in `set_parameter_vector` has no matching kernel definition, so it is dead code.

**Old test scripts, now in `examples/validation/` [ran grep]**

| | Count (of 20) |
|---|---:|
| Contain an assertion | 0 |
| Use RV data | 0 |
| Use a GP | 0 |
| Use TTVs | 0 |
| Call `plt.show()` | 19 |
| Protected by `if __name__ == '__main__'` | 4 |

16 of the 20 scripts do their work at import time, so collecting them started real fits and opened plots. Having moved them, `pytest` is now safe to run. Your untracked `tests/test_transit_and_eclipse_ltd.py` stays where it was, and `tests/conftest.py` excludes it.

### 2.4 Conventions that differ from the docs, and should not be silently "fixed"

Changing any of these would change results for existing prior files. Lock them in with tests and document them. Whether to change them later is your decision.

| Quirk | Documented | Actual |
|---|---|---|
| `GP_sigma` units, LC, george kernels | ppm | ppm **[ran]**: `GP_sigma=100` gives amplitude 1×10⁻⁴ |
| `GP_sigma` units, LC, celerite Matern | ppm | **relative flux [ran]**: `GP_sigma=100` gives k(0)=10⁴ |
| `GP_sigma` meaning, celerite exponential and exponential×Matern | Amplitude | **Acts as the variance [ran]**: `GP_sigma=100` gives k(0)=100 (#136) |
| `GP_timescale` | Time scale | **A rate [read]**: passed as celerite's `c` in `exp(−cτ)` (#136). Unchanged since at least 2019. |
| SE `GP_alpha` | Paper: `exp(−α d²)` | **`exp(−α d²/2)` [ran]**: only this form matched the dense calculation |
| White noise in global george GPs | None | **Variance `1.25e-12` is always added [ran]**: george's default, about 1.1 ppm in quadrature. It shifted ln L by about 0.3 on the 100-point test LC, and matters only for photometry. |

`tests/golden/CONVENTIONS.md` has the full covariance computed by each kernel.

`evaluate` also picks a random subset of samples unless `all_samples=True` **[ran]**, so tests must use `all_samples=True` or set a seed.

### 2.5 Issue themes

I read the open issues (excerpts and selected comments) and scanned closed issues by title.

| Theme | Open issues | Addressed by |
|---|---|---|
| Evaluation and components | #125, #121, #132, #128, #56, #40 | Priority 2 |
| Noise and GP design | #136, #18, #35 | Priority 2 |
| Forward model and new physics | #135, #134, #114, #122, #48, #69 | Priority 3 |
| Validation and user errors | #31, #41, #54, #94, #119 | Priority 4 |
| Derived outputs | #21, #61, #115 | Priority 4 |
| Samplers and parallelism | #138, #130, #133, #95, #87, #78 | Priority 5 |
| Installation, docs and constants | #129, #92, #77, #37, #27 | Priority 5 |

**A cautionary example:** in the comments on #118, a user reports that an SE-kernel fit that worked through 2.2.5 began to diverge after upgrading to 2.2.6/2.2.7. The changelog for 2.2.6 records a kernel parametrization change, for the exp-sine-squared `Gamma`. That change was itself a bug (section 2.3), but it doesn't touch the SE kernel, so it doesn't explain this report. I didn't establish what does, which is the point: frozen baselines would show exactly what changed between those versions.

---

## 3. Ground rules for every refactoring PR

Every change falls into one of three categories, and a PR must state which:

| Category | Meaning | Rule |
|---|---|---|
| **A: Pure refactor** | No numerical change | Reference values untouched. Must be the default. |
| **B: Crash to working** | Something that raised now works | Remove the matching `xfail`. No previously valid output may change. |
| **C: Science change** | Valid inputs give different numbers | Separate PR, your explicit approval, a "Changed results" `CHANGELOG` entry, and a regenerated baseline with the reason recorded. |

Never combine a refactor and a bug fix in one commit.

---

## 4. The five priorities

### Priority 1: Testing suite, frozen baselines and CI

**Status: built on `tests/framework`, in commit `69f2634`.** Everything below this block, including the prompts, is the original plan, kept for the record.

| Check | Result |
|---|---|
| Default suite | 448 passed, 69 strict xfails, 17 skipped (9 need a flag, 8 are kelp), in about 21 s |
| Baselines | 132 configurations, 1.1 MB |
| Coverage, measured before the kernel fixes | 63% overall, 67% of `fit.py`, 57% of `utils.py`, 15% of `KeplerOrbit.py` |
| Deliberate breaks, in a throwaway copy | 12 of 12 caught, including a 10⁻⁹ dilution change. A no-mutation control run was clean. |
| Noise against baselines | 0 on repeat and on Python 3.13. At most 3×10⁻¹³ on arrays with Python 3.9/numpy 1.24; 2×10⁻⁸ on a ln L near −8×10⁶. |
| Sampler recovery at a 5σ limit | No failures: dynesty 40 runs (worst 0.87σ), dynamic dynesty 5, UltraNest 3, emcee 5, MultiNest 5 (each 0.83σ or better) |

**Strict xfails:** 34 in the evaluation matrix, 23 known bugs, 9 on state and 3 unit tests. `pytest -rx` lists them with their reasons.

**Not verified:**

- The CI workflows have never run. `celerite` and `george` builds on Python 3.10–3.13 are the likeliest problem, and Linux differences are unmeasured.
- The scientific fits haven't been run, so no scientific baseline is stored. `test_against_paper` for K2-32 may fail (section 6).
- `K2_LONG_CADENCE` supersampling (20, 0.020434) in those fits is an assumption.
- kelp wouldn't install, so its 4 configurations are placeholders with no baselines and untried parameters. zeus isn't tested.
- The `paper_k2_32` baseline is only a regression pin, not evidence that the values are right.
- Golden tests emit about 22,000 numpy "ndim>0 to scalar" deprecation warnings, whose source I haven't traced.

**Differences from the prompts:**

- Noise uses `RandomState`, not `Generator`, because numpy guarantees that its stream won't change.
- All three prompts were done in one session and one commit, rather than one PR each.

**Goal.** Any numerical or API change becomes visible on the PR that introduces it, in minutes.

**Test layers:**

| Layer | What it checks | Speed | Runs |
|---|---|---|---|
| **Unit** | Parametrizations `(r1,r2)→(b,p)`, `(q1,q2)→(u1,u2)`, eccentricity forms, prior transforms against scipy, each kernel against a closed form | Seconds | Every PR |
| **Frozen numerics** | `generate` + `get_log_likelihood` + model arrays at fixed parameters, compared with stored `.npz` files | Seconds to minutes | Every PR |
| **API contract** | The whole evaluation matrix in section 2.2, known bugs as strict `xfail`, public signatures, input dictionaries left unmodified, `posteriors.pkl`/`posteriors.dat` read and write | Seconds | Every PR |
| **Sampler smoke** | Small fits recover injected values for each installed sampler | Minutes | Every PR, or nightly |
| **Scientific** | K2-140 and K2-32 from the repo's tutorial data, compared with the paper's tables and with a stored baseline run | Hours | Weekly or before a release |

**Coverage for the frozen numerics layer.** Build a matrix across these dimensions rather than every combination:

- **Data:** LC only, RV only, joint; 1, 2 or 3 instruments; 1 or 2 planets
- **Transit parametrization:** `(b,p)`, `(r1,r2)`, `a` or `rho`; each of the 3 eccentricity forms
- **Limb darkening:** all laws; `q`, `u` and `c` coefficient forms
- **Signals:** transit, eclipse, transit+eclipse, each phase-curve model, catwoman, light-travel delay, supersampling, TTVs (`dt` and `T`)
- **Noise:** white noise, every kernel × global and per-instrument × LC and RV
- **Additional terms:** linear models, non-linear functions (additive and multiplicative), RV trends, shared instrument parameters

**Tolerances.** Measure them; don't guess. Start around 10⁻⁸ absolute on log-likelihood and 10⁻¹⁰ relative on arrays, then adjust based on the differences seen between Linux and macOS, especially for HODLR.

**Baselines must include known-wrong behavior.** The single-planet TTV baseline stays. Multi-planet TTVs should be a strict `xfail` asserting correct physics, so that the fix is impossible to miss.

**Seeded dynesty runs need a small fix first** (#138). Until then, the smoke layer has to use statistical tolerances.

**Done when:**

- `pytest` passes on a clean clone in under 5 minutes without a sampler or matplotlib window.
- Every row in section 2.2 has a test.
- CI runs pinned and latest-dependency jobs.
- A one-line change to any likelihood formula fails CI.

#### Prompt 1A: move the scripts and build the framework

```text
Read claude_refactoring.md, sections 2.3, 3 and "Priority 1". Do not change anything under juliet/.

Set up the juliet test framework:
1. Tag the current commit `baseline-2.2.11` locally. Do not push.
2. Create a branch from the current one.
3. `git mv` the existing tracked scripts in tests/ to examples/validation/ with their names unchanged, and
   move their README too. tests/test_transit_and_eclipse_ltd.py is untracked: leave it where it is, tell me
   about it, and add it to pytest's ignore list. These scripts do real fits at import time, so do not
   execute them.
4. Add pyproject.toml containing only [tool.pytest.ini_options] and test/coverage configuration, with
   testpaths=["tests"] and the markers: unit, golden, contract, sampler, scientific, requires_catwoman,
   requires_kelp, requires_multinest. Do not replace setup.py yet.
5. Add tests/conftest.py with: small deterministic synthetic datasets for LC and RV (fixed numpy Generator
   seeds, one and two instruments, unsorted interleaved multi-instrument times); a helper that builds
   juliet.load + juliet.model directly, with posterior samples, without running a sampler; deep-copied
   inputs for each test; a non-interactive matplotlib backend; and tmp_path for every out_folder.
6. Add a `--update-golden` option. It must refuse to run unless a non-empty JULIET_GOLDEN_REASON environment
   variable is set, and must append the reason, commit, date and dependency versions to
   tests/golden/CHANGES.md.
7. Move the assertions in juliet/test_utils.py to tests/unit/, removing the empty test.
8. Add first unit tests for: reverse_ld_coeffs/reverse_q_coeffs round trips, reverse_bp against the formula in
   Sec. 2.1 of Espinoza et al. 2019 including pl/pu != (0,1), get_phases, get_quantiles, bin_data, every
   transform_* against scipy.stats ppf, and every evaluate_* against scipy.stats logpdf.

If a test reveals behavior that looks wrong, do not fix it: mark it xfail(strict=True) with a reason and list it
for me. Finish by running pytest and reporting the results, the runtime and coverage. Commit nothing until I
have reviewed the summary.
```

#### Prompt 1B: freeze the numerics

```text
Read claude_refactoring.md, sections 2 and 3 and "Priority 1". Do not change anything under juliet/.

Add a frozen-numerics test layer to tests/golden/:
1. Define configurations declaratively in tests/golden/configs.py. Each has an id, data builder, priors, fixed
   parameter values and load/model options. Cover every dimension in the Priority 1 coverage list. Find further
   supported options by reading juliet/fit.py: treat load.generate_datadict and the kernels in
   gaussian_process.__init__ as the authoritative list. Use pairwise combinations, not the full product, but
   every kernel must run in global and per-instrument mode for both LC and RV.
2. For each, store in tests/golden/data/<id>.npz: log-likelihood, each instrument's deterministic model, M, LM,
   variances, each planet's model, RV Keplerian and trend, and, at evaluate(all_samples=True), the full model,
   the GP component, and return_components output, wherever each one is reachable today.
3. Run every test under both log_like_calc=True and False, and assert that they agree. Those code paths are
   separate copies in generate_lc_model.
4. Add tests/unit/test_gp_dense.py: for each kernel, compare juliet's log-likelihood and GP prediction with an
   independent dense numpy calculation. Note celerite's Matern is
   sigma^2 exp(-w tau)(cos(eps tau) + (w/eps) sin(eps tau)), with eps multiplying tau, which differs slightly
   from the paper's expression. George's SE is exp(-alpha d^2 / 2). Assert what the code does today, and record
   each difference from the docs in tests/golden/CONVENTIONS.md.
5. Measure noise before choosing tolerances: generate twice, under the installed dependency versions, and report
   the largest difference. Put the tolerances in one place. Skip optional dependencies with the markers rather
   than failing. Tell me clearly which configurations were not generated because a package was missing.
6. For configurations that can't run because of a known bug (see section 2), do not store anything: add a strict
   xfail with the exact exception and a pointer to the report's section.

Prove the safety net works: on a temporary, uncommitted change, alter (a) the dilution formula, (b) an RV jitter
term, and (c) a kernel's parameter mapping, confirm tests fail for each, and revert. Report the results, the
total runtime and any configurations you could not cover.
```

#### Prompt 1C: API contract, CI and scientific regression

```text
Read claude_refactoring.md, sections 2.2, 2.3 and 3 and "Priority 1".

1. Add tests/contract/test_evaluate_matrix.py, which parametrizes every row and column of the table in section 2.2.
   Assert the correct, documented behavior: expected output shape, per-instrument results, and evaluating twice in
   any order gives the same answer. Mark current failures xfail(strict=True, raises=<exact exception>).
2. Add contract tests for: public signatures of load, fit, model.evaluate_model, gaussian_process and
   juliet.utils functions, snapshotted with inspect.signature; input dictionaries unchanged after load, fit and
   evaluate, including failed calls; model state (inames, times, t, instrument_indexes, noise_model.X) unchanged
   after evaluate, including failed calls; reading posteriors.pkl and running from an existing out_folder; the
   text format of posteriors.dat. Add a small committed fixture pickle created by the baseline code so that
   backward compatibility with existing pickles is tested.
3. Add tests/sampler: a small fit for each installed sampler (dynesty, dynamic_dynesty, ultranest, emcee,
   multinest), asserting recovery of injected parameters within a statistical tolerance chosen so that it can't
   realistically flake. Measure this over at least 20 runs for dynesty and report the observed failure rate.
4. Add tests/scientific: K2-140 and K2-32 using tutorial1/ and tutorial2/, with their prior files. The paper's
   values are in Tables 8 and 10 of arXiv:1812.08549. Check them against the PDF, not a transcription, and don't
   use Table 9's priors for K2-32, because its P and t0 rows are a known copy error. Compare in units of the quoted
   uncertainty. Do not run these fits yourself unless I ask, since they can take hours. Produce a script that
   stores a baseline for later comparison.
5. Add .github/workflows/: tests.yml running on PRs against Python 3.10-3.13 on Linux, plus one macOS job, with
   a pinned-dependency job as the reference and a latest-dependency job that is allowed to fail but visibly reports;
   and scientific.yml, on manual dispatch and a weekly schedule. Report coverage, without a threshold for now. Add a PR
   template with the A/B/C category from section 3 of the report, and a CI check that fails if tests/golden/data
   changes without an entry in tests/golden/CHANGES.md.

Do not fix any juliet bugs in this PR. Report the final list of xfails as a checklist that I can work through. Do not
push or open PRs without asking me.
```

---

### Priority 2: One noise model and stateless evaluation

**Goal.** Evaluation becomes one simple code path, and `evaluate` can't change the model. This is the most direct fix for the GP mess you described, particularly with RVs.

**Target design:**

```python
class NoiseModel:                # White, PerInstrumentGP, GlobalGP
    def log_likelihood(self, residuals, variances, params) -> float: ...
    def predict(self, residuals, variances, params, X_new) -> ndarray: ...

class Kernel:                    # one small declarative class per kernel
    param_names = ("sigma", "rho")
    backend = "celerite"
    def build(self, ...): ...
    def to_backend(self, params) -> ndarray: ...
    def value(self, tau, params) -> ndarray: ...   # closed form, used in tests
```

```text
evaluate(instrument, t, regressors, params)
  1. condition:  model at DATA times  → residuals, variances
  2. predict:    model at REQUESTED times  (a separate object, never the fitted model)
  3. noise:      noise.predict(residuals, variances, params, X_new)
  4. summarize:  median / quantiles / samples, in one place, separate from 1–3
```

**Key points:**

- **No more writing to `self`.** Prediction uses its own objects built for the requested times. This one change removes most of the failures in section 2.2.
- **LC and RV share all of this code,** so the differences in section 2.2 can't return.
- **A new kernel becomes one class in one file,** tested automatically against the dense calculation. Today, it needs edits in at least three places in `gaussian_process`.
- **The existing `evaluate()` signature and return tuples stay.** They wrap the new core. A new, structured result is added alongside them for people who want it (#40, #56).
- **Jitter handling is unified only if the likelihoods stay within tolerance.** The frozen tests decide.

**Done when:** every section 2.2 `xfail` is gone, frozen values are unchanged, `evaluate_model` is under about 100 lines, and adding a kernel touches one file.

#### Prompt 2A: extract kernels and noise models

```text
Read claude_refactoring.md, sections 2.2, 2.4, 3 and "Priority 2". The tests from Priority 1 must already exist and pass.
If they don't, stop and tell me.

Category A refactor. No numerical changes; do not touch tests/golden/data.
1. Create juliet/noise/kernels.py with one small class per kernel in gaussian_process: the mapping from juliet
   parameters to backend parameters, the backend construction, and a closed-form value(tau, params). Keep every
   existing convention exactly as it is, including the celerite amplitude and timescale behavior and the phantom
   parameter for Matern*Exp. Document each convention in the class docstring, and point to tests/golden/CONVENTIONS.md.
2. Add a registry to look up a kernel from prior names. An unknown set of GP_* names should raise an error naming the
   instrument, the names given and valid options. This is the only intended behavior change (crash to a better crash).
3. Create juliet/noise/models.py with WhiteNoise, PerInstrumentGP and GlobalGP sharing the interface in the report. Move
   the logic from get_log_likelihood and get_GP_plus_deterministic_model.
4. Keep juliet.gaussian_process importable with the same constructor, attributes and behavior, as a thin wrapper.
5. Do not remove the global/per-instrument difference in exception handling here: list it for me as a decision, with a
   recommendation.
6. Add a test that iterates over the registry and runs the dense-calculation check on every kernel, so a new kernel can't
   be added without coverage. Remove the dead CeleriteTripleSHOKernel branch only if you can show it's unreachable.

Run the entire test suite and show me the summary, and compare a likelihood timing benchmark with baseline-2.2.11 on 3
configurations. A slowdown above 5% needs my approval.
```

#### Prompt 2B: stateless evaluation

```text
Read claude_refactoring.md, sections 2.2, 2.3, 3 and "Priority 2". Prompt 2A must be merged.

Category B: rewrite model.evaluate_model so that it doesn't change the model.
1. Add a prediction context. To predict on new times/regressors, build a separate object for those times: batman/catwoman
   models, LM regressors, indexes. Never assign to self.times, self.t, self.instrument_indexes,
   self.ndatapoints_per_instrument, self.lm_arguments, self.inames, self.model[...]['m'] or noise_model.X.
2. Split the work into: conditioning at data times; deterministic model at requested times; noise.predict; and a single
   sample-summarizing function that handles scalars and arrays with the same code. Handle parameter_values as scalars by
   treating them as one sample, so that `t=` is honored.
3. Use one implementation for LC and RV, and for global and per-instrument models.
4. Keep the evaluate_model signature and the tuple return orders exactly, including the median-of-components behavior
   in #56. Add evaluate_full(), returning a dataclass with samples, median, quantiles and named components (including GP,
   LM and NLM).
5. Stop load from aliasing input dictionaries: copy on input, keeping the same public attributes.
6. The GPregressors docstring says a dict is required for global models, while the code and tutorials pass an array. Accept
   both, and fix the docstring. For global models, evaluate(inst) must return only that instrument's points, and evaluate()
   with no instrument must return the full model at all timestamps, in the order of self.t (section 6, decision 1). Document
   both, and add a test that the evaluate(inst) results for all instruments, put together, match evaluate().
7. Remove xfails as they pass. All frozen values must stay unchanged, and evaluate(inst, t=data times) must match
   evaluate(inst). Add a seeded test that runs random orders of evaluate calls and checks for identical
   results (seeded, all_samples=True) and unchanged model state.
8. Confirm #125 and #132 (compare with open PR #137, which can be closed afterwards), the t= path in #128, and #121. Do not close
   anything: draft a comment for each, for me to review. If you cannot confirm a fix for #121, say so.

Report the lines removed, any xfail still present, the tests and the performance on 3 configurations.
```

---

### Priority 3: Composable forward models

**Goal.** Adding a physical effect means writing a function and registering it, and the code reads like Eq. 1 and Eq. 5 of the paper.

**Why it matters.** Lines 3978–4515 of `generate_lc_model` contain four copies of the transit+eclipse and phase-curve logic (472 of the function's 1052 lines): log-likelihood or not × TTV or not. After removing comments and blank lines, the copies differ by one character (`1` vs `1.`). **[ran diff]** Adding a phase-curve model today means changing:

- the flag initialization and detection in `generate_datadict`
- a name-parsing block in `model.__init__`
- parameter extraction in `generate_lc_model`
- all four copies
- the long condition at `fit.py:3985`, `4115`, `4279` and `4405`

The TTV copies also read `self.model[instrument]['m'][1]` while using a separately built `m`. So TTVs combined with phase curves probably use unshifted times and stale orbital values. **[read, not run]**

**Target design:**

```text
for each instrument:
    orbit  = resolve_orbit(params, planet)         # one function: rho→a, r1r2→bp, ecc forms
    times  = shift_for_ttv(t, planet, params)      # per planet: this is the #135 fix
    signal = sum of planet signals − n + 1         # transit, eclipse, phase-curve registry
    M      = apply_dilution_and_offset(signal)     # Eq. 5
    det    = M + linear_model(...)  ± nonlinear     # Eq. 1
```

**Key points:**

- Do the planet-signal extraction as a pure refactor with the wrong multi-planet TTV behavior preserved, then fix #135 in a **separate** PR. That way, the numerical change is visible and isolated.
- Keep likelihood performance in mind. Batman objects must still be reused between calls. Measure every step.
- Open PRs #134 (depth) and #114 (spotrod) are good test cases. If they're small to rebase after this step, the design works.

**Done when:** `generate_lc_model` is under about 100 lines, the phase-curve logic exists once, a new phase-curve model takes one registered class plus tests, and frozen values are unchanged.

#### Prompt 3A: extract the signal components

```text
Read claude_refactoring.md, sections 3 and "Priority 3". Priorities 1 and 2 must be merged.

Category A refactor of generate_lc_model and generate_rv_model. No numerical changes; do not touch tests/golden/data.
Deliberately PRESERVE the current multi-planet TTV behavior for now, including its bug.
Work in small steps, each with its own commit and passing tests:
1. Add juliet/models/orbit.py: pure functions for rho→a, (r1,r2)→(b,p), each eccentricity parametrization and inclination,
   shared by LC and RV, and later by writepp in utils.py (don't change writepp yet).
2. Add juliet/models/phase_curves.py: a registry of phase-curve models (sinusoid, Cowan & Agol, Lambert, kelp homogeneous,
   kelp thermal, kelp inhomogeneous). Each declares its parameter names and how to evaluate them. Replace the 4 copies with
   one call. Before removing them, add temporary characterization for any TTV + phase-curve behavior not already frozen.
3. Add juliet/models/photometry.py to compose the signal; dilution and flux offset (Eq. 5 of Espinoza et al. 2019); the
   linear and non-linear terms (Eq. 1-2). Use a single implementation of the log_like_calc=True and False paths: the only
   difference should be whether per-planet arrays are saved.
4. Add juliet/models/radial_velocity.py and use the same linear-model code in LC and RV.
5. Keep model.generate_lc_model, generate_rv_model and the contents of self.model[...] that users read (see the docs and
   tutorials), as a thin layer.
6. Replace the 6-flag conditions at the old fit.py:3985-style sites with a registry check.
7. Reuse batman objects as now. Benchmark likelihood speed on LC transit, phase curve, multi-instrument and RV configurations
   against baseline-2.2.11 after each step. A slowdown above 5% needs my approval.

Add docs/dev/adding_a_phase_curve.md and verify it by following it, using a toy model that lives only in a test. Report any
behavior that you had to preserve even though it looks wrong.
```

#### Prompt 3B: fix multi-planet TTVs

```text
Read claude_refactoring.md, sections 2.3 and 3. Category C: science change. Separate PR. Prompt 3A must be merged.

Fix issue #135. dummy_time is overwritten per planet, so all planets use the last planet's time shifts.
1. First show the failure with a strict xfail test: two and three planets, each planet's dt only affecting its own light curve,
   and an injection test where known TTVs for both planets are recovered at fixed parameters by finding the likelihood maximum
   on a dt grid. Do this for both `dt` and `T` parametrizations.
2. Fix it so that each planet uses its own shifted times. Make sure the fix handles eclipse and phase-curve fits correctly,
   since these use the times independently, and tell me if you find anything else affected.
3. Confirm that single-planet TTV baselines are unchanged and that only multi-planet TTV baselines differ. Regenerate only those,
   with JULIET_GOLDEN_REASON pointing to #135.
4. Write the CHANGELOG entry under "Changed results": who is affected (fits with TTVs on 2+ planets), what was wrong
   and that those fits should be re-run. Draft, but do not post, a comment on #135.

Do not merge or publish without my approval.
```

---

### Priority 4: One parameter system and a validated config

**Goal.** Prior names are parsed once, by one set of rules. Before any sampling, a scientist can see exactly what juliet understood.

**Why it matters.** Almost everything in juliet is inferred from prior names, in several places:

- `load.generate_datadict` uses them to set behavior flags.
- `model.__init__` repeats almost identical parsing blocks for each parameter family.
- `gaussian_process.get_kernel_name` and `set_input_instrument` parse them again, using substring matching.

This is why shared parameters work differently for LC and RV (section 2.3), and why user errors can surface as obscure `KeyError`s, or silently (#31, #41, #54, #94, #119).

**Target:**

```python
dataset = juliet.load(...)
dataset.describe()
```
```text
Light curves:  TESS (1200 pts, quadratic LD, q1/q2), LCOGT (300 pts, linear LD)
Planets:       p1 (transit, r1/r2, rho), p2 (transit+RV, sqrt(e) parametrization)
Noise:         TESS → per-instrument GP, CeleriteMaternKernel [GP_sigma_TESS, GP_rho_TESS]
Shared:        q1_TESS_LCOGT → TESS, LCOGT
Free / fixed:  18 / 6      Warning: mflux_LCOGT prior is very broad compared with the data
```

**Key points:**

- `resolve_parameters(priors, instruments, planets)` returns an immutable `ModelConfig`. Everything else reads it instead of parsing strings.
- **New validation warns first** rather than raising, where existing scripts could currently run. Only inputs that are certain to fail should raise.
- Derived parameters (#21, #61, #115) come from the shared orbit functions in Priority 3. They would be available as samples, and `posteriors.dat` stays in its current format.

**Done when:** no module other than the parser splits parameter names, shared parameters behave the same across LC, RV and GP, and existing prior files load identically.

#### Prompt 4: parameter resolver and describe()

```text
Read claude_refactoring.md, sections 2.3, 3 and "Priority 4". Priorities 1-3 must be merged.

1. Before refactoring, add characterization tests: for many prior dictionaries and both tutorial prior files, snapshot the flags
   in lc_options/rv_options and the *_iname mappings. Include edge cases: instrument names that are substrings of each other or
   match tokens like "p1" or "rv", more than 10 GP regressors, different capitalization of distribution names, and
   shared parameters.
2. Add juliet/parameters.py: a grammar for every prior name family, with one resolve function returning a frozen ModelConfig. Report any
   existing inconsistencies you find, and preserve them (Category A).
3. Migrate generate_datadict, the iname blocks in model.__init__ and the GP name parsing to ModelConfig, one family at a time.
   lc_options, rv_options and the *_iname attributes must keep working for users. All snapshots and frozen values stay unchanged.
4. In a separate commit (Category B), make these work the same way across LC, RV and GPs: shared sigma_w and shared mu. Fix the missing
   `return` in the global likelihood at fit.py:4592 and the missing `self.` in extra_loglikelihood at fit.py:1693. Non-linear
   functions in RV are a decision for me (see the report): implement them only if I have chosen that, and warn otherwise, instead of
   silently ignoring them.
5. In a separate commit, add validation: distribution names are case-insensitive (#94); NaN/inf in the data (#41); fixed unphysical
   values (#31); period ordering, including mixed fixed and prior values (#54); light_travel_delay with t_secondary (#119); unused priors;
   and GP regressor length and order. Warn where an existing script could currently run; raise only for what would certainly fail.
6. Add dataset.describe() in the format shown in the report, plus results.derived_samples() using the shared orbit functions, and test
   them. Do not change posteriors.dat's format.

Report any change to what load accepts or rejects as a table.
```

---

### Priority 5: Sampler adapters, module split, packaging and developer docs

**Goal.** `fit.py` stops being a 5781-line file. Adding a sampler is a small, isolated change, a new contributor can find things, and installation works.

**Target layout:**

```text
juliet/
  data.py        io.py        parameters.py     results.py
  models/        orbit, photometry, radial_velocity, phase_curves
  noise/         kernels, models
  samplers/      base, dynesty, multinest, ultranest, emcee, zeus
  fit.py         thin: load, fit, model, gaussian_process stay importable
```

**Key points:**

- **Samplers.** `fit.__init__` uses `co_varnames`/`vars()` reflection to filter keywords, on 13 lines, which is how #138 silently loses options. Replace it with an explicit adapter interface and `inspect.signature`, and warn or raise on unknown keywords. Fixing keyword handling changes dynesty's behavior for anyone already passing those keywords, so it's Category C (a smaller version of the fix may need to land earlier, in Priority 1, to allow seeded runs).
- **Nautilus (#95) as the acceptance test.** If it can be added in one small file, the interface works.
- **Packaging.** Add a complete `pyproject.toml` with a realistic Python version, working extras and optional dependencies. Check #129 and #92 on a clean environment. Fix or remove `plots.py` and `__main__.py`.
- **Docs for scientists.** A "How juliet computes your model" page mapping paper equations to code, including a clear explanation of global vs per-instrument GPs. Fix the conventions table (section 2.4) and the doc errors found during this review (listed in the prompt).
- **Docs for developers.** Architecture, how to add a kernel/phase curve/sampler, and how to run tests and refresh baselines.

**Done when:** `fit.py` is under about 300 lines, all existing imports work, a clean `pip install` works on supported Python versions, and a new contributor can add a kernel by following the docs.

#### Prompt 5A: sampler adapters and module split

```text
Read claude_refactoring.md, sections 3 and "Priority 5". Priorities 1-4 must be merged.

1. Add juliet/samplers/ with a small adapter interface: run(loglike, prior_transform or logprior, ndim, names, out_folder, **options),
   returning a standard result (samples, weights, lnZ, lnZerr, raw output). Move each sampler from fit.__init__ into its own file,
   one per commit, and keep file prefixes and pickle contents the same, so that old out_folders still load.
2. Use one shared approach to multiprocessing, retaining the current user-visible behavior, and put the logic in one place so that #133
   and #87 can be handled later.
3. Replace co_varnames/vars() filtering with explicit handling based on inspect.signature. Unknown keywords should cause a clear warning. This
   changes dynesty behavior for users passing keywords that were previously dropped, so make it a separate Category C commit with a
   CHANGELOG entry. Test against both dynesty 2.x and 3.x, including that `rstate` makes runs exactly reproducible. If reproducibility
   still fails, find out why, and report it.
4. Verify the design by writing a Nautilus adapter (#95) on a separate branch. Report how many lines and files it took.
5. Split the remaining code in fit.py into data.py, results.py and io.py. juliet.load, juliet.fit, juliet.model, juliet.gaussian_process and
   all juliet.utils names must stay importable and pickles must still load, including old pickles referencing old module paths. Add a test of
   the public import surface.

Do not change the default sampler (#130) without my approval: give me a recommendation including how it affects existing users.
```

#### Prompt 5B: packaging and documentation

```text
Read claude_refactoring.md, sections 2.3, 2.4 and "Priority 5". The code must not change numerically in this work.

1. Complete pyproject.toml and remove setup.py when you can prove equivalence: dynamic version from juliet/_version.py, correct
   python_requires (propose a minimum based on what CI actually passes), core dependencies, and working extras (the current
   `extras_requires` is a typo) for multinest, kelp, catwoman, plotting, dev and docs. Test a clean install in new virtual environments
   for each supported Python, including radvel (#129). Remove or repair juliet/plots.py and __main__.py, telling me which and why.
2. Add docs/user/how_juliet_works.rst for scientists: the call flow, Eq. 1-7 of Espinoza et al. 2019 mapped to functions, a clear
   explanation with a small figure of global vs per-instrument GPs including when each is appropriate, and how prior names control the
   model. Use dataset.describe() in examples.
3. Add docs/user/gp_conventions.rst from tests/golden/CONVENTIONS.md: the true meaning and units of every kernel parameter, as tested. Link
   to it from the kernel tables. Do this for #136.
4. Add docs/dev/: architecture.md, adding_a_kernel.md, adding_a_phase_curve.md, adding_a_sampler.md, testing.md (including the A/B/C rules and how
   to refresh baselines), and CONTRIBUTING.md at the root. Verify each how-to by following it literally.
5. Fix these known documentation errors: swapped `lc`/`rv` in the GP_regressors_lc and GP_regressors_rv descriptions; evaluate vs evaluate_model and
   GPRegressors vs GPregressors; deprecated use_dynesty/dynesty_nthreads in the multithreading tutorial; mislabeled components['keplerian'] in
   rvfits; q1_p1 vs q1_TESS in quicktest; syntax errors in gps.rst; the "201X" citation; the personal path in installation; capitalized
   distribution names (#94); missing documentation for eclipse fits and extra samplers (#77, #78). Check that each error still exists before
   you change it.
6. Make the tutorial code run as part of the docs build or a scheduled CI job, so they can't silently go out of date.

Build the docs with warnings shown, and report remaining warnings.
```

---

## 5. Sequencing

```text
1A → 1B → 1C → 2A → 2B → 3A → 3B → 4 → 5A → 5B
```

- **Priority 1 is built, and the safety-net proof in 1B passes.** Review and merge it, and let CI run once, before starting 2A.
- **The kernel fixes are in their own commit** (category C), `8399ef6`, on top of the test suite, so its diff shows the 8 changed baselines.
- **The first release checkpoint is after 2B.** It is entirely bug fixes from the user's perspective, and it fixes the most visible problems.
- **Consider releasing 2.2.11 soon, with the kernel fixes and the TTV fix (3B).** All are correctness problems that affect users now. TTVs could be fixed directly at `fit.py:3649` before any of the refactoring, and then carried through 3A.
- **Each prompt is one PR or a small series.** Don't try to run them as a single session.

---

## 6. Decisions for you

### Decided

1. **Global `evaluate` shape.** `evaluate(inst)` returns only that instrument's points. A plain `evaluate()` evaluates the full model at all timestamps, because the model is global. The contract tests expect both, as strict xfails until Prompt 2B implements them.
2. **Kernel fixes.** `GP_Gamma` and the double SHO `omega2` are fixed, with docs and `CHANGELOG` (section 2.3).

### Still open

These change results or user-visible behavior, so I've left them for you. My recommendation comes first in each case.

1. **Multi-planet TTV fix: release soon, with a clear notice.** It affects published or in-progress analyses with TTVs on two or more planets, so you may want to do more than a CHANGELOG entry. The reporter of #135 should be able to confirm the fix.
2. **`GP_sigma`, `GP_timescale` and `GP_alpha` conventions: keep them and document them.** Changing them would invalidate existing prior files. If you want the documented behavior, add new names and deprecate the old ones in a 3.0.
3. **RV `non_linear_functions`: implement it.** Silently ignoring it is the worst option, and a warning is the minimum.
4. **Likelihood errors in global GPs: return `-inf`, as per-instrument does, but catch specific exceptions and count them.** Currently one numerical failure ends a global run, while per-instrument hides all errors behind a bare `except`.
5. **Minimum Python version and default sampler (#130): choose the Python minimum from what CI passes, and move to dynesty as the default in a major version.** The default doesn't currently matter for users without MultiNest, because juliet already falls back to dynesty.
6. **`phaseoffset` for eclipse-only models: make it work.** Failing that, raise an error, as silently ignoring a fitted parameter is the worst option.
7. **`evaluate_beta`, `get_quantiles` and `bin_data`: fix them, each as a small category C change.** All three are plain bugs, and `get_quantiles` touches reported medians only slightly.
8. **Hidden white noise in global george GPs: document it now, and remove it in Priority 2 if you approve the change.** Removing it shifts global photometric likelihoods slightly.

### K2-32 and the paper's Table 10

**Today's K2-32 tutorial may not reproduce Table 10, whose GP values very likely used different conventions.** The table reports the exponential time scale as `T = 16647 d`, and this code passes that parameter as a rate, already in a 2019 commit (`98da5ca`). `GP_sigma` has a similar question, since it isn't converted from ppm for this kernel.

I computed the K2-32 LC likelihood with the other parameters fixed at the table's values, varying only the GP:

| GP values | ln L |
|---|---:|
| As printed, in today's code | −7668 |
| `T` read as a time scale (`c = 1/T`), and `σ` read as ppm (10.46×10⁻⁶) | +28440 |
| Best point of a coarse grid inside today's prior file, which lands at the prior edges | +13835 |

This is strong evidence, not proof, because I ran no fit and didn't determine which code version produced the table. **Please check this** before treating Table 10's GP values as a regression target. The scientific tests leave GP hyperparameters out of the paper comparison for this reason, but `test_against_paper` for K2-32 may still fail.

---

## 7. What this review didn't cover

- **Not run:** kelp, starry and zeus. catwoman, MultiNest, UltraNest, emcee and dynesty ran only on small problems, and none of the paper's fits has run.
- **GP checks:** the dense comparison now covers all 10 kernels, but the evaluation matrix uses the Matern kernel only. Other kernels may behave differently under `evaluate`.
- **Kernel fixes:** tested at fixed parameters only. I didn't rerun any real fit using these two kernels, or check how the fixes change a posterior.
- **Linux:** every measurement so far was made on one macOS machine.
- **Reading depth:** I read the model and GP classes closely, and `load` and the sampler code in `fit.__init__` less completely. I didn't read `KeplerOrbit.py` or the kelp functions in detail. I didn't fetch the `jointfits` and `ttvs` documentation pages.
- **Paper values:** `tests/scientific/targets.py` takes them from the PDF's Tables 8 and 10.
- **Issues:** open issues were read as excerpts plus selected comments. The API returned about 100 closed issues, which I scanned by title only. #121 belongs to the same family of evaluation bugs, but I haven't determined its specific cause.
- **Time estimates:** I've given none, since the main unknown is how many hidden behaviors the frozen tests expose.
- **Probe scripts:** these were kept outside the repository, in a temporary session directory. The permanent tests replace them, except for the mutation and noise scripts, which aren't in the repository, and the K2-32 likelihood comparison above.
