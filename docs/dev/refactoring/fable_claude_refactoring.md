# juliet refactoring plan (Fable review)

Written 2026-11-24 against `dev` at `986724e` (juliet 2.2.11 in preparation) and the unpushed branch `tests/framework` (`69f2634`, `8399ef6`). It reviews `claude_refactoring.md` (the "Opus plan"), proposes changes, and gives a step-by-step plan that another model can execute one step at a time. No code was changed to write it.

Evidence labels used below: **[ran]** I executed it here, **[read]** from reading the code, **[issue]** from GitHub, **[Opus]** taken from the Opus plan and not re-verified.

---

## 0. Summary

**The Opus plan is a good diagnosis and a good test design, and I would keep most of it.** Its test suite exists and works: `pytest` gave 442 passed, 69 strict xfails and 23 skipped in 17 s here **[ran]**. Its plan for the refactor itself needs five changes:

1. **Fix first, then refactor.** The Opus plan spreads the bug fixes through the refactor: multi-planet TTVs wait for its step 3B, dynesty keywords for 5A, the missing `return` for step 4. I move every fix that can be made surgically into one legacy release, 2.2.11, before any restructuring. That is the order you asked for, and it gives users the fixes without exposing them to refactoring risk.
2. **Make the scientific validation real before touching anything.** The scientific layer has never run and has no stored baseline. It covers two systems, and none of the datasets in the docs. I add reference fits on the code users run today, and a deterministic "posterior replay" test that brings those real datasets into the fast suite.
3. **Reorder the refactor by dependency.** The Opus plan rewrites evaluation (its 2B) while `generate_lc_model` is still a 1052-line method reading everything from `self`. I refactor the forward models into pure functions first. Stateless evaluation then becomes a small step instead of the riskiest one.
4. **Add the seam that astrometry needs.** The Opus design keeps the light-curve/RV dichotomy. A third data type would mean a third copy of everything. I add a "data kind" abstraction, a shared orbit layer and a likelihood built from terms, so astrometry is a plug-in.
5. **Plan the plotting suite.** It isn't in the Opus plan. It depends on structured evaluation results and derived parameters, so I schedule those earlier and add a plotting phase.

**Full code review (2026-11-24, second pass).** I have now read every line of `juliet/fit.py`, `juliet/utils.py` and the packaging files, and probed what looked wrong. The architecture and the order of the phases stand. The review adds eleven defects that neither plan had (section 2.6), six of which return wrong results without any error, and it changes twelve steps and adds one, 1.7b (listed at the end of section 2.6).

**Fastest route to astrometry.** Gaia DR4 is weeks away and the full refactor is not. juliet already has an `extra_loglikelihood` hook that receives every sampled parameter. Once its `NameError` is fixed (step 1.4), an astrometric likelihood written as a standalone module can be mixed with transits and RVs on the legacy code. Step 9.0 does this, and step 9.2 later promotes the same code to a first-class data type.

**Confirmed by the maintainer (2026-11-24).** Existing usage must keep working: every existing script, prior file, input folder and output folder keeps working, and anything new is added alongside. The fixes ship as 2.2.11 from a `dev-2.2.11` branch, the refactor is 3.0 on its own branch cut from 2.2.11, and the reference fits run on the maintainer's server and come back as lightweight bundles (sections 4 and 5.5).

---

## 1. What must not break

juliet (Espinoza, Kossakowski & Brahm 2019) fits transits and RVs jointly and compares models with Bayesian evidences. Users drive it through prior names (`r1_p1`, `q1_TESS`, `GP_sigma_HARPS`) and a small set of calls. That surface is the compatibility contract:

| Surface | Examples | How it is protected |
|---|---|---|
| Calls and signatures | `juliet.load(...)`, `dataset.fit(...)`, `results.lc.evaluate(...)`, `juliet.utils.*` | Signature snapshot (exists) |
| Prior names and prior files | Every name family, `priors.dat`, mixed case distributions | Parser snapshots (step 6.1) |
| Input and output folders | `lc.dat`, `rvs.dat`, `GP_*_regressors.dat`, `posteriors.dat`, `*posteriors.pkl` | Persistence tests (exist), round trip (step 0.3) |
| Attributes that tutorials read | `dataset.times_lc`, `results.posteriors['posterior_samples']`, `results.lc.model[inst]['deterministic']` | Attribute snapshot and tutorial replay (step 0.3) |
| Numbers | Likelihoods, models, GP predictions, posteriors, evidences | Golden baselines (exist), scientific suite (steps 0.4, 0.5) |
| Running in parallel | `nthreads=`, MPI with MultiNest | Pickling and pool tests (step 0.3) |

Two facts about the current code make the contract easier to keep than it looks **[read]**:

- **The result pickle holds a plain dictionary,** not juliet objects. Moving classes between modules can't break old `posteriors.pkl` files. The raw dynesty and UltraNest objects stored inside it depend on those packages' versions, which is outside juliet's control.
- **`results` is the `fit` object itself.** `results.lc` and `results.rv` are `model` objects. Any new results API has to be reachable from those same objects.

---

## 2. Review of the Opus plan

### 2.1 What I checked

| Claim in the Opus plan | Check | Result |
|---|---|---|
| The suite passes in about 21 s | Ran `pytest` in the `transitspectroscopy` conda environment | 442 passed, 69 xfailed, 23 skipped, 17 s. The 6 extra skips are catwoman, which isn't installed in this environment. |
| 132 baselines | Counted configurations and files | 136 configurations, 132 with a stored baseline. The other 4 are kelp placeholders. |
| Baseline environment is pinned | Compared `pip list` with `tests/requirements-pinned.txt` | Not reproducible as recorded: catwoman is absent here, and celerite reports version `0.0.0` |
| The same, in a clean environment (step 0.2) | Built a fresh Python 3.12 virtual environment from the pinned file alone | Every stored baseline reproduces: 453 passed, 71 xfailed, 18 skipped. One package was missing from the file, `setuptools` (now added). The `0.0.0` was a quirk of the conda environment, not of the pins. |
| `evaluate_model` rewrites model state | Read the function | Confirmed. It rebuilds batman objects, narrows `self.inames`, and swaps times and indexes back and forth. |
| The result pickle and prior parsing | Read `fit.__init__`, `readpriors`, `generate_datadict` | See 2.4 for what I found beyond the Opus plan |
| The whole package | Second pass: read all of `fit.py` (5782 lines) and `utils.py` (1265), and ran probes in a scratch directory | See 2.6 |
| Open issues | Read the 38 open issues and PRs (the first 1100 characters of each, without comments), and the titles of the closed ones | The Opus themes hold. Section 2.5 adds the pitfalls I'd draw from them. |

I did not re-run the Opus probes behind its **[ran]** findings. I treat them as correct, because the strict xfails in the suite encode them and those xfails behave as recorded.

### 2.2 What to keep unchanged

- **The A/B/C rule:** every PR is a pure refactor, a crash fix, or a science change, and never two at once.
- **Known bugs as strict xfails,** so a fix can't land unnoticed.
- **The dense-matrix check of every GP kernel.** It found two real kernel bugs.
- **"Conventions that differ from the docs are frozen, not fixed"** (`GP_sigma`, `GP_timescale`, `GP_alpha`).
- **Measured tolerances and deliberate-break testing** of the safety net.
- **The acceptance tests it proposes:** Nautilus (#95) for the sampler interface, and PRs #134 and #114 for the forward-model interface.
- **Its list of open decisions** (its section 6). My recommendations match on all eight.

### 2.3 What I would change

| # | Opus plan | Problem | Change |
|---|---|---|---|
| 1 | Bug fixes are interleaved with refactor steps (TTVs at 3B, dynesty at 5A, likelihood crashes at 4). | Users wait for fixes behind unrelated restructuring, and each fix lands on freshly moved code. | Phase 1: all surgical fixes on legacy code, released as 2.2.11. |
| 2 | The scientific layer is two K2 systems, never run, with no stored baseline. Its thresholds (2σ, 1σ) are guesses. | The "before" half of before/after doesn't exist. | Steps 0.4 and 0.5: nine real-data fits from the paper and the docs, synthetic truth-recovery fits, thresholds calibrated from repeat runs, references captured on the code users run today. |
| 3 | Stateless evaluation (2B) comes before the forward-model refactor (3A). | 2B must build models at new times while `generate_lc_model` still reads times, batman objects and regressors from `self`. It either hides a rewrite of 3A or adds a workaround that 3A discards. It is the largest and riskiest step, in one prompt. | Phases 4 then 5: pure forward models first. The cheap evaluation crashes are patched in step 1.5 so users aren't kept waiting. |
| 4 | New kernel and signal classes (2A, 3A) are written against the legacy flag dictionaries, and the single parameter system arrives in step 4. | Components get written twice: once against string-built names, once against `ModelConfig`. | Step 2.2: a frozen `ModelSpec` built from the legacy dictionaries, which every new component consumes. Phase 6 only replaces how it is built. |
| 5 | The design is "LC and RV share code". | Nothing lets a third data type join. Astrometry would add `t_astrometry`, a third `model` branch and a third evaluation path. | Section 3: data kinds, likelihood terms and a shared orbit layer. |
| 6 | Plotting: "fix or remove `plots.py`". | No plotting suite, and its prerequisites (component samples for #40 and #56, derived parameters for #21, #61 and #115) come late. | Step 5.3 delivers the prerequisites. Phase 8 builds the suite. |
| 7 | Release plan: a checkpoint after 2B, CI on pushes to `master` only. | No maintenance branch and no way to try the refactor without it being the default `pip install`. | Section 4: 2.2.x maintenance branch, 3.0 pre-releases, CI on `dev`. |
| 8 | Tutorials are replayed in the last step (5B, item 6). | The tutorials are the clearest statement of what users do. They should guard the whole refactor. | Step 0.3. |
| 9 | Prompts are large. 1A to 1C ran as one session and one commit, against the plan's own rule. | Steps like 2B, 3A and 4 are too big to review. | About 35 smaller steps, each one PR. |

### 2.4 What I found that the Opus plan doesn't mention

| Finding | Evidence |
|---|---|
| **Distribution names are lower-cased in some places and not in others.** `set_prior_transform` compares the raw string with `'jeffreys'` and `'loguniform'`. `fit.__init__` compares the raw string with `'fixed'` when building `paramnames`, in the emcee and zeus starting points, and when mapping samples back to names. `model.set_posterior_samples` and `evaluate_model` do the same. | **[read]**, 9 sites. A dictionary prior with `'Fixed'` or `'LogUniform'` is likely to be mishandled. Prior files are safe, because `readpriors` lower-cases them. This is a second cause behind #94. Step 1.6 tests it before fixing. |
| **The likelihood must stay picklable.** `nthreads` hands `self.loglike` to a `multiprocessing` pool, so the whole `fit` object, with its batman and GP objects, is pickled. | **[read]**. No test covers it, and #87 and #105 show it has broken before. Step 0.3. |
| **`extra_loglikelihood` is a ready-made extension point.** Parameters that no light-curve or RV model uses are still sampled and passed to it. | **[read]**. It currently raises `NameError` **[Opus]**. Step 9.0 uses it for astrometry. |
| **`convert_time` only converts to UTC.** Any other target scale returns the input unchanged. | **[read]**. It matters for astrometry, where Gaia epochs are barycentric TCB. |
| ~~No `baseline-2.2.11` tag exists~~ **Corrected in step 0.1:** both tags exist. `v.2.2.10` is on `01836b5` (also on `origin`), and `baseline-2.2.11` is a local tag on `986724e`. My first check read a list sorted as text, where `v.2.2.10` sorts before `v.2.2.5`. | **[ran]** `git tag --sort=creatordate` |
| **Golden tests don't go through `fit.loglike`.** They call `model.generate` and `get_log_likelihood` directly. | **[read]**, from the test layout. The unit-cube to likelihood path, with its `-1e101` returns and period-ordering check, is covered only by the slow sampler tests. Step 0.3. |

### 2.5 Pitfalls from the issue tracker

These recur across the tracker and should shape the design, not only the bug list.

- **Evaluation is where users get hurt most.** Thirteen issues over six years (#13, #33, #42, #43, #46, #51, #57, #121, #125, #128, #132, and the component issues #40 and #56) are all "I fitted fine, then `evaluate` failed or returned something I didn't expect". Each fix was local, and the next combination broke.
- **Silent wrong answers are worse than crashes.** #135 (TTVs), #138 (dynesty keywords) and #54 (period order ending a run after one iteration) all ran without an error. New code should warn loudly when a parameter or keyword has no effect.
- **Everything is inferred from names, and names collide.** #2 and #29 are instrument names clashing with tokens. #116, #30 and #45 look like parametrization confusion, judging by their titles. A `describe()` that prints what juliet understood would have answered most of them.
- **Each new feature multiplies code paths.** TTVs (#81, #82, #97, #110, #111), eclipses (#44, #73) and phase curves (#131) each needed follow-up fixes, because every feature touches four copies of the light-curve code plus `evaluate_model`.
- **Samplers and parallelism break with upstream releases** (#71, #74, #105, #107, #109, #138). CI needs a job on the latest dependencies. The Opus plan has one, and it must be watched.
- **Users want derived quantities and plots** (#21, #27, #61, #115). These are cheap once the orbit functions and structured results exist.
- **GP hyperparameters shared across data types** (#35) is the same need astrometry has: one physical quantity constrained by different kinds of data.

### 2.6 Findings from the full code review

Each row was checked against the Opus plan, this plan and the test suite, and none of them covers it. **[ran]** means a probe reproduced it on the current branch, and **[read]** means it follows from the code.

**Wrong results with no error:**

| Finding | Evidence | Step |
|---|---|---|
| **A fit without `out_folder` reads and writes its results in the current directory.** `fit.__init__` sets `out_folder` to `os.getcwd()` and loads `<prefix>posteriors.pkl` if one is there. A second fit of different data in the same directory returns the first fit's posterior. | **[ran]**: two RV datasets, the second returned the first's samples and printed only "Detected dynesty sampler output files". 18 of the 27 `juliet.load(` calls in the docs have no `out_folder`. In the code since 2020. | 1.7b |
| **The same happens inside an `out_folder`:** `load.save()` never overwrites `lc.dat`, `rvs.dat` or `priors.dat`, and `fit` never overwrites the pickle or `posteriors.dat`. Changing the priors and rerunning into the same folder returns the old fit. | **[read]** `fit.py:1086–1140`, `1924`, `2483` | 1.7b |
| **`evaluate()` returns zeros when the posterior has fewer than `nsamples` (default 1000) samples.** The output array is allocated with `nsamples` rows, only the available samples are filled, and the median is taken over all rows. | **[ran]**: 400 emcee samples gave an all-zero model and a 68% band 70 times too wide. `all_samples=True` is correct, and the Opus matrix always uses it, which is why it was missed. | 1.5 |
| **`inc_p1` in `posteriors.dat` is wrong when one of `ecc`, `omega` is free and the other fixed.** The fixed or free `omega` is used in degrees inside `np.sin`. | **[ran]**: true 87.62°, written 88.28°. The branch was added in 2.2.8. | 1.6 |
| **An instrument name containing `_` silently loses its transit.** Name matching splits on `_`, so `q1_LCO_i` never matches instrument `LCO_i`. | **[ran]**: `TransitFit` is `False` and no error is raised at `load`. Issue #2 (2019) is closed but the behaviour remains. | 1.7 |
| **The `Exponential` prior is a Gamma distribution.** `transform_exponential` calls `gamma.ppf(x, a)`: shape `a`, scale 1. It is exponential only for `a = 1`. It also returns a one-element array, which numpy is deprecating as a scalar. | **[ran]**: hyperparameter 2 gives a median of 1.68, where an exponential of rate 2 gives 0.35. | 1.6, D12 |

**Crashes:**

| Finding | Evidence | Step |
|---|---|---|
| **Capitalised distribution names in a prior dictionary crash** (confirms 2.4). `'Fixed'` gives `IndexError`, `'LogUniform'` gives `KeyError`. | **[ran]** | 1.6 |
| **A blank line in a prior file crashes `readpriors`** with "not enough values to unpack". | **[ran]** | 1.6 |
| **`ld_laws='TESS-S1-quadratic,...'` crashes** for instrument names containing `-`, because the string is split on `-`. | **[ran]** | 1.7 |
| **`load.save()` copies files with `os.system('cp ...')`.** It fails on Windows and on paths with spaces, without raising. | **[read]** `fit.py:1088–1131`. #133 shows there are Windows users. | 1.7 |
| **`from juliet import *` fails,** because `__all__` lists `plots`, which can't be imported. | **[ran]** | 1.7 |
| Smaller ones, to record as strict xfails in step 6.1 and fix in 6.4: `LMlceparamfile` with array inputs (`NameError`); reading an old pickle with `sigma_w_rv_*` names (`fit` has no `sigmaw_iname`); `nthreads=1` with emcee (`Pool(0)`); `r1, r2` together with catwoman (`p1` undefined); the period-order check tests `'truncated normal'`, which never matches `'truncatednormal'`. | **[read]** | 6.1, 6.4 |

**Facts that constrain the refactor:**

- **`fit.py` gets `pickle` from `from .utils import *`.** It has no `import pickle` of its own. Giving `utils` an `__all__`, or splitting it, breaks saving results. The same star import puts `juliet.pickle`, `juliet.np`, `juliet.beta`, `juliet.os` and every `utils` function in the top-level namespace. **[ran]**
- **The likelihood rebuilds batman objects on every call in two modes:** TTVs (one `init_batman` per planet with TTVs, per instrument) and `light_travel_delay` (one per planet per instrument). The times depend on the sampled parameters, so a cache keyed by the time array can't help. **[read]** `fit.py:3917–3935`, `4229–4261`.
- **The `T` parametrization writes derived `P_p1` and `t0_p1` into `posterior_samples`, and `evaluate` then reads them from there.** A user-supplied `parameter_values` without them fails. The `dt` parametrization writes derived `T_p1_<inst>_<n>` samples. So `posterior_samples` already mixes sampled and derived quantities, plus `unnamed` and `loglike`. **[read]** `fit.py:2304–2362`, `3574`.
- **`posterior_samples['loglike']` is already stored** for every posterior sample, recomputed serially after sampling. The replay test needs nothing new from juliet.
- **The default sampler depends on the machine:** `multinest` when pymultinest imports, `dynesty` otherwise. A script with no `sampler=` gives different samplers on a laptop and on a server.
- **dynesty runs on 2.2.10 can't be seeded.** Because of #138 no keyword reaches dynesty, including `rstate`, and dynesty 2 draws from its own unseeded generator.
- **The model description is spread over `load` and `fit`:** `pl`, `pu`, `ecclim`, `ta`, `light_travel_delay`, `stellar_radius`, `stellar_teff` and the seven kelp options are `fit` arguments. `matern_eps`, `george_hodlr`, supersampling, `ld_laws`, `non_linear_functions` and `extra_loglikelihood` are `load` arguments. The last two hold user callables, and neither is documented outside the CHANGELOG.
- **`mag_to_flux`, `lctimedef` and `rvtimedef` are accepted by `load` and never used.** **[read]**
- **The astrometry bridge has three limits** that step 9.0 must state: `load` refuses to run without photometry or RVs; a planet seen only by astrometry (no `K_pN`, `r1_pN` or `b_pN`) isn't counted, so the period-order check ignores it; and the extra likelihood must be picklable for `nthreads`, so it must be a module-level class and not a closure.

**What this changes in the plan:**

| Step | Change |
|---|---|
| 0.3 | Tests for the rows above as strict xfails. Every test and tutorial replay runs in a temporary working directory with an explicit `out_folder`. The round-trip test compares to the ten decimals that `save_data` writes. |
| 0.5a | The harness passes `sampler=` explicitly, always sets `out_folder`, evaluates with `all_samples=True`, reads `loglike` from the pickle, and records that 2.2.10 dynesty runs are unseeded. |
| 0.6 | Two more benchmark configurations: TTVs and light-travel delay. |
| 1.5 | New item: the `nsamples` fix. |
| 1.6 | New items: `inc` in `posteriors.dat`, blank lines in prior files, the `Exponential` prior (decision D12). |
| 1.7 | New items: file copies without the shell, warnings for instrument names with `_` or `-`, an explicit `import pickle`. |
| 1.7b | New step: guard against stale results (decision D11). |
| 2.2 | The spec also carries the `fit`-time options, the kelp options and the user callables. |
| 4.4 | The cache rule excludes the TTV and light-travel paths. |
| 5.1, 5.3 | The predictor reproduces the derived `P`, `t0` of the `T` parametrization. `derived_samples()` includes the quantities `fit` already derives. |
| 9.0 | The three limits of the bridge. |

---

## 3. Target architecture

The principle is a **strangler refactor**: `juliet.load`, `juliet.fit`, `juliet.model`, `juliet.gaussian_process` and `juliet.utils` stay as the public face, with the same attributes. Their bodies move, piece by piece, into small modules that the old classes call.

```text
juliet/
  __init__.py     load, fit, model, gaussian_process, utils: unchanged public names
  fit.py          thin legacy classes that delegate (shrinks over time)
  spec.py         ModelSpec: frozen description of the model (what was parsed)
  parameters.py   prior-name grammar → ModelSpec; priors and transforms
  orbits.py       parametrizations and Kepler geometry, shared by every data kind
  data.py         InstrumentData, DataKind registry
  models/         photometry.py, radial_velocity.py, phase_curves.py, astrometry.py
  noise/          kernels.py, models.py
  likelihood.py   Likelihood = sum of terms; picklable
  predict.py      stateless evaluation
  results.py      ModelEvaluation, derived parameters
  samplers/       one adapter per sampler
  io.py           reading and writing files and pickles
  plots/          data.py (arrays), draw.py (matplotlib)
```

### 3.1 The five ideas

**1. `ModelSpec` is the only description of the model.** It is a frozen dataclass tree: data kinds, instruments, planets, which signal components each instrument has, which sampled name feeds each physical quantity (so `q1_TESS_LCOGT` is resolved once), the noise model and its bindings, and the global options (`pl`, `pu`, `ecclim`, `ta`, light-travel delay). No other module splits a parameter name. It is first built from today's `lc_options`, `rv_options` and `*_iname` dictionaries (step 2.2) and later directly from the grammar (step 6.2).

**2. Forward models are pure functions of `(spec, params, times, regressors, cache)`.** They never read times from an object. The cache holds batman and catwoman objects keyed by instrument and time array, so the likelihood path keeps today's speed. This is what makes evaluation at new times trivial.

**3. A data kind is a plug-in.** Photometry and RVs become two entries in a registry. Each declares its data columns, its parameter families, its forward model, the unit of its jitter and its default noise model. `fit.loglike` becomes a sum over registered terms plus `extra_loglikelihood`. Astrometry is a third entry.

**4. Orbits are shared.** `orbits.py` owns every conversion (`r1,r2 → b,p`; `rho → a`; the three eccentricity forms; inclination; `t0 ↔` periastron time). Photometry, RVs, `writepp`, derived parameters and astrometry all call it. One planet has one set of orbital elements, however many data kinds see it.

**5. Evaluation returns a structure, and plotting consumes only that.** `evaluate_full()` returns a `ModelEvaluation`: median, quantiles, optional samples, and named components (each planet, GP, linear model, non-linear model, trend). The existing `evaluate()` tuples are views of it. Plot functions take a `ModelEvaluation` and never touch model internals.

### 3.2 What astrometry needs from this design

These are design constraints to respect from phase 2 on, so astrometry doesn't force a second refactor.

| Need | Consequence |
|---|---|
| **Data with extra columns per point.** Gaia epoch astrometry is a 1-D along-scan measurement with a scan angle and a parallax factor per epoch. Relative astrometry has two coordinates and a correlation. | `InstrumentData` carries named auxiliary columns, not only `t, y, yerr`. |
| **System-level parameters** that belong to no planet and no instrument: parallax, proper motion, position offsets. | The grammar has a "system" family. Today only `rho`, `rv_slope` and similar behave this way. |
| **Orbital elements that transits and RVs don't need:** longitude of the ascending node, inclination over 0–180°, photocentre semi-major axis. | The orbit layer accepts them, and detects conflicts such as `b_p1` together with an inclination. |
| **Couplings between data kinds:** `K`, the astrometric semi-major axis, parallax and inclination are tied by geometry. | Decide whether juliet samples them independently or through masses (decision D4). The spec must be able to express a derived binding. |
| **A pinned ω convention.** juliet passes the same `omega_p1` to batman and radvel. Astrometry makes the orientation observable. | Step 2.1 adds a test that pins the convention on both paths before anything depends on it. |
| **Time scales.** Gaia epochs are barycentric TCB. | `convert_time` must really convert (see 2.4). |
| **A name for the node.** `omega_p1` exists, and `Omega_p1` differs only by case. | Choose a distinct name (decision D4). |

### 3.3 What the plotting suite needs

- **Structured components with samples** (step 5.3), so a phased plot of planet 2 can subtract everything else with correct uncertainties (#40, #56).
- **Derived-parameter samples** (step 5.3) for corner plots and tables (#21, #61, #115).
- **Compute separated from drawing:** `juliet.plots.data` returns arrays (phased, binned, residuals, O−C), and `juliet.plots.draw` only draws. The arrays can be tested exactly, and the drawing only needs smoke tests.
- **matplotlib stays optional.** Importing juliet must not import it.

---

## 4. Branching and releases

**Current state.** `origin/master` is 2.2.10 (`01836b5`). `origin/dev` is one commit ahead (`986724e`). `tests/framework` is `dev` plus two unpushed commits, so it branches from `dev` cleanly. Your local `master` is 89 commits behind `origin/master` and should be fast-forwarded.

```text
master       2.2.10 ───────────────────────────── 2.2.11 ──────────────────────── 3.0.0
                                                    ↑ (3) tag v2.2.11                ↑
dev          986724e ─────────────────────────── merge ── later 2.2.x fixes ──┐     │
               │                                  ↑ (2)                       │     │
dev-2.2.11     └─ phase 0 ─ phase 1 ─ tested (1) ─┘                           ↓     │
                                                                        (merged forward)
dev-3.0                               (cut from v2.2.11) ─ phases 2–7 (3.0.0a1 … rc1) ┴─ phases 8–9 (3.1)
```

**Rules:**

- **Phases 0 and 1 happen on `dev-2.2.11`,** cut from `dev` at `986724e`. One step, one branch, one PR into `dev-2.2.11`. Step branches are named `<phase>/<short-name>`, such as `p1/ttv-multiplanet`. Every PR states its category and passes CI.
- **2.2.11 moves through three gates:**
  1. `dev-2.2.11` passes the code suite and the scientific suite (step 1.8).
  2. It merges into `dev`, and the suites run again there.
  3. `dev` merges into `master`, which is tagged `v.2.2.11` and published.
- **`master` only receives releases.**
- **The refactor lives on `dev-3.0`,** cut from the `v.2.2.11` tag. Phases 2 to 9 are PRs into `dev-3.0`. It merges into `dev`, and then `master`, only for the 3.0.0 release.
- **`dev` stays the 2.2.x line until 3.0.0.** A bug found after 2.2.11 is fixed on `dev`, released from `master` as 2.2.12, and merged forward into `dev-3.0` in the same week, so the two lines never drift.
- **The refactor ships as 3.0 pre-releases** (`3.0.0a1`, and so on), tagged on `dev-3.0`, which pip installs only with `--pre`. Default installs stay on 2.2.x until the scientific suite passes on a release candidate and volunteers have tried it.
- **The public API doesn't change in 3.0.** The major version signals that the internals are new, and it gives room for the few approved result changes.
- **Golden baselines change only in category C PRs,** with the reason recorded. This rule already exists.

---

## 5. Validation strategy

Two suites answer two different questions.

| | Code validation | Scientific validation |
|---|---|---|
| Question | Did any number or behavior change? | Do fits of data with known answers still give those answers? |
| Layers | Unit, golden, contract (exist), plus the additions in step 0.3 | Posterior replay, synthetic recovery, reference fits |
| When | Every PR, in seconds | Replay on every PR. Fits at each milestone and before each release. |

### 5.1 Real datasets

All of these are already used by the paper or the docs. Three need a TESS file from MAST, which step 0.4 stores locally so tests never use the network.

| ID | System and data | Source | What it exercises |
|---|---|---|---|
| R1 | K2-140: K2 + LCOGT photometry, CORALIE + HARPS + FIES RVs | Paper Table 8, `tutorial1/` | Joint fit, supersampling, two limb-darkening laws |
| R2 | K2-32: K2 photometry with GP, HIRES + HARPS + PFS RVs | Paper Table 10, `tutorial2/` | Three planets, `rho`, celerite GP. The GP values in Table 10 are not a target (the Opus caveat stands). |
| R3 | TOI-141: TESS transit | `quicktest.rst` | Simplest transit, input and output folders |
| R4 | TOI-141: RVs with 1, 2 and 3 planets, and a linear trend | `rvfits.rst` | RV-only fits. The ordering of the evidences tests `lnZ`. |
| R5 | TOI-141: TESS + RVs | `jointfits.rst` | Joint fit, `return_components` |
| R6 | HATS-46: TESS, and LCOGT + SWOPE | `transitfits.rst`, Brahm et al. 2017 | Multi-instrument photometry, other limb-darkening laws |
| R7 | HATS-46: TESS with a Matérn GP | `gps.rst` | Per-instrument photometric GP |
| R8 | TOI-141 RVs with a GP, global and per-instrument | `gps.rst`, `GP_regressors_rv*.dat` | The global RV path that fails most in the issues |
| R9 | CHAT light curve with linear regressors; the TTV tutorial | `linearmodels.rst`, `ttvs.rst` | Linear models, TTVs on real data |

### 5.2 Synthetic datasets with known truth

One small simulated dataset per mode that no real dataset covers. Each is generated from fixed seeds by a script in the repository.

- Two-planet TTVs, in both the `dt` and `T` parametrizations
- Eclipse and phase curve, with and without light-travel delay
- Global photometric GP over two instruments
- RVs with a quadratic trend, a linear model and shared jitter
- catwoman, when installed
- One problem per sampler (dynesty, dynamic dynesty, UltraNest, emcee, MultiNest), reusing `tests/sampler`

### 5.3 Three kinds of scientific test

**Posterior replay (every PR, deterministic).** For each reference fit, the bundle (section 5.5) holds 200 posterior samples and the log-likelihood juliet gave each one. The test rebuilds the dataset and model, recomputes the log-likelihood at those samples, and requires agreement to the golden tolerance. It also stores and compares `evaluate()` output at the median. This puts real data, real prior files and realistic parameter values into the fast suite. A refactor that changes any likelihood on K2-32 fails within seconds.

**Reference fits (milestones).** Rerun the full fit and compare the posterior with the stored reference, parameter by parameter: the shift of the median in units of σ, the ratio of the widths, and `lnZ`. **The thresholds come from measurement:** step 0.5 runs each reference three times and sets the thresholds from the scatter between runs.

**Truth recovery (milestones).** For synthetic data and literature values, the truth must lie inside the posterior at a level calibrated in the same way.

### 5.4 Which code produces the references

| Reference | Code | Purpose |
|---|---|---|
| `ref-2.2.10` | `origin/master` | What users have today |
| `ref-2.2.11` | `dev-2.2.11` after phase 1 | The target that every later phase must reproduce |

The difference between the two is the measured effect of the phase 1 fixes. It should be zero outside multi-planet TTVs and the two corrected kernels, and that comparison is itself a test of phase 1.

### 5.5 The reference bundle

The fits run on the maintainer's server. What comes back must be small enough to commit, pull on any computer, and read without rerunning anything. Each fit produces one directory, `validation/references/<reference>/<dataset>/`:

| File | Content | Typical size |
|---|---|---|
| `summary.json` | Medians and 68% intervals, `lnZ` and its error, sampler and settings, seed, run time, juliet commit, the version of every dependency, and checksums of the input data | 10 kB |
| `posteriors.dat` | The file juliet itself writes, unchanged | 5 kB |
| `priors.dat` | The prior file used | 2 kB |
| `samples.npz` | 1000 equally weighted posterior samples of every sampled parameter, in float64, with a `names` array | 100–300 kB |
| `replay.npz` | 200 of those samples with the log-likelihood juliet gave each, and `evaluate()` output at the median | 50 kB |
| `model.npz` | Per instrument: the median model and its 68% band at the data times and on a dense grid, the components (per planet, linear model, GP) and the residuals | 100–300 kB |
| `plots/*.png` | Per instrument: data, model and residuals against time; the phase-folded view per planet; one corner plot of the physical parameters. 100 dpi. | 50–150 kB each |

- **Budget (decision D2):** 1 MB per fit, 15 MB per reference, and 100 MB for all references together. The collection script fails if a bundle exceeds the first two, and prints the running total of `validation/references/`.
- **Write once.** A committed bundle is never regenerated in place: a rerun gets a new reference name. git keeps every version of a binary file, so rewriting bundles is what would make the repository grow.
- **Left on the server:** `posteriors.pkl`, the sampler's own state files and MultiNest's output. The script writes them to a scratch directory outside the repository. `summary.json` records that path so a full posterior can be found later.
- **Repeat runs** (for the thresholds) store only `summary.json`, as `<dataset>/repeat-<n>.json`.
- **The plots are for people.** No test reads a PNG. Tests read `summary.json`, `samples.npz`, `replay.npz` and `model.npz`.
- **The plots are drawn by the validation script with plain matplotlib,** because juliet has no plotting suite yet. Phase 8 replaces that code with calls to the new suite, and `model.npz` is the first draft of the plot-data layer of step 8.1.

**The round trip:**

```text
laptop:  git push origin dev-2.2.11            (harness and data, steps 0.3 to 0.5)
server:  git pull
         python validation/run_reference.py --reference ref-2.2.10 --all --repeats 3 --jobs 8
         git add validation/references/ref-2.2.10 && git commit && git push
laptop:  git pull
         pytest tests/scientific                (replay and comparisons, no fitting)
```

The script is resumable: a fit whose bundle has a complete `summary.json` is skipped, so an interrupted run continues where it stopped.

---

## 6. The steps

Each step is written to be handed to a model on its own. **Every step starts the same way: read this file's sections 3 to 5 and the step itself, create the branch from an up-to-date `dev-2.2.11` (phases 0 and 1) or `dev-3.0` (phases 2 to 9), and confirm that `pytest` passes before changing anything. Every step ends the same way: run `pytest`, report the result, the category and anything unexpected, and don't push or open a PR without the maintainer's approval.** I don't repeat these in each step.

Common rules for all steps:

- **Never change `tests/golden/data` except in a category C step,** using `--update-golden` with a reason.
- **If a test shows behavior that looks wrong, don't fix it.** Mark it strict xfail, report it, and continue.
- **Line numbers in the Opus plan refer to `986724e`.** Find code by function name.
- **Benchmark every phase 3 to 6 step** with the harness from step 0.6. A likelihood slowdown above 5% needs approval.

### Phase 0: land the safety net

#### Step 0.1: branch hygiene and splitting the existing work

- **Category:** none (repository only). **Depends on:** nothing.
- **Do:**
  1. Fast-forward local `master` to `origin/master`.
  2. Confirm the tags `v.2.2.10` (on `origin/master`) and `baseline-2.2.11` (on `986724e`, local only). Both exist, so none is created.
  3. Create `dev-2.2.11` from `dev` at `986724e`. Split `tests/framework` into two step branches off it: `p0/test-framework` holding `69f2634`, and `p1/gp-kernel-fixes` holding `8399ef6` on top of that.
  4. In `.github/workflows/tests.yml`, add `dev` and `dev-*` to the `push` branches, and make pull requests into them trigger the suite.
  5. Leave the empty `claude-refactoring` branch (local and on `origin`, identical to `dev`) alone, and ask the maintainer whether to delete it.
  6. Move `tests/test_transit_and_eclipse_ltd.py` (untracked, from 2024) to `examples/validation/` or delete it, as the maintainer chooses, and remove its entry from `collect_ignore`.
  7. Ask where the two plan files should live (decision D7).
- **Done when:** two clean branches exist, each passing `pytest`.

#### Step 0.2: reproducible environment and first CI run

- **Category:** none. **Depends on:** 0.1.
- **Do:**
  1. Create a fresh virtual environment from `tests/requirements-pinned.txt` only, and run `pytest`. Record every package that fails to install or reports a different version. Known suspects: celerite reports `0.0.0` in the current environment, and catwoman is missing from it.
  2. Correct the pinned file so that a clean install reproduces the baselines exactly. If a baseline differs in the clean environment, stop and report. Don't regenerate.
  3. After approval, push `dev-2.2.11` and `p0/test-framework`, and open the PR into `dev-2.2.11` so CI runs for the first time.
  4. Read the Linux results. If golden tests fail only by floating-point noise, measure the largest difference and propose new tolerances with the measurements. Don't loosen a tolerance without numbers.
  5. Fix the roughly 22,000 numpy deprecation warnings only if they come from test code. If they come from `juliet/`, list their sources for step 1.6.
- **Done when:** the pinned job and the Python 3.10 to 3.13 matrix are green on Linux and macOS, or each failure is explained.
- **Status:** done on 2026-11-24 (PR #139, all 9 jobs green on `e875740`). Results:
  - Linux differs from the macOS baselines in two places only: catwoman light curves (`7e-10` relative, up to `1.2e-7` in the log-likelihood) and near-zero elements of one celerite prediction (`1.5e-11` of the array's largest value, on some runners only). Tolerances were widened for those; measurements are in `tests/README.md`.
  - `six` was missing from the pinned file (ultranest 4.5.0 imports it without declaring it).
  - All the numpy deprecation warnings come from one line in a dependency, `george/metrics.py:59` (george 0.4.4), not from `juliet/` or the tests. Nothing to do in step 1.6.
  - Python 3.10 to 3.13 all pass, which answers D6's question of what CI supports.
  - Test failures now appear as annotations on the PR's checks page (`tests/conftest.py`).

#### Step 0.3: widen the compatibility contract

- **Category:** none (tests only). **Depends on:** 0.2.
- **Do**, each as its own test module under `tests/contract/`:
  1. **Tutorial replay.** Extract every Python code block from `docs/tutorials/*.rst` and `docs/user/quicktest.rst`. Run each tutorial as a test with the smallest sampler settings that finish (for example `n_live_points=50`), a temporary `out_folder`, the Agg backend, and data from step 0.4. Assert only that the calls succeed and return the documented shapes. Record blocks that fail today as strict xfails with the reason. Several doc errors are listed in the Opus prompt 5B.
  2. **Attribute surface.** After a small fit of a joint dataset, snapshot the public attribute names of the `load`, `fit` and `model` objects, the keys of `results.posteriors`, `lc_options`, `rv_options` and `results.lc.model[instrument]`, and the type and shape of each value. Store it as JSON, as `signatures.json` is.
  3. **Fit-level likelihood.** Build a `fit` object without sampling, by pointing `out_folder` at a fixture that already holds a `posteriors.pkl`. For a subset of golden configurations, check `prior_transform`, `logprior`, `loglike` and `logprob` on fixed unit-cube vectors against stored values, including vectors that hit the eccentricity limit and the period-ordering check.
  4. **Pickling and pools.** Assert that `pickle.loads(pickle.dumps(results.loglike))` gives the same likelihood, for photometry with a celerite GP, photometry with a george GP, and RVs. Add a dynesty smoke fit with `nthreads=2` to the sampler layer.
  5. **File round trip.** `load(...)` with `out_folder`, then `load(input_folder=...)`, must give the same data to the ten decimals `save_data` writes, and identical options. Report the likelihood difference that this rounding causes for each golden configuration. Cover linear regressors, GP regressors and a global GP.
  6. **Dictionary-prior case handling.** Test `'Fixed'`, `'Uniform'`, `'LogUniform'` and `'TruncatedNormal'` in a prior dictionary through `fit` with emcee and dynesty at tiny settings. Mark what fails as strict xfail (section 2.4).
  7. **Findings of section 2.6**, each as a strict xfail stating the correct behaviour: two fits of different data without `out_folder` in one directory give different posteriors; `evaluate()` equals `evaluate(all_samples=True)` for a posterior of 400 samples; `inc_p1` in `posteriors.dat` for a fixed `omega`; a prior file with a blank line; an instrument named `LCO_i`; `ld_laws` with an instrument named `TESS-S1`; `from juliet import *`. Freeze, without an xfail, what the `Exponential` prior does today.
  8. **Isolation.** An autouse fixture changes into a temporary directory for every test, and every `load` in the suite and in the tutorial replay passes an explicit `out_folder`, because of the first row of section 2.6. With the small sampler settings of item 1, call `evaluate` with `all_samples=True`.
- **Don't:** change anything in `juliet/`.
- **Done when:** all eight modules exist, and every current failure is a strict xfail with a reason.

#### Step 0.4: store the validation data

- **Category:** none. **Depends on:** 0.1.
- **Do:**
  1. Download once the three TESS light curves the docs fetch from MAST (TOI-141 for `quicktest.rst` and `jointfits.rst`, HATS-46 for `transitfits.rst` and `gps.rst`, and the target in `ttvs.rst`). Convert each to juliet's plain-text format with the same calls the tutorials use, and record the URL, the date and a checksum.
  2. Put them, with the datasets already in `docs/tutorials/`, `tutorial1/` and `tutorial2/`, under `validation/data/<dataset>/`, each with a short README giving the source and the reference paper.
  3. Add a loader, `tests/scientific/datasets.py`, that returns the `juliet.load` arguments for R1 to R9 exactly as the tutorials write them.
  4. Add scripts that generate each synthetic dataset in section 5.2 from fixed seeds, and store the true parameter values beside them.
- **Done when:** every dataset in section 5 loads offline with one call.

#### Step 0.5: capture the scientific references

This step has three parts, because the fits run on the maintainer's server and not in the session that writes the code.

**0.5a: the harness (a model writes it, on a branch off `dev-2.2.11`).** **Depends on:** 0.3, 0.4.

- **Do:**
  1. Write `validation/run_reference.py`. Arguments: `--reference <name>`, `--dataset <id>` or `--all`, `--repeats <n>`, `--jobs <n>`, `--scratch <dir>`, and `--quick`. It runs each fit with the settings fixed in one table, `validation/settings.py` (sampler, live points, threads, seed), sends juliet's `out_folder` to the scratch directory, and writes the bundle of section 5.5. `--quick` uses 50 live points, to test the harness itself.
  2. **Settings that the code review makes mandatory:** pass `sampler=` explicitly for every fit, because the default depends on whether pymultinest is installed; always pass an `out_folder` inside the scratch directory, and refuse to start if it already holds a `*posteriors.pkl`; call `evaluate` with `all_samples=True` on the thinned samples; take the per-sample log-likelihood from `posterior_samples['loglike']`. Seed numpy and pass `rstate` where the juliet version forwards it, and record in `summary.json` whether the run was seeded. dynesty runs on 2.2.10 are not (section 2.6), which is why the thresholds come from repeats.
  3. **It uses only juliet's public API as it exists in 2.2.10,** because it has to run unchanged on 2.2.10, on 2.2.11 and on every 3.0 pre-release. It imports nothing from `tests/`. It reports `juliet.__version__` and the installed commit in `summary.json`, and refuses to write into a reference whose recorded version differs.
  4. Write `validation/make_bundle.py`, called by the runner: it thins the posterior, evaluates the models, draws the plots, and enforces the size budget. It can also rebuild a bundle from a scratch directory without refitting.
  5. Write `validation/compare_references.py A B`: a Markdown table per dataset (shift of each median in σ, width ratio, difference in `lnZ`, each against its threshold), and overlay plots of the two posteriors and the two models, written to `validation/reports/A_vs_B/`.
  6. Write `validation/README.md` with the exact server commands: creating the environment from `tests/requirements-pinned.txt`, installing a given juliet version (`pip install juliet==2.2.10`, or `pip install .` from a checkout), running, and committing the bundles.
  7. Add `tests/scientific/test_replay.py`, the posterior replay test of section 5.3. It runs in the default suite and skips, with a clear message, any dataset whose bundle isn't there yet.
  8. Rewrite `tests/scientific/test_tutorials.py` to read the bundles: comparisons with the paper's tables for R1 and R2, with Brahm et al. 2017 for R6, and with the TOI-141 paper for R4 and R5. Enter each literature value from the publication itself.
  9. Run `--quick --all` locally once, to prove that every dataset fits, bundles and stays inside the budget.
- **Done when:** the quick run produces a valid bundle for every dataset and the README commands work in a fresh environment.

**0.5b: the reference runs (the maintainer, on the server).** **Cost:** hours.

- **Do:**
  1. In an environment with **juliet 2.2.10** installed, run `python validation/run_reference.py --reference ref-2.2.10 --all --repeats 3`.
  2. Commit `validation/references/ref-2.2.10/` to `dev-2.2.11` and push.
- **Done when:** every dataset has a bundle and its repeat summaries, or a recorded reason why the fit can't run on 2.2.10 (multi-planet TTVs are expected to run and give wrong answers, which is itself the reference).

**0.5c: thresholds and findings (a model, after pulling the bundles).**

- **Do:**
  1. Measure the run-to-run scatter of every median, width and `lnZ` from the repeats. Set the comparison thresholds from it, and write them with the measurements in `validation/THRESHOLDS.md`.
  2. Run the replay test and the literature comparisons on `dev-2.2.11`.
  3. Report every dataset where today's juliet disagrees with the literature or with the injected truth. Those are findings, not test failures to tune away.
- **Don't:** adjust a threshold to make a comparison pass.
- **Done when:** `ref-2.2.10` exists for every dataset, the replay test passes on `dev-2.2.11`, and the report lists each disagreement.

#### Step 0.6: benchmark harness

- **Category:** none. **Depends on:** 0.2.
- **Do:** add `benchmarks/likelihood.py`, timing 1000 likelihood calls on six configurations: a transit, a transit with supersampling, a phase curve, three photometric instruments, RVs with a celerite GP, the K2-32 joint dataset, a two-planet TTV fit, and an eclipse with `light_travel_delay`. The last two rebuild batman objects on every call (section 2.6), so they are the slowest paths and the ones a refactor can most easily slow further. It prints a table and can compare against a stored result from another commit. Store the result for `baseline-2.2.11`.
- **Done when:** repeated runs agree to within 2%, or the measured noise is reported so the 5% rule can be adjusted.

### Phase 1: fix the legacy code and release 2.2.11

Every step here is a minimal edit inside the existing functions. **No code moves and nothing is renamed.** Each fix gets a test that fails before it and passes after, and each removes its strict xfail.

#### Step 1.1: the two kernel fixes

- **Category:** C. **Depends on:** 0.2. **Status:** written (`8399ef6`).
- **Do:** open the PR from `p1/gp-kernel-fixes`. Confirm that the replay tests for R7 and R8 are unaffected, because those fits use the Matérn kernel.
- **Done when:** merged with the maintainer's approval of the CHANGELOG text.

#### Step 1.2: multi-planet TTVs (#135)

- **Category:** C. **Depends on:** 0.5.
- **Background:** in `generate_lc_model`, `dummy_time` is assigned inside the loop over planets and used for all planets in a later loop, so every planet gets the last planet's shifted times **[Opus, ran]**.
- **Do:**
  1. Add a strict xfail test with two and with three planets: changing one planet's `dt` must change only that planet's light curve. Add a second test that finds the maximum of the likelihood on a grid of `dt` and recovers the injected values for every planet. Cover both the `dt` and `T` parametrizations.
  2. Keep one shifted time array per planet (a dictionary keyed by planet number) and use each planet's own array. Check the eclipse and phase-curve branches, which use the times separately.
  3. Confirm that single-planet TTV baselines are unchanged, and regenerate only the multi-planet ones.
  4. Write the "Changed results" CHANGELOG entry, and draft a comment for #135 asking the reporter to confirm.
- **Done when:** the synthetic two-planet TTV fit recovers the truth for both planets, where `ref-2.2.10` does not.

#### Step 1.3: dynesty keywords and seeding (#138)

- **Category:** C (for users who pass keywords that were being dropped). **Depends on:** 0.3.
- **Do:**
  1. Replace the `vars(DynestySampler)['__init__'].__code__.co_varnames` lookups, in the single-thread and pool branches, with `inspect.signature` on the sampler class and on `run_nested`. It must work on dynesty 2.x and 3.x.
  2. Pass `rstate` through, and warn once for every keyword that no sampler function accepts.
  3. Test that `bound`, `sample` and `rstate` reach the sampler, and that two runs with the same seed give identical samples. If they don't, find out why and report.
  4. Make the dynesty tests in `tests/sampler` seeded.
- **Don't:** touch the other samplers' keyword handling. That belongs to step 7.1.

#### Step 1.4: crashes in the likelihood

- **Category:** B. **Depends on:** 0.3.
- **Do**, one commit each:
  1. `get_log_likelihood`: add the missing `return` for a global model without a GP.
  2. `fit.loglike`: `extra_loglikelihood['loglikelihood']` becomes `self.extra_loglikelihood[...]`. Add a test with an extra likelihood that reads a parameter no model uses, which is the pattern step 9.0 relies on.
  3. Global GP likelihood: return `-inf` when the GP solver raises, as the per-instrument path does, catching the specific linear-algebra exceptions and counting them. This is Opus decision 4 and needs approval.

#### Step 1.5: minimal evaluation fixes

- **Category:** B. **Depends on:** 0.3.
- **Do**, one commit each, **and only where the fix is a few lines.** Anything that needs restructuring stays a strict xfail for phase 5.
  1. Set `original_GPregressors` when the model is built, so `evaluate(inst)` works on a global GP before any `t=` call (#125 and the first row of the Opus matrix).
  2. Restore `noise_model.X` after a per-instrument `t=` evaluation, so a later `evaluate(inst)` works.
  3. Wrap the body of `evaluate_model` in `try/finally` so `self.inames` and the instrument times are restored after an exception.
  4. Review and merge PR #137 for #132, keeping its author's credit.
  5. Make the RV linear model work with `t=` (the rest of #128).
  6. **Category C, with approval:** when the posterior has fewer samples than `nsamples`, use the number available, so `evaluate()` no longer returns a median over zero rows (section 2.6). The CHANGELOG entry says who is affected: anyone who called `evaluate` without `all_samples=True` on a posterior of fewer than 1000 samples, mostly short emcee runs.
- **Done when:** the matching xfails are removed, and the count of the remaining ones is reported.

#### Step 1.6: small utilities and prior names

- **Category:** C for items 1 to 3 and 5, B for items 4, 6 and 7. **Depends on:** 0.3.
- **Do**, one commit each, each with approval:
  1. `evaluate_beta` tests `x`, not the hyperparameters.
  2. `get_quantiles` gives the true median for even-sized input.
  3. `bin_data` uses every point in each bin.
  4. Lower-case the distribution name once, where priors enter `load`, so the nine raw comparisons in section 2.4 behave the same for prior dictionaries as for prior files. Don't modify the user's dictionary: normalise a copy. Fix `'truncated normal'` in the period-order check in the same commit.
  5. `writepp`: convert `omega` to radians in the two branches where only one of `ecc` and `omega` is free, so `inc_pN` in `posteriors.dat` is right. Only the derived `inc` line changes.
  6. `readpriors` skips blank lines and lines of whitespace.
  7. The `Exponential` prior (decision D12, decided): keep what it computes, a Gamma distribution of shape `a` and scale 1, and say so in the docs and the docstring. Return a scalar in place of the one-element array, checking that no sampled value changes.

#### Step 1.7: packaging hotfix and input warnings

- **Category:** B. **Depends on:** 0.2.
- **Do:**
  1. `setup.py`: `extras_requires` becomes `extras_require`, and `python_requires` becomes the lowest version CI passes on (decision D6).
  1b. Add `setuptools` to `install_requires`. Found in step 0.2: in a fresh Python 3.12 environment `import juliet` fails, because batman 2.5.3 imports `distutils`, which Python 3.12 removed and only setuptools still provides. Conda environments hide this because they ship setuptools.
  2. Make `juliet/plots.py` importable on current matplotlib, or empty it. Its import of `mpl_toolkits.axes_grid` fails today.
  3. Warn, without raising, on NaN or infinite values in the data (#41), and on `light_travel_delay=True` together with a `t_secondary` prior (#119).
  4. `load.save()` copies files with `shutil.copyfile` in place of `os.system('cp ...')`. Same for the `rm -r` in `get_all_TESS_data`.
  5. Add `import pickle` to `fit.py`, so it no longer depends on the star import from `utils`.
  6. Warn at `load` when an instrument name contains `_`, or contains `-` while `ld_laws` names instruments. Both are certain to be misread today (section 2.6).
- **Don't:** add any check that could reject an input that runs today.

#### Step 1.7b: guard against stale results

- **Category:** C (decision D11, decided). **Depends on:** 0.3.
- **Background:** the first two rows of section 2.6. juliet treats any existing `<prefix>posteriors.pkl` as the answer, in the output folder or, when none is given, in the current directory.
- **Do:**
  1. From now on, store in the result pickle a fingerprint of the fit: a hash of the data arrays, the regressors, the prior dictionary and the model options. Old pickles don't have one and stay readable.
  2. When a pickle is found and its fingerprint matches, load it, with or without an `out_folder`. Reloading a finished fit behaves as today.
  3. When the fingerprint differs (decision D11, decided):
     - **explicit `out_folder`:** raise an error that names the folder, says that its results were made with different data, priors or options, and tells the user to choose a new folder or delete the old results. Never overwrite a finished fit.
     - **no `out_folder`:** ignore the pickle found in the current directory, say so in one printed line, and run the fit.
  4. When a pickle has no fingerprint (written before this version), load it as today. Warn loudly if its parameter names differ from the priors, or if no `out_folder` was given. "Same model, different data" can't be detected for these files, and the docs say so.
  5. Leave `load.save()` as it is, and include in the same error the case where `priors.dat` or the data files in the folder differ from the inputs.
- **Don't:** change the file names or the pickle's existing keys. The fingerprint is one new key.
- **Done when:** the two-fits xfail of step 0.3 passes, and loading every fixture and every reference bundle's folder still works.

#### Step 1.8: release 2.2.11

- **Category:** none. **Depends on:** 1.1 to 1.7b.
- **Do:**
  1. The maintainer runs `validation/run_reference.py --reference ref-2.2.11 --all --repeats 3` on the server with `dev-2.2.11` installed, and pushes the bundles.
  2. Run `validation/compare_references.py ref-2.2.10 ref-2.2.11`. Every difference beyond the thresholds must be explained by a phase 1 fix. Report the table and the overlay plots.
  3. Finish the CHANGELOG. Merge `dev-2.2.11` into `dev` and run the code suite and the replay test there.
  4. Merge `dev` into `master` and tag `v.2.2.11`. The maintainer publishes.
  5. Create `dev-3.0` from the `v.2.2.11` tag, and set its version to `3.0.0.dev0`.
- **Done when:** the maintainer has published, and the open issues fixed here have draft replies.

### Phase 2: foundations

#### Step 2.1: orbit and parametrization functions

- **Category:** A. **Depends on:** 1.8.
- **Do:**
  1. Create `juliet/orbits.py` with pure, vectorized functions: `(r1, r2, pl, pu) → (b, p)`; `(rho, P) → a/R*`; each eccentricity form to `(e, ω)`; `(a, b, e, ω) → inclination`; the time of transit to and from the time of periastron; and the derived quantities `writepp` computes (duration, depth and the rest).
  2. Unit test each against the paper's formulas and against the values `generate_lc_model`, `generate_rv_model` and `writepp` produce today.
  3. **Pin the ω convention.** For an eccentric orbit, record the phase of the transit and of the RV maximum that batman and radvel give for the same `omega_p1`, state in the module docstring whose argument of periastron this is, and freeze it in a test.
  4. In separate commits, make `writepp`, then `generate_rv_model`, then `generate_lc_model` call these functions. `posteriors.dat` for the fixture and all baselines must be unchanged.
- **Don't:** change constants. `G` stays as it is (#37 is a separate category C decision).

#### Step 2.2: `ModelSpec` from the legacy dictionaries

- **Category:** A. **Depends on:** 2.1.
- **Do:**
  1. Create `juliet/spec.py` with frozen dataclasses for everything in section 3.1, idea 1. Leave room for a data kind other than `lc` and `rv`, for auxiliary data columns, and for system-level parameters, without implementing them. Include everything that defines the model today, wherever the user passed it (section 2.6): the `fit` arguments (`pl`, `pu`, `ecclim`, `ta`, `light_travel_delay`, `stellar_radius`, `stellar_teff`, the kelp options) and the `load` arguments (`matern_eps`, `george_hodlr`, supersampling, limb-darkening laws, `non_linear_functions`, `extra_loglikelihood`). User callables and filter arrays appear in the JSON snapshot by name and shape only.
  2. Write `spec_from_legacy(dataset, lc_model, rv_model)`, which fills the spec from `lc_options`, `rv_options`, the `*_iname` dictionaries, the TTV tables and the GP objects. It parses no parameter name itself.
  3. Snapshot the spec, as JSON, for every golden configuration and for R1 to R9. These snapshots become the acceptance test of step 6.2.
  4. Add `dataset.describe()`, printing the spec in the format shown in the Opus plan's Priority 4. Nothing else uses the spec yet.
- **Done when:** the snapshots exist and `describe()` output has been reviewed on K2-32.

### Phase 3: noise

#### Step 3.1: kernel classes and registry

- **Category:** A. **Depends on:** 2.2.
- **Do:** as in the Opus prompt 2A, items 1, 2 and 6, with two differences. Each kernel class takes its parameter bindings from the spec, not from prior-name strings. `tests/support/kernels.py` already holds the closed forms, so move those into each class's `value()` and have the dense test iterate over the registry.
- **Don't:** change a convention. Remove `CeleriteTripleSHOKernel` only after showing it is unreachable.

#### Step 3.2: noise models

- **Category:** A. **Depends on:** 3.1.
- **Do:**
  1. Create `juliet/noise/models.py` with `WhiteNoise`, `PerInstrumentGP` and `GlobalGP`, sharing `log_likelihood(residuals, variances, params)` and `predict(residuals, variances, params, X_new)`.
  2. **`predict` never modifies the object:** it takes the conditioning inputs and the new inputs as arguments.
  3. Move the bodies of `get_log_likelihood` and `get_GP_plus_deterministic_model` into these classes, and make `juliet.gaussian_process` a wrapper with the same constructor and attributes.
  4. Keep the hidden white-noise term of global george GPs exactly as it is (Opus decision 8), and document it in the class.
- **Done when:** all baselines and the replay tests pass, and the pickling test from step 0.3 still passes.

### Phase 4: forward models

These steps first shrink `generate_lc_model` in place and then move it. Doing the extraction inside `fit.py` first keeps each diff reviewable.

#### Step 4.1: extract the parameter handling

- **Category:** A. **Depends on:** 2.2.
- **Do:** inside `fit.py`, extract into module-level functions that take the spec and the parameter dictionary and return plain values: limb-darkening coefficients for an instrument; the orbital elements of a planet as seen by an instrument, using `orbits.py`; the TTV-shifted times of a planet; and the validity checks that set `modelOK`. Replace each inline block with a call. One commit per function.

#### Step 4.2: one copy instead of four

- **Category:** A. **Depends on:** 4.1.
- **Do:**
  1. Before changing anything, add golden configurations for TTVs combined with an eclipse and with each phase-curve model, in both `log_like_calc` modes. The Opus plan suspects that these paths use unshifted times. Whatever they do today is frozen, and anything that looks wrong is reported.
  2. Merge the four copies of the transit, eclipse and phase-curve logic (likelihood or not, TTVs or not) into one function. The only difference between likelihood and evaluation mode is whether per-planet arrays are kept.
  3. If item 1 shows wrong physics, propose the fix as a separate category C step.

#### Step 4.3: signal registry

- **Category:** A. **Depends on:** 4.2.
- **Do:** create `juliet/models/phase_curves.py` with one class per model (sinusoid, Cowan & Agol, Lambert, and the three kelp models). Each declares its parameters and an `evaluate`. Replace the long flag conditions with a registry lookup driven by the spec. Verify the interface by writing a toy model that exists only in a test, and by checking how small PR #134 (depth) would become.

#### Step 4.4: move and finish photometry and RVs

- **Category:** A. **Depends on:** 4.3, 3.2.
- **Do:**
  1. Create `juliet/models/photometry.py` and `juliet/models/radial_velocity.py`, holding functions of `(spec, params, times, regressors, cache)` that return the deterministic model and its named components. Dilution and flux offset follow Eq. 5 of the paper, and the linear and non-linear terms Eqs. 1 and 2.
  2. Use one implementation of the linear model for both.
  3. `model.generate_lc_model` and `generate_rv_model` become wrappers that call these with the fitted times and fill `self.model[...]` exactly as today.
  4. The cache must reuse batman and catwoman objects between likelihood calls. Benchmark. Planets with TTVs and models with `light_travel_delay` build a new batman object on every call today, because their times depend on the parameters. Keep that, and don't try to cache it in this step. A faster scheme (shifting `t0` per transit window in place of the times) is a separate proposal, since it changes numbers at the rounding level.
- **Done when:** `generate_lc_model` is under about 100 lines, all baselines and replay tests pass, and the benchmark is within 5%.

### Phase 5: evaluation

#### Step 5.1: the stateless predictor

- **Category:** A (new code, not yet used by the legacy path). **Depends on:** 4.4.
- **Do:** create `juliet/predict.py` with one function for every data kind and noise mode:

  ```text
  predict(spec, data, samples, instrument=None, t=None, regressors=None)
    1. deterministic model at the DATA times     → residuals, variances
    2. deterministic model at the REQUESTED times (phase 4 functions, own cache)
    3. noise.predict(residuals, variances, params, X_new)
    4. return every sample of the full model and of each component
  ```

  A single dictionary of scalars is treated as one sample. For the `T` parametrization, compute `P` and `t0` from the transit times of every instrument, as the likelihood does, when the samples don't already contain them: today `evaluate` reads them from the posterior, where `fit` stored them (section 2.6). Test it against today's `evaluate` for every cell of the evaluation matrix that works today.

#### Step 5.2: `evaluate_model` on top of it

- **Category:** B. **Depends on:** 5.1.
- **Do:**
  1. Rewrite `evaluate_model` as argument handling plus a call to `predict` and a summary function. Keep the signature and the order of every returned tuple, including the per-component medians that #56 describes.
  2. `evaluate(inst)` on a global model returns that instrument's points, and `evaluate()` returns the whole model in the order of `self.t` (already decided).
  3. Accept `GPregressors` as an array or a dictionary, and correct the docstring.
  4. `load` copies its input dictionaries.
  5. Remove every evaluation xfail that now passes. Add a seeded test that calls `evaluate` in random orders and checks for identical results and unchanged state.
  6. Draft replies for #121, #125, #128 and #132.
- **Done when:** no xfail remains in `test_evaluate_matrix.py`, or each survivor is explained.

#### Step 5.3: structured results and derived parameters

- **Category:** B (additions). **Depends on:** 5.2.
- **Do:**
  1. Add `evaluate_full(...)`, returning a `ModelEvaluation` dataclass: times, median, quantiles, optional samples, and a dictionary of components (each planet, GP, linear model, non-linear model, trend, instrument offset), each with the same fields. Offer a choice between per-component medians and components taken from the sample closest to the median model, so they add up (#56).
  2. Add `results.derived_samples()`, built on `orbits.py`: `b`, `p`, `a/R*`, inclination, `e`, `ω`, durations and depths, for every posterior sample. It includes what `fit` already derives and stores among the samples: `T_pN_<inst>_<n>` from `dt`, and `P_pN`, `t0_pN` from `T`. Those stay in `posterior_samples` too, for compatibility. `posteriors.dat` keeps its format.
  3. Draft replies for #21, #40, #56, #61 and #115.
- **Milestone:** `3.0.0a1`, tagged on `dev-3.0`. The maintainer runs the reference fits as `ref-3.0.0a1` on the server, and `compare_references.py ref-2.2.11 ref-3.0.0a1` must show no difference beyond the thresholds.

### Phase 6: parameters, data kinds and the likelihood

#### Step 6.1: characterize name parsing

- **Category:** none (tests). **Depends on:** 2.2.
- **Do:** extend the spec snapshots of step 2.2 to edge cases: instrument names that contain each other, or that equal tokens such as `p1`, `rv`, `B` or `C`; more than 10 GP regressors; shared parameters across three instruments; every TTV name form; and prior files in the old `sigma_w_rv_<inst>` form. Record crashes as strict xfails.

#### Step 6.2: the grammar

- **Category:** A. **Depends on:** 6.1, 5.2.
- **Do:**
  1. Create `juliet/parameters.py`: a declarative table of every parameter family (name pattern, scope, units, which component uses it), and one `resolve(priors, data, options) → ModelSpec`.
  2. Run it alongside `spec_from_legacy` on every snapshot. They must agree exactly, including on the inconsistencies, which are listed for the maintainer.
  3. Switch consumers to the new resolver one family at a time, and delete the matching legacy parsing in `generate_datadict`, `model.__init__` and `gaussian_process` as each family moves. `lc_options`, `rv_options` and the `*_iname` attributes are then rebuilt from the spec for users who read them.
  4. Families register themselves, so a new data kind adds its families without editing the grammar.

#### Step 6.3: data containers, data kinds and the likelihood

- **Category:** A. **Depends on:** 6.2.
- **Do:**
  1. Create `juliet/data.py`: `InstrumentData` (times, values, errors, named auxiliary columns, linear and GP regressors, time scale) and a `DataKind` registry with entries for photometry and RVs.
  2. `load` fills these, and its existing attributes (`t_lc`, `times_lc`, `instrument_indexes_rv` and the rest) become views of them.
  3. Create `juliet/likelihood.py`: a `Likelihood` holding one term per data kind plus the extra likelihood, with `__call__(params)`. `fit.loglike` delegates to it. It must pass the pickling test.
  4. Replace the O(N²) loop in `sort_GP` with an inverse permutation (#120), checking that the indexes are identical.
  5. Make `convert_time` convert between the scales astropy supports. No baseline may change, and a result change for existing UTC or TDB inputs is a finding to report.

#### Step 6.4: consistency fixes and validation

- **Category:** B, and C where noted. **Depends on:** 6.3.
- **Do**, one commit each:
  1. Shared `sigma_w` and `mu` behave the same for photometry, RVs and GPs.
  2. RV `non_linear_functions` is implemented (Opus decision 3).
  3. `phaseoffset` works for eclipse-only models, or raises (Opus decision 6). This is category C if it changes a model that runs today.
  4. Validation as warnings: unused priors, unphysical fixed values (#31), period order with mixed fixed and free periods (#54), and GP regressor length and order. Raise only for inputs that are certain to fail.
  5. A keyword or parameter that has no effect always produces a warning.
- **Report:** a table of every input that `load` now accepts, warns about or rejects differently.

#### Step 6.5: a new way to build a dataset (optional, decision D3)

- **Category:** B (addition). **Depends on:** 6.3.
- **Do:** add an alternative to the growing list of `load` keywords, without removing any:

  ```python
  dataset = juliet.load(priors=priors, datasets=[
      juliet.Photometry('TESS', t, f, ferr, ld_law='quadratic', gp_regressors=t),
      juliet.RadialVelocities('HARPS', t_rv, rv, rv_err),
  ])
  ```

  Both forms must produce the same spec and the same likelihood. The new form is the one astrometry will use.

### Phase 7: samplers, module split, packaging and docs

#### Step 7.1: sampler adapters

- **Category:** A, with one C commit. **Depends on:** 6.3.
- **Do:** as in the Opus prompt 5A, items 1 to 4. The dynesty keyword fix already landed in step 1.3, so the category C commit here covers only the other samplers' keyword filtering. File prefixes and pickle contents stay the same. Keep the Nautilus adapter as the acceptance test.

#### Step 7.2: results, input and output, and a thin `fit.py`

- **Category:** A. **Depends on:** 7.1.
- **Do:**
  1. Move reading and writing of data files, prior files, `posteriors.dat` and pickles to `juliet/io.py`, and posterior handling to `juliet/results.py`.
  2. `fit.py` keeps the four public classes as wrappers. `juliet.fit` must remain the class, as `from .fit import *` makes it today. Test the import surface.
  3. Replace the 19 bare `except:` statements with specific exceptions, one file at a time, checking that no baseline changes.
  4. Route `print` calls through `logging`, with a default handler that prints what is printed today, so `verbose` behaves the same.

#### Step 7.3: packaging

- **Category:** B. **Depends on:** 7.2.
- **Do:** as in the Opus prompt 5B, item 1. Add extras for `plots` and `astrometry`.

#### Step 7.4: documentation

- **Category:** none. **Depends on:** 7.2.
- **Do:** as in the Opus prompt 5B, items 2 to 6. The tutorial replay of step 0.3 becomes the docs' own test, and its xfails are fixed by correcting the docs.

#### Step 7.5: release 3.0.0

- **Category:** none. **Depends on:** 7.1 to 7.4.
- **Do:**
  1. Publish `3.0.0rc1` from `dev-3.0`. The maintainer runs the reference fits as `ref-3.0.0rc1`, and they are compared with `ref-2.2.11`.
  2. Ask the users who reported #121, #125, #128, #132 and #135 to try it on their own data.
  3. Release 3.0.0 when the suite passes and no report is open: merge `dev-3.0` into `dev`, then `dev` into `master`, and tag.

### Phase 8: plotting suite

#### Step 8.1: plot data

- **Category:** B (addition). **Depends on:** 5.3.
- **Do:** create `juliet/plots/data.py`, with functions that take `results` and return plain arrays ready to draw, built only on `evaluate_full` and `derived_samples`:
  - the phased light curve of one planet, with every other component removed, optionally binned;
  - the same for RVs;
  - the time series with the full model and each component;
  - residuals;
  - observed-minus-calculated transit times;
  - prior and posterior densities for a parameter.

  Add golden tests of these arrays.

#### Step 8.2: drawing

- **Category:** B (addition). **Depends on:** 8.1.
- **Do:**
  1. Create `juliet/plots/draw.py`: one matplotlib function per item above, each accepting an `ax` and returning it; a corner plot that includes derived parameters; and `results.plot_summary(out_folder)`, which writes the standard set for a fit (#27).
  2. matplotlib and corner are imported inside the functions.
  3. Test with the Agg backend that every function runs for every golden data kind and noise mode.
  4. Rewrite the tutorials' plotting code to use these, keeping the old code in one tutorial as an example of doing it by hand.
- **Note:** the old branches `origin/automated_plots` and `origin/errors_on_components` (both 2020) hold earlier attempts. Read them for intent, and don't merge them.

### Phase 9: astrometry

#### Step 9.0: astrometry through the extra likelihood (can start after 1.4 and 2.1, on `dev-3.0`)

- **Category:** B (addition). **Depends on:** 1.4, 2.1, and decision D4.
- **Do:**
  1. Create `juliet/models/astrometry.py` as a pure module: the position of the star's photocentre from the five astrometric parameters plus a Keplerian orbit in Thiele-Innes form, and its projection onto a scan direction for along-scan data. It uses `orbits.py` for Kepler's equation and for the conversion between transit time and periastron time.
  2. Unit test it against an independent implementation, and against closed-form cases: a circular face-on orbit, an edge-on orbit, and zero companion mass.
  3. Add a helper that builds an `extra_loglikelihood` dictionary from astrometric data, so a user can write the following on the legacy code:

     ```python
     astro = juliet.astrometry_likelihood(t, abscissa, err, scan_angle, parallax_factor, instrument='Gaia')
     dataset = juliet.load(priors=priors, t_rv=..., y_rv=..., yerr_rv=..., extra_loglikelihood=astro)
     ```

     The priors include the astrometric names, and `P_p1`, `t0_p1`, `ecc_p1` and `omega_p1` are shared with the RV and transit models automatically, because every sampled parameter reaches the extra likelihood.
  4. Add a synthetic joint RV and astrometry recovery test to the sampler layer.
- **Constraints from the code (section 2.6):** `load` needs photometry or RVs, so the bridge can't fit astrometry alone. A planet seen only by astrometry has no `K_pN`, `r1_pN` or `b_pN`, so juliet doesn't count it and the period-order check skips it: the helper must check the order itself. The object returned by `juliet.astrometry_likelihood` must be an instance of a module-level class, so it pickles for `nthreads`.
- **Limits to state in the docs:** no `results.astrometry.evaluate`, no astrometry in `posteriors.dat` beyond the sampled parameters, and no plots, until step 9.2.

#### Step 9.1: design note

- **Category:** none. **Depends on:** 9.0.
- **Do:** write `docs/dev/astrometry_design.md`, settling the following with the maintainer: the parameter names; the conventions for inclination and node; whether `K` and the astrometric semi-major axis are independent or tied through masses and parallax; which data types come first (Gaia along-scan epochs, two-dimensional relative astrometry, Hipparcos–Gaia proper-motion anomalies); the reference epoch; and how a transit's `b` and an astrometric inclination coexist.

#### Step 9.2: astrometry as a data kind

- **Category:** B (addition). **Depends on:** 6.5, 9.1.
- **Do:**
  1. Register an `astrometry` data kind: its auxiliary columns, its parameter families (system and per planet), the forward model from step 9.0, jitter in milliarcseconds, and white noise by default. The noise classes of phase 3 apply unchanged.
  2. Add `juliet.Astrometry(...)` to the new-style `load`, an `astrometry.dat` file format for input folders, `results.astrometry.evaluate(...)` through `predict`, and derived parameters such as the companion mass.
  3. The likelihood must equal that of step 9.0 for the same inputs, to the golden tolerance.
  4. Add golden configurations and plot-data functions for the sky track and for along-scan residuals.

#### Step 9.3: validation

- **Category:** none. **Depends on:** 9.2.
- **Do:**
  1. Synthetic: simulate Gaia-like epochs with a realistic scan law for a known companion, alone and with RVs, and recover the truth.
  2. Literature: Gaia BH3 is a candidate, because I recall its epoch astrometry being published ahead of DR4 with the discovery paper. **Confirm that the data are available and in what format before relying on this.** A DR4 source with a published orbit can be added once the release is out.
  3. Add both to the scientific suite with calibrated thresholds.

---

## 7. Decisions for you

| # | Decision | My recommendation |
|---|---|---|
| D1 | **Versions and branches.** | **Decided:** fixes as 2.2.11 on `dev-2.2.11`, merged to `dev` and then `master`. The refactor is 3.0 on `dev-3.0`, cut from 2.2.11. No separate maintenance branch: `dev` is the 2.2.x line until 3.0.0. |
| D2 | **Where validation data and references live.** | **Decided (2026-11-24):** in this repository, under `validation/`, excluded from the built package. No separate repository. It must stay well inside GitHub's limits (files under 50 MB, repository well under 1 GB), so: 1 MB per fit, 15 MB per reference, and 100 MB for all references over the whole project. Bundles are written once and never regenerated in place, because every rewrite of a binary file stays in the history. |
| D3 | **A new-style `load(datasets=[...])`** beside the keyword form. | Yes. Astrometry needs per-point columns that the `t_x, y_x, yerr_x` pattern can't carry. |
| D4 | **Astrometry conventions:** names (I suggest `plx`, `pmra`, `pmdec`, `dra_<inst>`, `ddec_<inst>`, `a0_p1`, `inc_p1`, `lan_p1`), and whether `K_p1` and `a0_p1` are sampled independently. | Independent first. It matches juliet's "names define the model" approach and needs no stellar mass. Add a mass-based option later. Avoid `Omega_p1`, which differs from `omega_p1` only by case. |
| D5 | **Global GP likelihood errors return `-inf`** (step 1.4, item 3). | Yes, catching specific exceptions and counting them. |
| D6 | **Minimum Python version.** | The lowest that CI passes in step 0.2, probably 3.10. |
| D7 | **Commit the two plan files?** | **Decided (2026-11-24):** yes, under `docs/dev/refactoring/`. Done in step 0.1. |
| D8 | **The eight open decisions in the Opus plan's section 6.** | I agree with its recommendation on each. Steps 1.2, 1.4, 1.6, 6.4 and 3.2 are where they take effect. |
| D9 | **Where the reference fits run.** | **Decided:** the maintainer's server, returning the lightweight bundles of section 5.5. The weekly CI job runs only the fits that finish in an hour. |
| D10 | **Reading of "legacy mode".** | **Decided:** existing usage must keep working. |
| D11 | **What juliet does when it finds results that don't match the current data or priors** (step 1.7b). Today it returns them silently. | **Decided (2026-11-24):** a matching fingerprint always loads. On a mismatch, raise when `out_folder` was given, and refit when it wasn't. Files without a fingerprint load as today, with a warning when the parameter names differ. |
| D12 | **The `Exponential` prior** is a Gamma distribution of shape `a` (step 1.6). | **Decided (2026-11-24):** keep the behaviour and correct the documentation. No new prior for now. |

---

## 8. Sequence and risks

```text
0.1 → 0.2 → 0.3 ─┬→ 0.5a → 0.5b (server) → 0.5c → 1.1 … 1.7b → 1.8 (2.2.11, then cut dev-3.0)
      0.4 ───────┘         │
      0.6                  ├→ 9.0 (astrometry bridge, needs 1.4 and 2.1)
                           ↓
            2.1 → 2.2 → 3.1 → 3.2 ─┐
                   └→ 4.1 → 4.2 → 4.3 → 4.4 → 5.1 → 5.2 → 5.3 (3.0.0a1)
                                                      │       └→ 8.1 → 8.2
                              6.1 → 6.2 → 6.3 → 6.4 → 6.5 → 7.1 … 7.5 (3.0.0)
                                                       └→ 9.1 → 9.2 → 9.3 (3.1)
```

| Risk | Mitigation |
|---|---|
| A refactor step changes results in a mode no test covers. | Posterior replay on nine real datasets on every PR. Step 4.2 adds goldens before merging code paths. |
| The reference fits are too noisy to detect small shifts. | Thresholds are calibrated from repeats. The exact comparison is the replay test, not the refit. |
| Phase 1 fixes disturb users in the middle of an analysis. | A "Changed results" entry per fix that states who is affected. `ref-2.2.10` against `ref-2.2.11` shows that nothing else moved. |
| The likelihood gets slower. | The benchmark of step 0.6 runs on every phase 3 to 6 step, with a 5% limit. |
| The refactor stalls halfway. | Every step leaves `dev-3.0` passing both suites, and the 2.2.x line on `dev` is untouched by it. Phases 0 and 1 are worth doing on their own. |
| The astrometry bridge of step 9.0 sets conventions that step 9.2 regrets. | Decision D4 is settled before step 9.0, and the step 9.0 module is the one step 9.2 reuses. |
| Upstream packages change during the work (dynesty 3, numpy). | The latest-dependencies CI job, and pinned baselines. |

**What this review didn't cover:**

- **I didn't re-run the Opus probes,** only its test suite.
- **I read all of `juliet/fit.py`, `juliet/utils.py`, `__init__.py`, `__main__.py`, `setup.py` and `pyproject.toml`** in the second pass, and ran probes for the findings marked **[ran]** in section 2.6.
- **I only skimmed** `KeplerOrbit.py` (706 lines, used only by the light-travel correction), `plots.py` (it can't be imported), the kelp functions' physics, and the scripts in `examples/validation/`.
- **I didn't run** catwoman, kelp, MultiNest, UltraNest or zeus paths, or anything with `nthreads`.
- **I read no fit output,** because no scientific fit has been run.
- **The astrometry section is a design, not a tested model.** The claim about Gaia BH3 data is from memory.
- **GitHub issues were read through the public API** on 2026-11-24: the first 1100 characters of each open issue, without comments, and closed issues by title only.
