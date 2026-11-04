# GP conventions: what the code does today

These are checked by `tests/unit/test_gp_dense.py`, whose closed forms are in `tests/support/kernels.py`. Where they differ from the documentation, the baselines freeze **the code's** behaviour. Changing any of these changes results for existing prior files, so it's a category C change.

`τ` is the distance between two regressor values. `f` is the ppm factor: `1e-6` for photometry and `1` for RVs.

| Kernel | Covariance computed today | Differs from the documentation |
|---|---|---|
| `SEKernel` | `(f σ)² exp(−α τ²/2)` | The paper writes `exp(−α τ²)`. |
| `M32Kernel` | `(f σ)² (1 + r) exp(−r)`, with `r = sqrt(3 α) τ` | |
| `ExpSineSquaredSEKernel` | `(f σ)² exp(−α τ²/2) exp(−ln(Γ) sin²(π τ/P))` | **`Γ` enters as `ln(Γ)`.** juliet passes `log(Γ)` to george's `gamma`, which isn't a logarithm. `Γ < 1` gives a negative coefficient. |
| `CeleriteQPKernel` | `B/(2+C) exp(−τ/L) (cos(2πτ/P) + 1 + C)` | |
| `CeleriteExpKernel` | `σ exp(−T τ)` | **`GP_sigma` acts as a variance and isn't converted from ppm. `GP_timescale` acts as a rate**, not a time scale. |
| `CeleriteMaternKernel` | `σ² exp(−w τ) (cos(ε τ) + (w/ε) sin(ε τ))`, with `w = sqrt(3)/ρ`, `ε` = `matern_eps` | **`GP_sigma` isn't converted from ppm.** `ε` multiplies `τ`, and the paper has a slightly different expression. |
| `CeleriteMaternExpKernel` | `σ exp(−T τ) × [the Matern above with σ = 1]` | Same as the two above: **variance, no ppm conversion, and a rate.** |
| `CeleriteSHOKernel` | Standard celerite SHO | |
| `CeleriteDoubleSHOKernel` | Sum of two SHOs, with `ω₂ = 8π Q₁ / (P sqrt(4 Q₁² − 1))` | **Uses `Q₁` in `ω₂`.** The code's comment says this follows celerite2's `RotationTerm`, which uses `Q₂` there. |
| `CeleriteMaternSHOKernel` | Matern plus SHO | **`GP_sigma` isn't converted from ppm.** |

## Noise terms

- **Jitter:** per-instrument GPs carry it as a kernel term. Global GPs add it to `yerr`. For celerite kernels the per-instrument jitter parameter is `log(f σ_w)`, for george kernels it is `log((f σ_w)²)`; both give the same variance.
- **Hidden white noise in global george GPs:** global GPs keep george's default white noise, `1.25e-12` in variance, or about 1.1 ppm added in quadrature. It has an effect only for photometry. On the two-instrument, 100-point dataset here, it shifts the log-likelihood by about 0.3.
- **Precision:** photometric log-likelihoods agree with a dense calculation to about `6e-6`, as celerite loses precision with variances near `1e-8`. RVs agree to about `1e-10`.

## See also

Each of these has a strict `xfail` in `tests/contract/test_known_bugs.py`, so a fix will be noticed:
`test_exp_sine_squared_gamma_is_the_documented_coefficient` and `test_double_sho_matches_celerite2_rotation_term`.
