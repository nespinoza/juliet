"""The one place that says how close a result must be to its baseline. See tests/README.md for how these were measured."""
LOGLIKE_ABS = 1e-8
# Very poor fits have log-likelihoods of -1e7, where double precision alone gives differences above LOGLIKE_ABS:
LOGLIKE_REL = 1e-13
ARRAY_REL = 1e-10
# Relative comparisons are meaningless around zero. This is far below any flux or velocity signal in the configurations:
ARRAY_ABS = 1e-12
# Elements near zero of an array that isn't: they carry the rounding of the large elements. Times the largest |value|.
# Linux runners differ from the macOS baselines by 1.5e-11 of the largest value in a celerite prediction.
ARRAY_ABS_OF_LARGEST = 1e-9

# catwoman integrates numerically, so its light curves depend on the platform: Linux differs from the macOS baselines
# by 7e-10 relative in the model and up to 1.2e-7 relative in the log-likelihood (supersampled case).
CATWOMAN_ARRAY_REL = 1e-8
CATWOMAN_LOGLIKE_REL = 1e-6
