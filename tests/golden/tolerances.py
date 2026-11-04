"""The one place that says how close a result must be to its baseline. See tests/README.md for how these were measured."""
LOGLIKE_ABS = 1e-8
# Very poor fits have log-likelihoods of -1e7, where double precision alone gives differences above LOGLIKE_ABS:
LOGLIKE_REL = 1e-13
ARRAY_REL = 1e-10
# Relative comparisons are meaningless around zero. This is far below any flux or velocity signal in the configurations:
ARRAY_ABS = 1e-12
