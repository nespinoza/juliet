"""Helpers shared by the juliet tests: synthetic data, model building without a sampler, kernels."""
import importlib
import importlib.metadata
import platform

_PACKAGES = ['numpy', 'scipy', 'batman-package', 'radvel', 'george', 'celerite', 'dynesty', 'catwoman', 'kelp']
# Some builds of celerite report 0.0.0 in their metadata, but the right version in the module:
_FROM_MODULE = ['celerite']


def versions():
    out = {'python': platform.python_version()}
    for package in _PACKAGES:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    for module in _FROM_MODULE:
        if module in out:
            out[module] = importlib.import_module(module).__version__
    return out
