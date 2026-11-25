"""Shared pytest configuration for the juliet test suite. See tests/README.md."""
import datetime
import importlib
import os
import subprocess
from pathlib import Path

# Must happen before anything imports matplotlib.pyplot:
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pytest

import juliet
from support import versions

GOLDEN_DIR = Path(__file__).parent / 'golden'
REASON_VARIABLE = 'JULIET_GOLDEN_REASON'

OPTIONAL_MODULES = {
    'requires_catwoman': 'catwoman',
    'requires_kelp': 'kelp',
    'requires_multinest': 'pymultinest',
}


def pytest_addoption(parser):
    parser.addoption('--update-golden', action='store_true',
                     help='Rewrite stored baselines. Needs a non-empty ' + REASON_VARIABLE + ' environment variable.')
    parser.addoption('--run-sampler', action='store_true', help='Run the sampler tests (minutes).')
    parser.addoption('--run-scientific', action='store_true', help='Run the full tutorial fits (hours).')
    parser.addoption('--run-tutorials', action='store_true',
                     help='Run the code of the documentation pages with small sampler settings (minutes).')


def pytest_configure(config):
    config.golden_files_written = []
    if config.getoption('--update-golden') and not os.environ.get(REASON_VARIABLE, '').strip():
        raise pytest.UsageError('--update-golden needs the reason for the change in ' + REASON_VARIABLE +
                                '. Baselines may only change on purpose; see tests/README.md.')


def pytest_report_header(config):
    return ['juliet: {} from {}'.format(juliet.__version__, os.path.dirname(juliet.__file__)),
            'dependencies: ' + ', '.join('{}={}'.format(k, v) for k, v in versions().items())]


def optional_module_available(module):
    try:
        importlib.import_module(module)
        return True
    except (ImportError, OSError, SystemExit):
        return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, module in OPTIONAL_MODULES.items():
            if item.get_closest_marker(marker) and not optional_module_available(module):
                item.add_marker(pytest.mark.skip(reason=module + ' is not installed'))
        if item.get_closest_marker('sampler') and not config.getoption('--run-sampler'):
            item.add_marker(pytest.mark.skip(reason='needs --run-sampler'))
        if item.get_closest_marker('scientific') and not config.getoption('--run-scientific'):
            item.add_marker(pytest.mark.skip(reason='needs --run-scientific'))
        if item.get_closest_marker('tutorials') and not config.getoption('--run-tutorials'):
            item.add_marker(pytest.mark.skip(reason='needs --run-tutorials'))


def pytest_sessionfinish(session, exitstatus):
    written = session.config.golden_files_written
    if not written:
        return
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'juliet'], capture_output=True, text=True,
                           cwd=Path(__file__).parent.parent).stdout.strip()
    entry = ['', '## {}'.format(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')), '',
             '- **Reason:** ' + os.environ[REASON_VARIABLE].strip(),
             '- **Commit:** {}{}'.format(commit, ' (with uncommitted changes under juliet/)' if dirty else ''),
             '- **Dependencies:** ' + ', '.join('{}={}'.format(k, v) for k, v in versions().items()),
             '- **Files ({}):** {}'.format(len(written), ', '.join(sorted(written))), '']
    with open(GOLDEN_DIR / 'CHANGES.md', 'a') as f:
        f.write('\n'.join(entry))


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch, request):
    """
    juliet writes to the working directory when it has no out_folder, so never run in the repository.

    A fit without out_folder also reads any earlier results it finds in the working directory. So that no test does that
    by accident, a test that leaves result files there fails, unless it is marked writes_to_cwd.
    """
    monkeypatch.chdir(tmp_path)
    yield
    if request.node.get_closest_marker('writes_to_cwd'):
        return
    left = sorted(path.name for path in tmp_path.iterdir()
                  if path.name.endswith('posteriors.pkl') or path.name in ('posteriors.dat', 'lc.dat', 'rvs.dat'))
    assert not left, 'This test ran a fit without out_folder, which left {} in the working directory.'.format(left)


@pytest.fixture(autouse=True)
def seeded_global_rng():
    """evaluate_model draws from numpy's global generator when all_samples is False."""
    state = np.random.get_state()
    np.random.seed(20190101)
    yield
    np.random.set_state(state)


@pytest.fixture
def out_folder(tmp_path):
    return str(tmp_path / 'out') + '/'


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """On GitHub Actions, repeat each failure as an annotation, so it shows on the pull request's checks page."""
    if not os.environ.get('GITHUB_ACTIONS'):
        return
    reports = [r for outcome in ('failed', 'error') for r in terminalreporter.stats.get(outcome, [])]
    if not reports:
        return

    def escape(text):
        return text.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')

    def last_lines(report, n):
        return '\n'.join(str(report.longrepr).strip().splitlines()[-n:])

    def cause(report):
        crash = getattr(report.longrepr, 'reprcrash', None)
        return (crash.message if crash else last_lines(report, 1)).replace('\n', ' ')[:300]

    # GitHub keeps ten error annotations per step: nine failures in detail, then one with every name.
    for report in reports[:9]:
        print('::error title={}::{}'.format(escape(report.nodeid).replace(',', '%2C').replace(':', '%3A'),
                                           escape(last_lines(report, 25))))
    print('::error title={} failed or errored::{}'.format(
        len(reports), escape('\n'.join('{} | {}'.format(r.nodeid, cause(r)) for r in reports)[:60000])))
