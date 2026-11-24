"""
The documentation as a test: every Python block of the tutorials is run in order, as a reader would.

Fits use dynesty with 50 live points, so this checks that the calls work and return what the text says, not the
numbers. It needs --run-tutorials (about ten minutes) and the three TESS light curves of the tutorials, which step 0.4
of the refactoring plan stores in tests/data/tess. Until then, set JULIET_TEST_TESS_DATA to a folder that has them.

Blocks that fail today are listed in FAILING with their reasons. The list is strict in both directions: a block that
starts to fail, or one that is fixed, fails the test until the list is updated.
"""
from pathlib import Path

import pytest

from contract.tutorial_replay import REPOSITORY, MissingData, code_blocks, replay

DOCS = REPOSITORY / 'docs'
PAGES = ['user/quicktest', 'tutorials/transitfits', 'tutorials/rvfits', 'tutorials/jointfits', 'tutorials/gps',
         'tutorials/linearmodels', 'tutorials/ttvs', 'tutorials/photometry', 'tutorials/multithreading']

# Blocks that show the shape of a call and aren't meant to run by themselves. {page: {line: why}}
SNIPPETS = {
    'user/quicktest': {15: 'priors, times and flux stand for the reader\'s own data'},
    'tutorials/photometry': {15: 'priors, times and flux stand for the reader\'s own data'},
    'tutorials/multithreading': {42: 'assumes a dataset from another tutorial'},
    'tutorials/transitfits': {480: 'assumes the LCOGT and SWOPE data and priors are loaded'},
}

SUBPLOT_STRING = 'plt.subplot is given a string such as \'211\', which matplotlib no longer accepts (it needs 211).'
# Blocks that raise today. {page: {line: why}}
FAILING = {
    'tutorials/transitfits': {
        198: 'SyntaxError: a space follows the line-continuation backslash.',
        250: 'plt.errorbar gets xerr of shape (1, 2) for one point, which matplotlib no longer accepts.',
        345: SUBPLOT_STRING,
    },
    'tutorials/rvfits': {
        218: 'SyntaxError: a space follows the line-continuation backslash.',
    },
    'tutorials/gps': {
        531: SUBPLOT_STRING,
        570: 'Reads GP_regressors_rv_i-i-FEROS.dat, which isn\'t in docs/tutorials.',
        618: SUBPLOT_STRING + ' It also needs the results of the block at line 570.',
    },
    'tutorials/ttvs': {
        84: 'Uses np, but the tutorial never imports numpy.',
        135: 'Uses np, but the tutorial never imports numpy.',
        208: 'Uses n_live_points, which is never defined, and the priors of the block at line 135.',
    },
}


def page_path(page):
    return DOCS / (page + '.rst')


@pytest.mark.contract
@pytest.mark.parametrize('page', PAGES)
def test_blocks_are_valid_python(page):
    """Fast, and always run: catches a tutorial edit that breaks the syntax of a block."""
    blocks = code_blocks(page_path(page))
    assert blocks, 'No Python blocks found: has the page changed its markup?'
    invalid = set()
    for line, code in blocks:
        try:
            compile(code, '{}:{}'.format(page, line), 'exec')
        except SyntaxError:
            invalid.add(line)
    expected = {line for line, why in FAILING.get(page, {}).items() if why.startswith('SyntaxError')}
    assert invalid == expected


@pytest.mark.contract
@pytest.mark.parametrize('page', PAGES)
def test_listed_blocks_exist(page):
    """The lists above point at blocks by line number, so they must follow edits of the pages."""
    lines = {line for line, _ in code_blocks(page_path(page))}
    listed = set(SNIPPETS.get(page, {})) | set(FAILING.get(page, {}))
    assert listed <= lines


@pytest.mark.tutorials
@pytest.mark.parametrize('page', PAGES)
def test_tutorial_runs(page, tmp_path):
    pytest.importorskip('corner')
    folder = tmp_path / 'tutorial'
    folder.mkdir()
    try:
        failures = replay(page_path(page), folder, skip=SNIPPETS.get(page, {}))
    except MissingData as missing:
        pytest.skip('{}. See the top of this file.'.format(missing))
    expected = FAILING.get(page, {})
    unexpected = {line: error for line, error in failures.items() if line not in expected}
    fixed = sorted(set(expected) - set(failures))
    assert not unexpected, 'Blocks that used to run now fail: {}'.format(unexpected)
    assert not fixed, 'Blocks at lines {} now run: remove them from FAILING.'.format(fixed)
