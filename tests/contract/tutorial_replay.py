"""
Runs the Python code blocks of a documentation page in order, as a reader would, with small sampler settings.

Used by test_tutorial_replay.py. It can also be run by hand, to see what each block does:

    python tests/contract/tutorial_replay.py docs/tutorials/rvfits.rst
"""
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

REPOSITORY = Path(__file__).parents[2]
TUTORIALS = REPOSITORY / 'docs' / 'tutorials'
# Light curves that the tutorials download from MAST. Step 0.4 of the refactoring plan stores them here:
TESS_DATA = Path(os.environ.get('JULIET_TEST_TESS_DATA', REPOSITORY / 'tests' / 'data' / 'tess'))
LIVE_POINTS = 50


class MissingData(Exception):
    pass


def code_blocks(path):
    """[(line number of the directive, code)] for every '.. code-block:: python' of a reStructuredText file."""
    lines = Path(path).read_text().split('\n')
    out, i = [], 0
    while i < len(lines):
        match = re.match(r'^(\s*)\.\. code-block::\s*(\w+)', lines[i])
        if not match:
            i += 1
            continue
        indent, language = len(match.group(1)), match.group(2)
        body, i = [], i + 1
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
            body.append(lines[i])
            i += 1
        directive = i - len(body)
        # Options of the directive, such as ':emphasize-lines: 16', come before the code:
        while body and re.match(r'^\s*:[\w-]+:', body[0]):
            body.pop(0)
        if language == 'python':
            out.append((directive, textwrap.dedent('\n'.join(body)).strip('\n')))
    return out


@contextlib.contextmanager
def small_and_offline():
    """Fits use dynesty with few live points, and TESS light curves come from local files."""
    import matplotlib.pyplot as plt

    import juliet

    original_fit, original_tess, original_show = juliet.load.fit, juliet.utils.get_TESS_data, plt.show

    def fit(self, **kwargs):
        for name in ('use_dynesty', 'dynamic', 'use_ultranest', 'dynesty_nthreads', 'nthreads'):
            kwargs.pop(name, None)
        kwargs['sampler'] = 'dynesty'
        kwargs['n_live_points'] = min(kwargs.get('n_live_points', LIVE_POINTS), LIVE_POINTS)
        return original_fit(self, **kwargs)

    def get_TESS_data(filename, *args, **kwargs):
        local = TESS_DATA / os.path.basename(filename)
        if not local.exists():
            raise MissingData('{} is not in {}'.format(local.name, TESS_DATA))
        return original_tess(str(local), *args, **kwargs)

    juliet.load.fit, juliet.get_TESS_data, juliet.utils.get_TESS_data = fit, get_TESS_data, get_TESS_data
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        juliet.load.fit, juliet.get_TESS_data, juliet.utils.get_TESS_data = original_fit, original_tess, original_tess
        plt.show = original_show
        plt.close('all')


def replay(path, folder, skip=(), report=None):
    """
    Runs the blocks of a page inside folder, sharing one namespace. Returns {line: 'ExceptionName: message'} for the
    blocks that raised. Blocks in skip aren't run. Raises MissingData if a light curve isn't available.
    """
    failures = {}
    namespace = {'__name__': '__tutorial__'}
    for data_file in list(TUTORIALS.glob('*.dat')) + list(TUTORIALS.glob('*.txt')):
        shutil.copy(data_file, folder)
    previous = os.getcwd()
    os.chdir(folder)
    try:
        with small_and_offline():
            for line, code in code_blocks(path):
                if line in skip:
                    continue
                start = time.time()
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exec(compile(code, '{}:{}'.format(Path(path).name, line), 'exec'), namespace)
                except MissingData:
                    raise
                except Exception as error:
                    failures[line] = '{}: {}'.format(type(error).__name__, str(error).split('\n')[0][:200])
                    if report:
                        report('block at line {} failed after {:.0f} s\n{}'.format(line, time.time() - start,
                                                                                  traceback.format_exc(limit=-3)))
                else:
                    if report:
                        report('block at line {} ran in {:.0f} s'.format(line, time.time() - start))
    finally:
        os.chdir(previous)
    return failures


if __name__ == '__main__':
    sys.path.insert(0, str(REPOSITORY))
    for page in sys.argv[1:]:
        with tempfile.TemporaryDirectory() as folder:
            print(page, replay(page, folder, report=lambda text: print('   ', text, flush=True)), flush=True)
