import re
from setuptools import setup

VERSIONFILE='juliet/_version.py'
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='juliet',
      version=verstr,
      description='juliet: a versatile modelling tool for transiting exoplanets, radial-velocity systems or both',
      url='http://github.com/nespinoza/juliet',
      author='Nestor Espinoza',
      author_email='nespinoza@stsci.edu',
      license='MIT',
      packages=['juliet'],
      install_requires=['batman-package','radvel','dynesty>=1.2.2','george','celerite','astropy','numpy','scipy', 'emcee', 'ultranest'],
      python_requires='>=2.7',
      extras_requires={
            'seaborn':['seaborn'],
            'pymultinest':['pymultinest'],
            'matplotlib':['matplotlib'],},
      entry_points={
            'console_scripts': [
                 'juliet=juliet.__main__:main'
            ]},
      zip_safe=False)
