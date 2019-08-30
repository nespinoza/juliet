from setuptools import setup

setup(name='juliet',
      version='2.0',
      description='juliet: a versatile modelling tool for transiting exoplanets, RVs or both',
      url='http://github.com/nespinoza/juliet',
      author='Nestor Espinoza',
      author_email='espinoza@mpia.de',
      license='MIT',
      packages=['juliet'],
      install_requires=['batman-package','radvel','dynesty','george','celerite','astropy'],
      extras_requires={
            'seaborn':['seaborn'],
            'pymultinest':['pymultinest'],
            'matplotlib':['matplotlib'],},
      zip_safe=False)
