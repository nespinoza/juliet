from setuptools import setup

setup(name='juliet',
      version='2.0.22',
      description='juliet: a versatile modelling tool for transiting exoplanets, radial-velocity systems or both',
      url='http://github.com/nespinoza/juliet',
      author='Nestor Espinoza',
      author_email='espinoza@mpia.de',
      license='MIT',
      packages=['juliet'],
      install_requires=['batman-package','radvel','dynesty','george','celerite','astropy','numpy','scipy'],
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
