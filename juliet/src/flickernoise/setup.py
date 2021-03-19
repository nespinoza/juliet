from distutils.core import setup, Extension
import numpy

"""
According to GSL documentation (http://www.gnu.org/software/gsl/manual/html_node/Shared-Libraries.html), in order to run the different operations one must include the GSL library, the GSLCBLAS library and the math library. To compile in C one must do:
  	
  gcc -Wall -c filename.c

And then:

  gcc -static nombredelarchivo.o -lgsl -lgslcblas -lm

The first part is done by Python by this file. The second part (adding "-lgsl -lgslcblas -lm"), obviously isn't. To add any libraries that in C would be called by:

  gcc -static nombredelarchivo.o -lname1 -lname2 -lname3...

Is as simple as putting libraries=['name1','name2',...] inside the Extension module. Here we do it with "gsl", "gslcblas" and "m".
"""

module = Extension('FWT', sources = ['FWT.c'],libraries=['m'], include_dirs=[numpy.get_include(),'/usr/local/include']) 
setup(name = 'Fast Wavelet Transform, C/Python extension ', version = '1.0', ext_modules = [module])
