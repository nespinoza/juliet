# The Spotrod package: a semi-analytic model for transits of spotted stars
#
# Copyright 2013, 2014 Bence Béky
#
# This file was not part of Spotrod.
# These classes were written to a smilar way to the Batman package by Laura Kreidberg:
# https://github.com/lkreidberg/batman
#
# Spotrod is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Spotrod is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Spotrod.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from ctypes import c_void_p, c_double, c_int, cdll

#load the compiled C library
spotrod = cdll.LoadLibrary("./spotrod.so")  

__all__ = ['TransitModel', 'TransitParams']

class TransitModel(object):
	"""
	Class for generating model transit light curves.	

	:param params: A :attr:`TransitParams` object containing the physical parameters of the transit
	:type params: a `TransitParams` instance

	:param t: Array of times at which to calculate the model
	:type t: ndarray 
	
	:param n_r: Number of intergration rings
	:type n_r: float, optional

	:Example:
	
	>>> m = spotrod_mod.TransitModel(params, t)
	"""

	def __init__(self, params, t, n_r = 1000):
		#initializes model parameters
		self.t = t
		self.t0 = params.t0
		self.per = params.per
		self.rp = np.abs(params.rp)
		self.a = params.a
		self.b = params.b
		self.ecc = params.ecc
		self.w = params.w
		self.u = params.u
		self.limb_dark = params.limb_dark
		self.spotx = np.array([params.spotx])
		self.spoty = np.array([params.spoty])
		self.spotrad = np.array([params.spotrad])
		self.spotcont = np.array([params.spotcont])	
		self.n_r = n_r
		
	def quadratic_ld(self, params, r): 
		"""
		Calculate quadratic limb darkening function (Claret et al. 2000); I(mu)/I(1) = 1 - a(1-mu) - b(1-mu)^2.

		:param params: Transit parameters
		:type params: A `TransitParams` instance
		
		:param r: Integration annulii radii
		:type r: ndarray 

		:return: Specific intensity
		:rtype: ndarray
		"""
		oneminusmu = 1. - np.sqrt(1. - r**2)
		return 1. - params.u[0] * oneminusmu - params.u[1] * oneminusmu**2

	def elements(self, params):
		"""
		Calculate orbital elements eta and xi.

		:param params: Transit parameters
		:type params: A `TransitParams` instance
		
		:return: Orbital elements eta and xi.
		:rtype: ndarray
		"""
		deltaT = self.t - params.t0
		eta = np.empty(self.t.size)
		xi = np.empty(self.t.size)
		k = params.ecc*np.cos(params.w*np.pi/180.)
		h = params.ecc*np.sin(params.w*np.pi/180.)		
		spotrod.elements(
			c_void_p(deltaT.ctypes.data), c_double(params.per),c_double(params.a), c_double(k), c_double(h),c_int(deltaT.size),c_void_p(eta.ctypes.data),c_void_p(xi.ctypes.data))	
		return eta, xi
		
	def circleangle(self, params, r, z, planetangle):
		"""
		Calculate half central angle of the arc of circle of radius r (which concentrically spans the inside of the star during integration) that is inside a circle of radius rp (planet).

		:param params: Transit parameters
		:type params: A `TransitParams` instance

		:param r: Radii of integration annuli (in stellar radii)
		:type r: ndarray 

		:param z: Planetary center distance from stellar disk center (in stellar radii)  
		:type z: ndarray 

		:param planetangle: Planetangle array
		:type planetangle: ndarray
						
		:return: Planetangle array
		:rtype: ndarray
		"""
		return spotrod.circleangle(
			c_void_p(r.ctypes.data), c_double(params.rp), c_double(z), c_int(r.size),c_void_p(planetangle.ctypes.data))

	def integratetransit(self, params, planetx, planety, z, r, f, planetangle):
		"""
		Calculate integrated flux of a star if it is transited by a planet at projected position (planetx, planety).

		:param params: Transit parameters
		:type params: A `TransitParams` instance

		:param planetx: Planetary center x-coordinate (in stellar radii in sky-projected coordinate system)
		:type planetx: ndarray 
		
		:param planety: Planetary center y-coordinate (in stellar radii in sky-projected coordinate system)
		:type planety: ndarray 
		
		:param z: Planetary center distance from stellar disk center (in stellar radii)  
		:type z: ndarray 

		:param r: Radii of integration annuli (in stellar radii)
		:type r: ndarray 
		
		:param f: 2.0 * limb darkening at r[i] * width of annulus i
		:type f: ndarray 
		
		:param planetangle: Planetangle array
		:type planetangle: ndarray
		
		:return: Model transit lightcurve
		:rtype: ndarray
		"""
		tmodel = np.empty(self.t.size) 
		spotrod.integratetransit(
			c_int(self.t.size), c_int(r.size), c_int(params.spotcont.size),c_void_p(planetx.ctypes.data),c_void_p(planety.ctypes.data),c_void_p(z.ctypes.data),c_double(params.rp),
			c_void_p(r.ctypes.data),c_void_p(f.ctypes.data),c_void_p(params.spotx.ctypes.data), c_void_p(params.spoty.ctypes.data),
			c_void_p(params.spotrad.ctypes.data), c_void_p(params.spotcont.ctypes.data),c_void_p(planetangle.ctypes.data),c_void_p(tmodel.ctypes.data))					
		return tmodel
		
	def light_curve(self, params):
		"""
		Calculate a model transit light curve.

		:param params: Transit parameters
		:type params: A `TransitParams` instance

		:return: Relative flux 
		:rtype: ndarray

		:Example:

		>>> flux = m.light_curve(params, t)
		"""
		#updates transit params
		self.t0 = params.t0
		self.per = params.per
		self.rp = params.rp
		self.a = params.a
		self.b = params.b
		self.ecc = params.ecc
		self.w = params.w
		self.u = params.u
		self.limb_dark = params.limb_dark
		self.spotx = np.array(params.spotx)
		self.spoty = np.array(params.spoty)
		self.spotrad = np.array(params.spotrad)
		self.spotcont = np.array(params.spotcont)	
		
		#midpoint rule for integration. 
		r = np.linspace(1./(2*self.n_r), 1. - 1./(2*self.n_r), self.n_r)
		#weights: 2.0 times limb darkening times width of integration annulii.
		f = 2.0*self.quadratic_ld(params, r)/self.n_r

		#calculate orbital elements
		eta, xi = self.elements(params)		
		planetx = -xi
		planety = params.b*eta/params.a
		z = np.sqrt(planetx**2 + planety**2)
		
		#calculate planetangle array
		planetangle = np.empty((self.t.size, self.n_r))
		for i in range(self.t.size):
			self.circleangle(params, r, z[i], planetangle[i]) 
			
		return self.integratetransit(params, planetx, planety, z, r, f, planetangle) 
						
class TransitParams(object):
	"""
	Object to store the physical parameters of the transit.

	:param t0: Time of inferior conjunction (in the same unit as t)
	:type t0: float

	:param per: Orbital period (in days)
	:type per: float

	:param rp: Planet radius (in stellar radii)
	:type rp: float

	:param a: Semi-major axis (in stellar radii)
	:type a: float

	:param b: Impact parameter
	:type b: float

	:param ecc: Orbital eccentricity
	:type ecc: float

	:param w: Argument of periapse (in degrees)
	:type w: float

	:param u: List of limb darkening coefficients
	:type u: array_like 

	:param limb_dark: Limb darkening model ("quadratic")
	:type limb_dark: str

	:param spotx: Spot center x-coordinate (in stellar radii in sky-projected coordinate system)
	:type spotx: ndarray
	
	:param spoty: Spot center y-coordinate (in stellar radii in sky-projected coordinate system) 
	:type spoty: ndarray
	
	:param spotrad: Spot radius (in stellar radii) 
	:type spotrad: ndarray
	
	:param spotcont: Spot contrast   
	:type spotcont: ndarray
	"""
	def __init__(self):
		self.t0 = None
		self.per = None
		self.rp = None
		self.a = None
		self.b = None
		self.ecc = None
		self.w = None
		self.u = None
		self.limb_dark = None
		self.spotx = None
		self.spoty = None
		self.spotrad = None
		self.spotcont = None
