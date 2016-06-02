import os
import sys
import scipy
import random
import matplotlib
import numpy as np
import itertools
import pandas as pd
from scipy.stats import beta
from __future__ import division
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import numpy as np
output_notebook()

#############################################################
#--- Bare bones pbox and cbox Python code.               ---#
#---                                                     ---#
#--- For use in developing basal models for propegating  ---#
#--- uncertainty through calculations using genomic data ---#
#---                                                     ---#
#--- See page 157, O'Rawe 201X.                          ---#
#############################################################

def rep(x, times):
	return [x for i in range(times)]

def cartesian_p(*itrs):
   product = list(itertools.product(*itrs))
   return pd.DataFrame(product)

class pbox(object):
	steps = 100 # managable cartesian product
	
	def __init__(self,left = [], right = []):
		self.left = left
		self.right = right
	
	def __mul__(self, b):
		if isinstance(b, self.__class__):
			c_l = cartesian_p(self.left,b.left).apply(lambda x: x[0] * x[1],axis=1).values
			c_r = cartesian_p(self.right,b.right).apply(lambda x: x[0] * x[1],axis=1).values
			
			# Condensation via Williamson and Downs 1990 
			m = self.steps
			p = b.steps
			n = self.steps
			L = int(m * p / n)
			k = np.array(range(1,n+1))
			new_left = np.sort(c_l)[(k-1)*L]
			new_right = np.sort(c_r)[(k-1)*L + L - 1]
			new_pbox = pbox(left=new_left, right=new_right)
			
			return new_pbox
			
		elif isinstance(b, int) or isinstance(b, float):
			self.left = self.left * b
			self.right = self.right * b
			return self

def cbox_nom(k, n):
	a = k
	b = n - k + 1
	c = k + 1
	d = n - k
	cbox = pbox()
	steps = cbox.steps
	cbox_beta = beta.ppf([xy/steps for xy in [x+1 for x in range(steps)]], 
			[rep(a,steps),rep(c,steps)], 
			[rep(b,steps),rep(d,steps)])
	cbox.left = cbox_beta[0]
	cbox.right = cbox_beta[1]
	return cbox