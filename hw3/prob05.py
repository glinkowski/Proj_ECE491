import numpy as np
import math as ma



# Define the f(x) & the four g-functions

def f(x) :
	out = (x * x) - (3 * x) + 2
	return out
#end def ####### ####### ########

def g1(x) :
	out = (x * x) + 2
	out = out / 3
	return out
#end def ####### ####### ########
def g1p(x) :
	return (2/3) * x
#end def ####### ####### ########

def g2(x) :
	out = (3*x) - 2
	out = ma.sqrt(out)
	return out
#end def ####### ####### ########
def g2p(x) :
	return (3/2) / ma.sqrt((3*x) - 2)
#end def ####### ####### ########

def g3(x) :
	if x != 0 :
		out = 3 - (2 / x)
	else :
		x = np.inf
	return out
#end def ####### ####### ########
def g3p(x) :
	return 2 / (x*x)
#end def ####### ####### ########

def g4(x) :
	out = (x * x) - 2
	out = out / ((2 * x) - 3)
	return out
#end def ####### ####### ########
def g4p(x) :
	return 2 * ((x*x) - (3*x) + 2) / ma.pow( ((2*x) - 3), 2 )
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION


