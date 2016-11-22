# QUESTION 3.8

import numpy as np
# import scipy as sp
from scipy import linalg as sl
from scipy.sparse import linalg as spl
import matplotlib.pyplot as pt
import matplotlib.lines as mlines



######## ######## ####### #######
# PARAMETERS

# positional data
xPos = np.array(
	[1.02, 0.95, 0.87, 0.77, 0.67, 0.56, 0.44, 0.30, 0.16, 0.01] )
yPos = np.array(
	[0.39, 0.32, 0.27, 0.22, 0.18, 0.15, 0.13, 0.12, 0.13, 0.15] )

# amount by which to perturb data
ptbRange = 0.05
#NOTE: suggested range = 0.005
#	produced little noticable difference

######## ######## ####### #######



######## ######## ####### #######
# ANCILLARY FUNCTIONS

def generateT(m) :
	t = np.zeros( (m) )
	for i in range(m) :
		t[i] = ((i+1) - 1) / (m - 1)
	return t
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION



# Part A ######## ####### #######
print("\n>>>> Part A >>>>")

