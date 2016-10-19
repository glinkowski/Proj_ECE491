from scipy.optimize import root
import numpy as np
import math


######## ######## ####### #######
# PARAMETERS
s = 3.5
	# the mass & radius constant
x0 = np.array( [0, 0] )
	# starting point
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# The function f(x) for the ground state
def f0(x) :
	f1 = -(x[0] / math.tan(x[0])) - x[1]
	f2 = s**2 - x[0]**2 - x[1]**2

	return np.array( [f1, f2] )
#end def ####### ####### ########
# The Jacobian of f0(x)
def Jf0(x) :
	j11 = -(1 / math.tan(x[0])) + x[0] * (1 / (math.sin(x[0])**2))
	j12 = -1
	j21 = -2*(x[0])
	j22 = -2*(x[1])

	return np.array( [ [j11, j12], [j21, j22] ] )
#end def ####### ####### ########

# The function f(x) for the first state
def f1(x) :
	f1 = -1/(x[0] * math.tan(x[0])) + x[0]**(-2) + x[1]**(-1) + x[1]**(-2)
	f2 = s**2 - x[0]**2 - x[1]**2

	return np.array( [f1, f2] )
#end def ####### ####### ########
# The Jacobian of f0(x)
def Jf1(x) :
# 	j11 = x[0]**(-2) / math.tan(x[0])
# 	j11 += 1/(x[0] * (math.sin(x[0])**2))
# 	j11+= - 2*(x[0]**(-3))
	j11 = x[0]**(-2) / math.tan(x[0]) + 1/(x[0] * (math.sin(x[0])**2)) - 2*(x[0]**(-3))
	j12 = -x[1]**(-2) - 2*(x[1]**(-3))
	j21 = -2*(x[0])
	j22 = -2*(x[1])

	return np.array( [ [j11, j12], [j21, j22] ] )
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("")

x0 = [2, 2]
rootSol = root(f0, x0, method='anderson', jac=Jf0)
# print(rootSol)

x1 = rootSol.x[0]
y1 = rootSol.x[1]


x0 = [3.5, 0.5]
# rootSol = root(f1, x0, method='hybr', jac=Jf1)
# print(rootSol)

# x2 = rootSol.x[0]
# y2 = rootSol.x[1]


rootSol = root(f1, x0, method='anderson', jac=Jf1)
print(rootSol)

x2 = rootSol.x[0]
y2 = rootSol.x[1]