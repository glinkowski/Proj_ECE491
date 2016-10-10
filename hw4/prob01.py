import math


######## ######## ####### #######
# PARAMETERS
x0 = np.array( [-0.5, 1.4] )
	# starting point
xstar = np.array( [0, 1] )
	# the exact solution
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# The function f(x)
def f(x) :
	f1 = (x[0] + 3) * (math.pow(x[1] - 7) + 18)
	f2 = math.sin(x[1] * math.exp(x[0]) - 1)
	
	return np.array( [f1, f2] )
#end def ####### ####### ########

# The Jacobian matrix of f(x)
def JacobianF(x) :
	# row 1
	J11 = math.pow(x[1], 3) - 7
	J12 = 3 * math.pow(x[1], 2) * (3 * x[0] + 1)

	# row 2
	temp = x[1] * math.exp(x[0])
	J21 = temp * math.cos(temp - 1)
	J22 = math.exp(x[0]) * math.cos(temp - 1)

	return np.array( [[J11, J12], [J21, J22]] )
#end def ####### ####### ########


######## ######## ####### #######
# PRIMARY FUNCTION

print("")