import math
import numpy as np
import numpy.linalg as la


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
	f1 = (x[0] + 3) * (math.pow(x[1],3) - 7) + 18
	f2 = math.sin(x[1] * math.exp(x[0]) - 1)

	return np.array( [f1, f2] )
#end def ####### ####### ########

# The Jacobian matrix of f(x)
def JacobianF(x) :
	# row 1
	J11 = math.pow(x[1], 3) - 7
	J12 = 3 * math.pow(x[1], 2) * (x[0] + 3)

	# row 2
	temp = x[1] * math.exp(x[0])
	J21 = temp * math.cos(temp - 1)
	J22 = math.exp(x[0]) * math.cos(temp - 1)

	return np.array( [[J11, J12], [J21, J22]] )
#end def ####### ####### ########

# Newton Method: one iteration
def iterNewton(xk) :
	sk = la.solve( JacobianF(xk), -f(xk) )
	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

# Broyden Method: initial B0
def initBroyden(x0) :
	return JacobianF(x0)

# Broyden Method: one iteration
def iterBroyden(xk, Bk) :
	sk = la.solve( Bk, -f(xk) )
	xk1 = xk + sk
	yk = f(xk1) - f(xk)
	Bk1 = Bk + ( (yk - np.dot(Bk, sk)) ) / np.dot(sk, sk)

	return xk1, Bk1
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("")

# The iterations for Newton/Broyden (x-axis of plot)
iN = list()
iB = list()

# The error for Newton/Broyden (y-axis of plot)
eN = list()
eB = list()


eThresh = 60

# Newton Method: first iteration
i = 1
xNew = iterNewton(x0)
#err = math.sqrt( math.pow( (xNew - xstar), 2) )
err = la.norm( np.subtract(xNew, xstar), ord=2)
iN.append(i)
eN.append(err)

# Newton method: successive iterations
while(err > eThresh) :
	i += 1

	xNew = iterNewton(xNew)
#	err = math.sqrt( math.pow( (xNew - xstar), 2) )
	err = la.norm( np.subtract(xNew, xstar), ord=2)
	iN.append(i)
	eN.append(err)

	print("||{0} - {1}|| = {2:1.3f} < {3}".format(xNew, xstar, err, eThresh))
#end loop

print("||{} - {}|| = {:1.3f} < {}".format(xNew, xstar, err, eThresh))
print("{} iterations".format(i))
