import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt


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
def Jf(x) :
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
	sk = la.solve( Jf(xk), -f(xk) )
	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

# Broyden Method: initial B0
def initBroyden(x0) :
	return Jf(x0)

# Broyden Method: one iteration
def iterBroyden(xk, Bk) :
	sk = la.solve( Bk, -f(xk) )
	xk1 = xk + sk
	yk = f(xk1) - f(xk)

	Btemp = np.subtract(yk, np.dot(Bk, sk))
	stemp = np.dot(sk, sk)

	# catch a divide-by-zero error
	if stemp == 0 :
		Bk1 = np.add(Bk, np.inf)
	else :
		Bnumer = np.multiply(Btemp.reshape((2,1)), sk.reshape((1,2)))
		Bk1 = Bk + np.divide( Bnumer, stemp )
	#end if

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


# Newton Method: first iteration
i = 1
xNew = iterNewton(x0)
err = la.norm( np.subtract(xNew, xstar), ord=2)
iN.append(i)
eN.append(err)

# Newton Method: successive iterations
errPrev = 0.0
while(err != errPrev) :
	# print("||{} - {}|| = {:1.3e}".format(xNew, xstar, err))
	i += 1
	errPrev = err

	xNew = iterNewton(xNew)
	err = la.norm( np.subtract(xNew, xstar), ord=2)
	iN.append(i)
	eN.append(err)
#end loop

# print("||{} - {}|| => error = {:1.3e}".format(xNew, xstar, err))
# print("{} iterations".format(i))


# Broyden Method: initialization
BNew = initBroyden(x0)
xNew = x0

i = 0
errPrev = 0.0
err = 1
while(err != errPrev) :
	# print("||{} - {}|| = {:1.3e}".format(xNew, xstar, err))
	i += 1
	errPrev = err

	xNew, BNew = iterBroyden(xNew, BNew)
	err = la.norm( np.subtract(xNew, xstar), ord=2)
	iB.append(i)
	eB.append(err)

	if i > 15 :
		break
#end loop

# print("||{} - {}|| => error = {:1.3e}".format(xNew, xstar, err))
# print("{} iterations".format(i))


# Printed output
print("Iterations until error converges...")
print("Newton:  {:2d} iterations; error = {:.3e}".format(len(eN), eN[len(eN)-1]))
print("Broyden: {:2d} iterations; error = {:.3e}".format(len(eB), eB[len(eB)-1]))


# Plot the error vs iterations
pt.plot(iN, eN, iB, eB)
pt.legend( ['Newton method', 'Broyden method'] )
pt.title('Error convergence of Newton & Broyden methods')
pt.ylabel('log10( error )')
pt.xlabel('iterations')
pt.yscale('log')
pt.show()