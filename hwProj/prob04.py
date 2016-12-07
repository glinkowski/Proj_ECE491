# QUESTION 6.13

import math
import numpy as np
import numpy.linalg as npl
import scipy.optimize as sco


######## ######## ####### #######
# PARAMETERS

# stopping criteria
errTol = 1e-15
numIters = 64
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# The first function f(x)
def f1(x) :
	f1 = x[0]**2 + x[1]**2 - 2
	f2 = (x[0] - 2)**2 + x[1]**2 - 2
	f3 = (x[0] - 1)**2 + x[1]**2 - 9

	return np.array( [f1, f2, f3] )
#end def ####### ####### ########

# The Jacobian matrix of first f(x)
def Jf1(x) :
	# row 1
	J11 = 2 * x[0]
	J12 = 2 * x[1]
	# row 2
	J21 = 2 * (x[0] - 2)
	J22 = 2 * x[1]
	# row 3
	J31 = 2 * (x[0] - 1)
	J32 = 2 * x[1]

	return np.array( [[J11, J12], [J21, J22], [J31, J32]] )
#end def ####### ####### ########

# The second function f(x)
def f2(x) :
	f1 = x[0]**2 + x[1]**2 + (x[0] * x[1])# - 0.5
	f2 = math.sin(x[0])**2# - 0.5
	f3 = math.cos(x[1])**2# - 0.5

	return np.array( [f1, f2, f3] )
#end def ####### ####### ########

# The Jacobian matrix of second f(x)
def Jf2(x) :
	# row 1
	J11 = 2 * x[0] + x[1]
	J12 = 2 * x[1] + x[0]
	# row 2
	J21 = 2 * math.cos(x[0]) * math.sin(x[0])
	J22 = 0.0
	# row 3
	J31 = 0.0
	J32 = -2 * math.cos(x[1]) * math.sin(x[1])

	return np.array( [[J11, J12], [J21, J22], [J31, J32]] )
#end def ####### ####### ########

# Newton Method: one iteration
def iterNewton(bf, bJf, xk) :
	q, r = npl.qr( bJf(xk) )
	sk = npl.solve(r, np.dot( np.transpose(q), -bf(xk)) )
	
	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

def solveNewton(bf, bJf, x0) :
	# Newton Method: first iteration
	i = 0
	xNew = iterNewton(bf, bJf, x0)
	xDiff = 2

	# Newton Method: successive iterations
	while(xDiff > errTol) :
		i += 1

		xOld = xNew
		xNew = iterNewton(bf, bJf, xNew)
		xDiff = npl.norm( np.subtract(xNew, xOld), ord=2)

		if i >= numIters :
			break
	#end loop

	print("Newton iterations: {}, delta: {:.2e}".format(i, xDiff))
	return xNew
#end def ####### ####### ########

def g1(fx) :
	fr = -f1(fx)
	fg = np.multiply(0.5, np.dot(fr, fr))
	return fg
#end def ####### ####### ########

def g2(fx) :
	fr = -f2(fx)
	fg = np.multiply(0.5, np.dot(fr, fr))
	return fg
#end def ####### ####### ########

def printOutput(x0, xFound, yCalc) :
	print("  using x0 = {}".format(x0))
	print("  found x = [{:.4f}, {:.4f}]".format(xFound[0], xFound[1]))
	print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(
		yCalc[0], yCalc[1], yCalc[2]))
	return
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("\n\n>>>> Part A >>>>")

x0 = [2, 2]

print("\nNewton results ---------------------")
xFinal = solveNewton(f1, Jf1, x0)
printOutput(x0, xFinal, f1(xFinal))

result = sco.minimize(g1, x0)
xFinal = result.x
print("\nminimize from SciPy library --------")
printOutput(x0, xFinal, f1(xFinal))


x0 = [-1, -1]

print("\nNewton results ---------------------")
xFinal = solveNewton(f1, Jf1, x0)
printOutput(x0, xFinal, f1(xFinal))

result = sco.minimize(g1, x0)
xFinal = result.x
print("\nminimize from SciPy library --------")
printOutput(x0, xFinal, f1(xFinal))


print("\n\n>>>> Part B >>>>")

x0 = [2, 2]

print("\nNewton results ---------------------")
xFinal = solveNewton(f2, Jf2, x0)
printOutput(x0, xFinal, f1(xFinal))

result = sco.minimize(g2, x0)
xFinal = result.x
print("\nminimize from SciPy library --------")
printOutput(x0, xFinal, f1(xFinal))


x0 = [1, -1]

print("\nNewton results ---------------------")
xFinal = solveNewton(f2, Jf2, x0)
printOutput(x0, xFinal, f1(xFinal))

result = sco.minimize(g2, x0)
xFinal = result.x
print("\nminimize from SciPy library --------")
printOutput(x0, xFinal, f1(xFinal))


print("\n")