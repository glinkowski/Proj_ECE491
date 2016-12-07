# QUESTION 6.17

import math
import numpy as np
import numpy.linalg as npl
import scipy.optimize as sco
import scipy.misc as scm	
import warnings


######## ######## ####### #######
# PARAMETERS

# input data
tVals = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
yVals = [20.00, 51.58, 68.73, 75.46, 74.36, 67.09, 54.73, 37.98, 17.28]

# starting guess
x0 = [5, 4, 3, 2, 1]
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# The function to solve
def f(fx, ft) :
	fA = np.ones( (len(ft), 4) )
	fA[:,1] = ft
	fA[:,2] = np.power(ft, 2)
	fA[:,3] = np.exp( np.multiply(fx[4], ft), dtype=np.float64 )

	fy = np.dot(fA, fx[0:4])
	return fy
#end def ####### ####### ########


# for Part A ### ####### ######## ########
# Residual defined as (y_true - f(x))
def f_residual(fx, ft, fy) :
	fr = np.subtract(fy, f(fx, ft))
	return fr
#end def ####### ####### ########

# objective function for multidimensional unconstrained
#	minimalization (convert residual to scalar value)
def g(fx, ft, fy) :
	fr = f_residual(fx, ft, fy)

	fg = np.dot(fr, fr)
	fg = np.multiply(0.5, fg)
	return fg
#end def ####### ####### ########

# for Part B ### ####### ######## ########
# The Jacobian of f(x, t)
def Jf(fx, ft) :
	# col 1 = 1
	fJ = np.ones( (len(ft), len(fx)) )
	# col 2 = t
	fJ[:,1] = ft
	# col 3 = t^2
	fJ[:,2] = np.power(ft, 2)
	# col 4 = e^(x5 t)
	fJ[:,3] = np.exp( np.multiply(fx[4], ft) )
	# col 5 = t x4 e^(x5 t)
	fJ[:,4] = np.multiply(ft, fx[3])
	fJ[:,4] = np.multiply( fJ[:,4], fJ[:,3] )
	return fJ
#end def ####### ####### ########

# estimate the gradient of g(x)
def g_gradient(fx, ft, fy) :
	# fJ = Jf_hardcode(fx)
	fJ = Jf(fx, ft)
	fJt = np.transpose(fJ)

	fr = f_residual(fx, ft, fy)

	grad = np.dot(fJt, fr)

	return grad
#end def ####### ####### ########

# for Part C ### ####### ######## ########
# Solve with linear least squares for x[0:4], given x[4]
def f_linearLstsqSolve(fx5, ft, fy) :
	fA = np.ones( (len(ft), 4) )
	fA[:,1] = ft
	fA[:,2] = np.power(ft, 2)
	fA[:,3] = np.exp( np.multiply(fx5, ft), dtype=np.float64 )

	fxAll = np.zeros( 5 )
	fx1234 = npl.lstsq(fA, fy)
	# print(fx1234)
	fxAll[0:4] = fx1234[0]
	fxAll[4] = fx5

	return fxAll
#end def ####### ####### ########

# The function to minimize for part C
def solve_PartC(fx5, ft, fy) :
	# solve for full x vector from fx5
	fx = f_linearLstsqSolve(fx5, ft, fy)
	# return the one-dimension residual
	fg = g(fx, ft, fy)
	return fg
#end def ####### ####### ########

# for Part D ### ####### ######## ########
# The function to minimize for part D
def solve_PartD(fx5, ft, fy) :
	# solve for full x vector from fx5
	fx = f_linearLstsqSolve(fx5, ft, fy)
	# return the one-dimension residual
	fg = g_gradient(fx, ft, fy)
	return fg
#end def ####### ####### ########

# for Part E ### ####### ######## ########
# Residual defined as (y_true - f(x))
def f_residual_hardcode(fx) :
	fr = np.subtract(yVals, f(fx, tVals))
	return fr
#end def ####### ####### ########

# The Jacobian matrix of f(x)
def Jf_hardcode(fx) :
	# col 1
	Jc0 = np.ones(len(tVals))
	# col 2
	Jc1 = tVals
	# col 3
	Jc2 = np.power(tVals, 2.0)
	# col 4
	Jc3 = np.exp( np.multiply(tVals, fx[4]) )
	# col 5
	Jc4 = np.multiply(tVals, fx[3])
	Jc4 = np.multiply(Jc4, Jc3)

	Jf = np.zeros( (len(tVals), len(fx)) )
	Jf[:,0] = Jc0
	Jf[:,1] = Jc1
	Jf[:,2] = Jc2
	Jf[:,3] = Jc3
	Jf[:,4] = Jc4
	return Jf
#end def ####### ####### ########

# Newton Method: one iteration
def iterNewton(bfr, bJf, xk) :
	sklstsq = npl.lstsq( bJf(xk), bfr(xk) )
	sk = sklstsq[0]

	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

def solveNewton(bfr, bJf, x0) :
	# stopping criteria
	errTol = 1e-15
	numIters = 32

	# Newton Method: first iteration
	i = 0
	xNew = iterNewton(bfr, bJf, x0)
	xDiff = 2

	# Newton Method: successive iterations
	while(xDiff > errTol) :
		i += 1

		# Check how much the new x vector changes
		xOld = xNew
		xNew = iterNewton(bfr, bJf, xNew)
		xDiff = npl.norm( np.subtract(xNew, xOld), ord=2)

		if i >= numIters :
			break
	#end loop

	# print("Newton iterations: {}, delta: {:.2e}".format(i, xDiff))
	return xNew
#end def ####### ####### ########

# general use ## ####### ######## ########
# Calculate relative error
def getMaxRelErr(base, value) :
	err = np.abs( np.divide( np.subtract(base, value), base))
	errMax = np.amax(err)
	return errMax
#end def ####### ####### ########

# Print results to terminal
def printOutput(fx0, xFound, ft, yOrig) :
	yCalc = f(xFound, ft)
	print("  using x0 = {}".format(fx0))
	print("  found x  = [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
		xFound[0], xFound[1], xFound[2], xFound[3], xFound[4] ))
	stringOut = "  final y = f(t,x) = ["
	count = 0
	for item in yCalc :
		if count == 0 :
			stringOut = stringOut + '{:.2f}'.format(item)
		elif count >= 4 :
			stringOut = stringOut + ', {:.2f},\n\t\t\t'.format(item)
			count = -1
		else :
			stringOut = stringOut + ', {:.2f}'.format(item)
		count += 1
	stringOut = stringOut + ']'
	print(stringOut)
	err = getMaxRelErr(yOrig, yCalc)
	percErr = err * 100
	if percErr > 1.0 :
		print("  max relative error (y, f(t,x)):  {:.2f} %".format( percErr ))
	else :
		print("  max relative error (y, f(t,x)):  {:.5f} %".format( percErr ))

	return
#end def ####### ####### ########

# to suppress warnings thrown up by scipy
def warfunc():
	warnings.warn("deprecated", DeprecationWarning)
	warnings.warn("runtime", RuntimeWarning)
	return
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("\n\n>>>> Part A >>>>")

# scipy throws up a runtime warning for certain x0
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	warfunc()

	result = sco.minimize(g, x0, args=(tVals, yVals))
	x_partA = result.x
	print("\nwith objective function g(x) ------------------------")
	printOutput(x0, x_partA, tVals, yVals)
#end with


print("\n\n>>>> Part B >>>>")

result = sco.root(g_gradient, x0, args=(tVals, yVals))
print("\nwith estimated gradient of g(x) ---------------------")
x_partB = result.x
printOutput(x0, x_partB, tVals, yVals)


print("\n\n>>>> Part C >>>>")

result = sco.minimize(solve_PartC, x0[4], args=(tVals, yVals))
x5 = result.x
x_partC = f_linearLstsqSolve(x5, tVals, yVals)
print("\nwith linear least squares ---------------------------")
printOutput(x0, x_partC, tVals, yVals)


print("\n\n>>>> Part D >>>>")

result = sco.root(solve_PartD, x0[4], args=(tVals, yVals))
print("\nwith estimated gradient of g(x) ---------------------")
x_partD = f_linearLstsqSolve(result.x, tVals, yVals)
printOutput(x0, x_partD, tVals, yVals)


print("\n\n>>>> Part E >>>>")

result = solveNewton(f_residual_hardcode, Jf_hardcode, x0)
x_partE = result
print("\nby Gauss-Newton method ------------------------------")
printOutput(x0, x_partE, tVals, yVals)


print("\n")