# QUESTION 6.17

import math
import numpy as np
import numpy.linalg as npl
# from scipy.optimize import minimize
import scipy.optimize as sco
import scipy.misc as scm

#TODO: suppress warnings from scipy

######## ######## ####### #######
# PARAMETERS

# input data
tVals = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
yVals = [20.00, 51.58, 68.73, 75.46, 74.36, 67.09, 54.73, 37.98, 17.28]

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

# Residual defined as (y_true - f(x))
def f_residual(fx, ft, fy) :
	fr = np.subtract(fy, f(fx, ft))

	return fr
#end def ####### ####### ########

# objective function for
#	multidimensional unconstrained minimalization
def g(fx, ft, fy) :
	fr = f_residual(fx, ft, fy)

	# print(fx)

	fg = np.dot(fr, fr)
	fg = np.multiply(0.5, fg)

	# print(fg)

	return fg
#end def ####### ####### ########

# estimate the gradient of g(x)
def g_gradient(fx, ft, fy) :
	delta = 1e-3

	# print(fx)

	grad = np.zeros( len(fx), dtype=np.float64 )
	for i in range(len(fx)) :
		fxPlus = fx
		fxPlus[i] = fx[i] + delta
		fxMinus = fx
		fxMinus[i] = fx[i] - delta

		# print((f(fxPlus, ft), f(fxMinus, ft)))
		grad[i] = (g(fxPlus, ft, fy) - g(fxMinus, ft, fy)) / (2*delta)

	#end loop

	# print(grad)

	return grad
#end def ####### ####### ########

def solve_partB(fx, ft, fy) :
	deriv = scm.derivative(g, fx, args=(ft, fy))
	# print(grad)
	return np.abs(deriv)
#end def ####### ####### ########

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
	# get full x vector from fx5
	fx = f_linearLstsqSolve(fx5, ft, fy)

	# return the one-dimension residual
	# fr = f_residual(fx, ft, fy)
	fg = g(fx, ft, fy)

	return fg
#end def ####### ####### ########

# Newton Method: one iteration
def iterNewton(bfr, bJf, xk) :
	q, r = npl.qr( bJf(xk) )
	sk = npl.solve(r, np.dot( np.transpose(q), bfr(xk)) )

	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

# Residual defined as (y_true - f(x))
def f_residual_hardcode(fx) :
	fr = np.subtract(yVals, f(fx, tVals))

	return fr
#end def ####### ####### ########

# The Jacobian matrix of f(x)
def Jf_hardcode(fx) :
	# Create each column of the jacobian

	Jc0 = np.ones(len(tVals))

	Jc1 = tVals

	Jc2 = np.power(tVals, 2.0)

	Jc3 = np.exp( np.multiply(tVals, fx[4]) )

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
	# print(xk)
	# print(bJf(xk))

	# q, r = npl.qr( bJf(xk) )
	# # print(np.dot( np.transpose(q), bfr(xk)))
	# sk = npl.solve(r, np.dot( np.transpose(q), bfr(xk)) )

	sklstsq = npl.lstsq( bJf(xk), bfr(xk) )
	sk = sklstsq[0]

	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

def solveNewton(bfr, bJf, x0) :

	# stopping criteria
	errTol = 1e-15
	numIters = 32
	numIters = 4

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
	# print("  final y = f(t,x) = [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
	# 	yCalc[0], yCalc[1], yCalc[2], yCalc[3], yCalc[4] ))
	# print("  max relative error (y, f(t,x)):  {:.3e}".format(
	# 	getMaxRelErr(yOrig, yCalc) ))
	err = getMaxRelErr(yOrig, yCalc)
	percErr = err * 100
	if percErr > 1.0 :
		print("  max relative error (y, f(t,x)):  {:.2f} %".format( percErr ))
	else :
		print("  max relative error (y, f(t,x)):  {:.5f} %".format( percErr ))

	return
#end def ####### ####### ########


######## ######## ####### #######
# PRIMARY FUNCTION


print("\n\n>>>> Part A >>>>")

# x0 = [1, 2, 3, 4, 5]
x0 = [5, 4, 3, 2, 1]

result = sco.minimize(g, x0, args=(tVals, yVals))
x_partA = result.x
# y_partA = f(x_partA, tVals)
# print(x_partA)
# print(f(result.x, tVals))
print("\nwith objective function g(x) ------------------------")
# print("  using x0 = {}".format(x0))
# print("  found x  = [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
# 	x_partA[0], x_partA[1], x_partA[2], x_partA[3], x_partA[4]))
# print("  final y = f(t,x) = {}".format(y_partA))
# # print("  norm diff b/t this and expected: {:.3e}".format(
# # 	npl.norm( np.subtract(yVals, y_partA), ord=2) ))
# print("  max relative error (y, f(t,x)):  {:.3e}".format(
# 	getMaxRelErr(yVals, y_partA) ))
printOutput(x0, x_partA, tVals, yVals)


print("\n\n>>>> Part B >>>>")
#TODO: try gradient of f(t,x) ??

x0 = [5, 4, 3, 2, 1]
# x0 = [-500, -400, -300, 200, 1]

# result = sco.root(g_gradient, x0, args=(tVals, yVals))
result = sco.minimize(solve_partB, x0, args=(tVals, yVals))
x_partB = result.x
# y_partB = f(x_partB, tVals)
print("\nwith estimated gradient of g(x) ---------------------")
# print("  using x0 = {}".format(x0))
# print("  found x  = [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
# 	x_partB[0], x_partB[1], x_partB[2], x_partB[3], x_partB[4]))
# print("  final y = f(t,x) = {}".format(y_partB))
# print("  max relative error (y, f(t,x)):  {:.3e}".format(
# 	getMaxRelErr(yVals, y_partB) ))
printOutput(x0, x_partB, tVals, yVals)


print("\n\n>>>> Part C >>>>")

x0 = [1]

result = sco.minimize(solve_PartC, x0, args=(tVals, yVals))
x5 = result.x
x_partC = f_linearLstsqSolve(x5, tVals, yVals)
# y_partC = f(x_partC, tVals)
print("\nwith linear least squares ---------------------------")
# print("  using x5 = {}".format(x0))
# print("  found x  = [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
# 	x_partC[0], x_partC[1], x_partC[2], x_partC[3], x_partC[4] ))
# print("  final y = f(t,x) = [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
# 	y_partC[0], y_partC[1], y_partC[2], y_partC[3], y_partC[4] ))
# print("  max relative error (y, f(t,x)):  {:.3e}".format(
# 	getMaxRelErr(yVals, y_partC) ))
printOutput(x0, x_partC, tVals, yVals)



#TODO: ?? What is part D ??





print("\n\n>>>> Part E >>>>")

x0 = [5, 4, 3, 2, 1]

result = solveNewton(f_residual_hardcode, Jf_hardcode, x0)
x_partE = result
print("\nby Gauss-Newton method ------------------------------")
printOutput(x0, x_partE, tVals, yVals)
