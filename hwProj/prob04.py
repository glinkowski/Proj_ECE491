# QUESTION 6.13

import math
import numpy as np
import numpy.linalg as npl
# from scipy.optimize import root
# from scipy.optimize import broyden1
# import matplotlib.pyplot as pt


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
	f1 = x[0]**2 + x[1]**2 + (x[0] * x[1]) - 2
	f2 = math.sin(x[0])**2 - 0.5
	f3 = math.cos(x[1])**2 - 0.1

	return np.array( [f1, f2, f3] )
#end def ####### ####### ########

# The Jacobian matrix of second f(x)
def Jf2(x) :
	# row 1
	J11 = 2 * x[0] + x[1]
	J12 = 2 * x[1] + x[0]

	# row 2
	J21 = 2 * (math.cos(x[0])) * math.sin(x[0])
	J22 = 0.0

	# row 3
	J31 = 0.0
	J32 = -2 * (math.cos(x[1])) * math.sin(x[1])

	return np.array( [[J11, J12], [J21, J22], [J31, J32]] )
#end def ####### ####### ########

# Newton Method: one iteration
def iterNewton(bf, bJf, xk) :
	# sklstsq = npl.lstsq( bJf(xk), -bf(xk) )
	# sk = sklstsq[0]

	q, r = npl.qr( bJf(xk) )
	sk = npl.solve(r, np.dot( np.transpose(q), -bf(xk)) )
	
	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

def solveNewton(bf, bJf, x0) :
	# Newton Method: first iteration
	i = 0
	xNew = iterNewton(bf, bJf, x0)
	# err = la.norm( np.subtract(xNew, xstar), ord=2)
	# iN.append(i)
	# eN.append(err)
	xDiff = 2

	# Newton Method: successive iterations
	# errPrev = 0.0
	#while(err != errPrev) :
	# while(err > errTol) :
	while(xDiff > errTol) :
		# print("||{} - {}|| = {:1.3e}".format(xNew, xstar, err))
		i += 1
		# errPrev = err

		xOld = xNew
		xNew = iterNewton(bf, bJf, xNew)
		xDiff = npl.norm( np.subtract(xNew, xOld), ord=2)
		# err = la.norm( np.subtract(xNew, xstar), ord=2)
		# iN.append(i)
		# eN.append(err)
		if i >= numIters :
			break
	#end loop

	print("Newton iterations: {}, delta: {:.2e}".format(i, xDiff))
	return xNew
#end def ####### ####### ########


# Broyden Method: initial B0
def initBroyden(bJf, x0) :
	return bJf(x0)

# Broyden Method: one iteration
def iterBroyden(bf, xk, Bk) :

#TODO: a more accurate way to get sk ??
	sklstsq = npl.lstsq( Bk, -bf(xk) )
	sk = sklstsq[0]
	# print(Bk)
	# print(sk)
	xk1 = np.add(xk, sk[0])
	yk = bf(xk1) - bf(xk)

	Btemp = np.subtract(yk, np.dot(Bk, sk))
	stemp = np.dot(sk, sk)

	# catch a divide-by-zero error
	if stemp == 0 :
		Bk1 = np.add(Bk, np.inf)
	else :
		Bnumer = np.multiply(Btemp.reshape((Btemp.size,1)), sk.reshape((1,sk.size)))
		Bk1 = Bk + np.divide( Bnumer, stemp )
	#end if

	return xk1, Bk1
#end def ####### ####### ########

def solveBroyden( bf, bJf, bx0 ) :
	# Broyden Method: initialization
	BNew = initBroyden(bJf, x0)
	xNew = x0
	xDiff = 2

	i = -1
	errPrev = 0.0
	err = 1
	#while(err != errPrev) :
	while(xDiff > errTol) :
		# pass
		# # print("||{} - {}|| = {:1.3e}".format(xNew, xstar, err))
		i += 1
		# errPrev = err

		xOld = xNew
		xNew, BNew = iterBroyden(bf, xNew, BNew)
		xDiff = npl.norm( np.subtract(xNew, xOld), ord=2)

		# err = la.norm( np.subtract(xNew, xstar), ord=2)
		# # iB.append(i)
		# # eB.append(err)

		# print((i, xDiff))
		if i >= numIters :
			break
	#end loop

	print("Broyden iterations: {}, delta: {:.2e}".format(i, xDiff))
	return xNew
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("\n\n>>>> Part A >>>>")

x0 = [2, 2]
print("\nBroyden results ---------------------")
xFinal = solveBroyden(f1, Jf1, x0)
# print( (xFinal, f1(xFinal)) )
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))

print("\nNewton results ---------------------")
xFinal = solveNewton(f1, Jf1, x0)
# print( (xFinal, f1(xFinal)) )
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))


x0 = [-1, -1]
print("\nBroyden results ---------------------")
xFinal = solveBroyden(f1, Jf1, x0)
# print( (xFinal, f1(xFinal)) )
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))

print("\nNewton results ---------------------")
xFinal = solveNewton(f1, Jf1, x0)
# print( (xFinal, f1(xFinal)) )
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))


print("\n\n>>>> Part B >>>>")

x0 = [2, 2]
print("\nBroyden results ---------------------")
xFinal = solveBroyden(f2, Jf2, x0)
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))

print("\nNewton results ---------------------")
xFinal = solveNewton(f2, Jf2, x0)
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))


x0 = [1, -1]
print("\nBroyden results ---------------------")
xFinal = solveBroyden(f2, Jf2, x0)
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))

print("\nNewton results ---------------------")
xFinal = solveNewton(f2, Jf2, x0)
fx = f1(xFinal)
print("  using x0 = {}".format(x0))
print("  found x = [{:.4f}, {:.4f}]".format(xFinal[0], xFinal[1]))
print("  result f(x) = [{:.4f}, {:.4f}, {:.4f}]".format(fx[0], fx[1], fx[2]))


# xFinal = root(f1, x0, method='broyden1', jac=Jf1)
# print( (xFinal, f1(xFinal)) )

# x0 = [2, 2, 0]
# xFinal = broyden1( f1, x0 )
# print( (xFinal, f1(xFinal)) )