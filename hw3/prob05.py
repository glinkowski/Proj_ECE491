import math as ma
import numpy as np
import matplotlib.pyplot as pt


######## ######## ####### #######
# PARAMETERS
numIters = 10
	# number of iterations over which to collect error
startVal = 2.5
	# x-value at which to begin the first iteration
trueVal = 2.0
	# the expected root (one we're trying to find)
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS
# 	the f(x) & the four g-functions

def f(x) :
	y = (x * x) - (3 * x) + 2
	return y
#end def ####### ######## ########

def g1(x) :
	y = (x * x) + 2
	y = y / 3
	return y
#end def ####### ######## ########
def g1p(x) :
	return (2/3) * x
#end def ####### ######## ########

def g2(x) :
	y = (3*x) - 2
	y = ma.sqrt(y)
	return y
#end def ####### ######## ########
def g2p(x) :
	return (3/2) / ma.sqrt((3*x) - 2)
#end def ####### ######## ########

def g3(x) :
	if x != 0 :
		y = 3 - (2 / x)
	else :
		x = np.inf
	return y
#end def ####### ######## ########
def g3p(x) :
	return 2 / (x*x)
#end def ####### ######## ########

def g4(x) :
	y = (x * x) - 2
	y = y / ((2 * x) - 3)
	return y
#end def ####### ######## ########
def g4p(x) :
	return 2 * ((x*x) - (3*x) + 2) / ma.pow( ((2*x) - 3), 2 )
#end def ####### ####### ########


######## ######## ####### #######
# PRIMARY FUNCTION

print("")

# print("Verifying expected results:")
# print("f(2):   {}".format(f(2)))
# print("g1(2):  {}".format(g1(2)))
# print("g1p(2): {:0.3f}".format(g1p(2)))
# print("g2(2):  {}".format(g2(2)))
# print("g2p(2): {}".format(g2p(2)))
# print("g3(2):  {}".format(g3(2)))
# print("g3p(2): {}".format(g3p(2)))
# print("g4(2):  {}".format(g4(2)))
# print("g4p(2): {}".format(g4p(2)))


# Store the convergence error over 10 iterations
trials = np.array( range(1, numIters+1), dtype=np.int)
err1 = np.zeros( (numIters) )
err2 = np.zeros( (numIters) )
err3 = np.zeros( (numIters) )
err4 = np.zeros( (numIters) )

# Store the new x-values at each iteration
xVals = np.array( [startVal, startVal, startVal, startVal, startVal], dtype=np.float64 )

# Perform the iterations
for i in range(numIters) :

	# fY = f(xVals[1])
	# gY = g1(xVals[1])
	# xVals[1] = gY
	# err1[i] = abs( (fY - gY) / fY )

	# fY = f(xVals[2])
	# gY = g2(xVals[2])
	# xVals[2] = gY
	# err1[i] = abs( (fY - gY) / fY )

	# fY = f(xVals[3])
	# gY = g3(xVals[3])
	# xVals[3] = gY
	# err1[i] = abs( (fY - gY) / fY )

	# fY = f(xVals[4])
	# gY = g4(xVals[4])
	# xVals[4] = gY
	# err1[i] = abs( (fY - gY) / fY )

	gY = g1(xVals[1])
	xVals[1] = gY
	err1[i] = abs( (trueVal - gY) / trueVal )

	gY = g2(xVals[2])
	xVals[2] = gY
	err2[i] = abs( (trueVal - gY) / trueVal )

	gY = g3(xVals[3])
	xVals[3] = gY
	err3[i] = abs( (trueVal - gY) / trueVal )

	gY = g4(xVals[4])
	xVals[4] = gY
	err4[i] = abs( (trueVal - gY) / trueVal )
#end loop

print("For {} iterations, the  following roots were found:".format(numIters))
print("  expected: {}".format(trueVal))
print("  g1 = {:3.3e};  g2 = {:0.5f};  g3 = {:0.5f};  g4 = {:0.5f}".format(
	xVals[1], xVals[2], xVals[3], xVals[4]))


# Plot the resulting convergence errors
pt.plot(trials, err1, trials, err2, trials, err3, trials, err4)
pt.yscale('log')
pt.title('Convergence rate of four g(x) functions')
pt.xlabel('iterations')
pt.ylabel('Log10( relative error )')
pt.legend(['g1(x)', 'g2(x)', 'g3(x)', 'g4(x)'], loc=0)
pt.show()