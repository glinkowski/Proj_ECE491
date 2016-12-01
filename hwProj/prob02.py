# QUESTION 3.8

import numpy as np
from scipy import linalg as spl


######## ######## ####### #######
# PARAMETERS

# suggested params
m = 21
n = 12
eps = 1e-6
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

def generateTMatrix(fm, fn) :
	t = np.arange(fm)
	t = np.divide(t, (fm - 1))
	t = t.reshape(fm,1)
	t = np.repeat(t, (fn), axis=1)
	return t
#end def ####### ####### ########

# def generateXSingle(fn, fj) :
# 	x = np.multiply(np.ones(fn), fj)
# 	return x
# #end def ####### ####### ########

# def generateXMatrix(fm, fn) :
# 	one = np.ones( (1, fn) )
# 	j = np.add(np.arange(fn), 1)
# 	j = j.reshape( (fn, 1) )
# 	x = np.multiply(one, j)
# 	return x
# #end def ####### ####### ########

def generateYTData(fm, fn, fxj) :

	# Create the m x n matrix for t
	t = generateTMatrix(fm, fn)
	# Take t to the appropriate exponent
	#	ie: t^0, t^1, ... , t^(n-1)
	tVals = np.array( (t.shape) )
	tExp = np.arange(fn)
	tExp = tExp.reshape( (1, fn) )
	tVals = np.power(t, tExp)

	# Create x coefficients, fxj is either a scalar or
	#	a vector of length fn
	xVals = np.multiply( np.ones((fm, fn)), fxj )
	# xVals = np.multiply( np.ones((fm, fn)), np.arange(fn) )
	# print(xVals)

	yTemp = np.multiply(xVals, tVals)
	y = np.sum(yTemp, axis=1)

	return tVals, y
#end def ####### ####### ########

def perturbY(fy, fe) :
	u = np.random.uniform(0, 1, len(fy))
	# print(u)
	yAdd = np.multiply( np.subtract( np.multiply(2, u), 1), fe)
	# print(yAdd)
	yp = np.add(fy, yAdd)
	return yp
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION
 
# Test with several versions of x_j
xjList = [1]
xjList.append(2)
xjList.append(-1.5)
xjList.append(np.arange(1, n+1))
xjList.append(np.random.uniform(-2, 2, n))

for xjOrig in xjList :

	# Generate the A & B matrices & expected xj
	# xjOrig = 1
	mxA, mxB = generateYTData(m, n, xjOrig)
	# print(mxA)
	# print(mxB)
	mxBp = perturbY(mxB, eps)
	# print(mxBp)


	# Solve for xj using QR factorization
	q, r = np.linalg.qr(mxA)
	xjQR = np.linalg.solve(r, np.dot( np.transpose(q), mxB) )
	xjQRp = np.linalg.solve(r, np.dot( np.transpose(q), mxBp) )
	# print(xjQR)
	# print(xjQRp)
	qrMeanErr = np.mean(np.abs( np.divide(np.subtract(xjQR, xjOrig), xjOrig)))
	qrpMeanErr = np.mean(np.abs( np.divide(np.subtract(xjQRp, xjOrig), xjOrig)))
	qrpDiff = np.subtract(qrpMeanErr, qrMeanErr)
	# print(qrMeanErr)
	# print(qrpMeanErr)
	# print(qrpDiff)

	print("\n----------------------------------------------------------")
	print("Results for xj = {}".format(xjOrig))
	print("Using QR factorization, rel error = {}".format(qrMeanErr))
	print(" applied to perturbed data, error = {}".format(qrpMeanErr))
	print("            for a difference of ... {}".format(qrpDiff))



	# Solve for xj using Cholesky factorization

	# mxA, mxB = generateYTData(3, 5, xjOrig)
	# print(mxA)
	# print(np.dot(mxA, np.transpose(mxA)))
	# print(np.linalg.eigvals(np.dot(mxA, np.transpose(mxA))))

	# get cholesky factorization
	mxAtA = np.dot(np.transpose(mxA), mxA)
	# print(mxAtA)
	# print(np.linalg.eigvals(mxAtA))
	L = np.linalg.cholesky( mxAtA )
	z = np.linalg.solve( L, np.dot(np.transpose(mxA), mxB))
	xjCh = np.linalg.solve( np.transpose(L), z)
	chMeanErr = np.mean(np.abs( np.divide(np.subtract(xjCh, xjOrig), xjOrig)))

	z = np.linalg.solve( L, np.dot(np.transpose(mxA), mxBp))
	xjChp = np.linalg.solve( np.transpose(L), z)
	chpMeanErr = np.mean(np.abs( np.divide(np.subtract(xjChp, xjOrig), xjOrig)))

	chpDiff = np.subtract(chpMeanErr, chMeanErr)

	print("")
	print("Using Cholesky,    relative error = {}".format(chMeanErr))
	print(" applied to perturbed data, error = {}".format(chpMeanErr))
	print("            for a difference of ... {}".format(chpDiff))

#end loop


print("\n")