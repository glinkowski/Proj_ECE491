# QUESTION 3.13

import numpy as np
# import scipy as sp
# from scipy import linalg as spl
# from scipy.sparse import linalg as spl
# import matplotlib.pyplot as pt
# import matplotlib.lines as mlines



######## ######## ####### #######
# PARAMETERS

# values of epsilon to test
eps = [1]

######## ######## ####### #######



######## ######## ####### #######
# ANCILLARY FUNCTIONS

def createSystemMatrices(fe) :
	fB = np.zeros(4)
	fB[0] = 1

	fA = np.zeros( (4,3) )
	fA[0,:] = np.ones(3)
	for i in range(3) :
		fA[i+1,i] = fe

	return fA, fB
#end def ####### ####### ########

def getHouseholderQR(fA) :

	R0 = np.copy(fA)
	(numRows, numCols) = R0.shape

	allH = np.identity(numRows)
	e = np.identity(numRows)

	for n in range(numCols) :
		# find alpha
		alpha = np.linalg.norm(R0[n:numRows,n])
		signAlpha = 1
		if (R0[n,n] > 0) :
			signAlpha = -1
		alpha = alpha * signAlpha
		# print(alpha)

		# define vector v_i
		v = np.subtract( R0[:,n], np.multiply(alpha, e[:,n]) )
		v[0:n] = np.zeros(n)
		# print("v_{}".format(n))
		# print(v)

		# define rotation matrix H_i
		vT = v.reshape(len(v), 1)
		H_i = np.subtract(e, 
			np.multiply( (2 / np.dot(v, vT)), np.multiply(vT, v) ))
		# print("H_{}".format(n))
		# print(H_i)

		# apply this rotation & keep all previous
		R0 = np.dot(H_i, R0)
		# print("H_i * R0")
		# print(R0)
		allH = np.dot(H_i, allH)
	#end loop

	Q = np.transpose(allH)
	Q1 = Q[:,0:numCols]
	R = R0[0:numCols,:]
	return Q1, R
#end def ######## ####### #######




######## ######## ####### #######
# PRIMARY FUNCTION

# solve as ...
#	Normal equations
#	Augmented system
#	Householder QR
#	Givens QR
#	Classical Gram-Schmidt orthogonalization
#	Modified Gram-Schmidt ...
#	Clas. G-S with refinement (run twice)
#	SVD

for e in eps :

	mxA, mxB = createSystemMatrices(e)
	m, n = mxA.shape

#TODO: return true solX as a function of e

	# a) Normal equations
	# using cholesky factorization
	L = np.linalg.cholesky( np.dot(np.transpose(mxA), mxA) )
	z = np.linalg.solve( L, np.dot(np.transpose(mxA), mxB))
	xNormal = np.linalg.solve( np.transpose(L), z)
	print(xNormal)

	# b) Augmented system
	# create augmented A matrix
	bmxA = np.zeros( ((m+n), (m+n)) )
	bmxA[0:m,0:m] = np.identity(4)
	bmxA[0:m,m:(m+n)] = mxA
	bmxA[m:(m+n),0:m] = np.transpose(mxA)
	# create augmented b vector
	bmxB = np.zeros( (m+n) )
	bmxB[0:m] = mxB
	# solve & extract resulting x vector
	rxAug = np.linalg.solve(bmxA, bmxB)
	xAugmented = rxAug[m:(m+n)]
	print(xAugmented)

	# c) Householder QR

	# mxA = np.array( [
	# 	[1, -1, 1],
	# 	[1, -.5, .25],
	# 	[1, 0, 0],
	# 	[1, 0.5, 0.25],
	# 	[1, 1, 1],
	# 	])
	# mxB = np.array([1, 0.5, 0, 0.5, 2.0])

	hQ, hR = getHouseholderQR(mxA)
	xHouseholder = np.linalg.solve(hR, np.dot(np.transpose(hQ), mxB))
	print(xHouseholder)


