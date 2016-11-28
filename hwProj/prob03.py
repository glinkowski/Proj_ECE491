# QUESTION 3.13

import numpy as np
# import scipy as sp
# from scipy import linalg as spl
# from scipy.sparse import linalg as spl
# import matplotlib.pyplot as pt
# import matplotlib.lines as mlines
# import sys


######## ######## ####### #######
# PARAMETERS

base = 2.0
precision = 11

# values of epsilon to test
eps = np.array([
	5,
	3,
	2,
	1,
	0.5,
	0.05,
	0.005,
	np.power(base, 1-12),
	# 0.5 * np.power(base, 1-11),
	np.power( np.power(base, 1-24), 0.5 ),
	# np.power( 0.5 * np.power(base, 1-22), 0.5 )
	np.power( np.power(base, 1-24), 0.5 ) - 0.0001,
	# 0.00005,
	# 0.000005
	],
	dtype=np.float64)

# print(np.power(base, 1-precision))
# sys.exit()
######## ######## ####### #######



######## ######## ####### #######
# ANCILLARY FUNCTIONS

def createSystemMatrices(fe) :
	fB = np.zeros(4, dtype=np.float32)
	fB[0] = 1

	fA = np.zeros( (4,3), dtype=np.float32 )
	fA[0,:] = np.ones(3)
	for i in range(3) :
		fA[i+1,i] = fe

	return fA, fB
#end def ####### ####### ########

def getHouseholderQR(fA) :

	R0 = np.copy(fA)
	(numRows, numCols) = R0.shape

	allH = np.identity(numRows, dtype=np.float32)
	e = np.identity(numRows, dtype=np.float32)

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

def origGramSchmidt(fA) :

	A = np.copy(fA)
	(numRows, numCols) = A.shape

	# The arrays to return
	Q = np.zeros( A.shape, dtype=np.float32 )
	R = np.zeros( A.shape, dtype=np.float32 )

	for k in range(numCols) :
		Q[:,k] = A[:,k]
		for j in range(k) :
			R[j,k] = np.dot( Q[:,j], A[:,k] )
			Q[:,k] = np.subtract( Q[:,k], np.multiply(R[j,k], Q[:,j]) )
			# r = np.dot( Q[:,j], A[:,k] )
			# Q[:,k] = np.subtract( Q[:,k], np.multiply(r, Q[:,j]) )
		#end loop
		R[k,k] = np.linalg.norm( Q[:,k] )
		Q[:,k] = np.divide( Q[:,k], R[k,k] )
		# r = np.linalg.norm( Q[:,k] )
		# Q[:,k] = np.divide( Q[:,k], r )
	#end loop

	return Q[:,0:numCols], R[0:numCols,:]
#end def ######## ####### #######

def modGramSchmidt(fA) :

	A = np.copy(fA)
	(numRows, numCols) = A.shape

	# The arrays to return
	Q = np.zeros( A.shape, dtype=np.float32 )
	R = np.zeros( A.shape, dtype=np.float32 )

	for k in range(numCols) :
		R[k,k] = np.linalg.norm(A[:,k])
		Q[:,k] = np.divide(A[:,k], R[k,k])
		for j in range( (k+1), numCols ) :
			R[k,j] = np.dot(Q[:,k],A[:,j])
			A[:,j] = np.subtract( A[:,j], np.multiply(R[k,j], Q[:,k]))
	#end loop

	return Q[:,0:numCols], R[0:numCols,:]
#end def ######## ####### #######



######## ######## ####### #######
# PRIMARY FUNCTION

# initialize output file
with open('figs/p03errors.txt', 'w') as fout :
	fout.write('\n')
	fout.write('epsilon\tNormal\tAugmented\tHouseholder\tGivens')
	fout.write('\tGSClassic\tGSModified\tGSIterative\tSVD\n')
#end with

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
	print("\nUsing epsilon = {}".format(e))
	if e == 0 :
		print("SKIPPING: epsilon == 0")
		continue
	#end if

	# create the A & B matrices
	mxA, mxB = createSystemMatrices(e)
	m, n = mxA.shape

	# 0) the true x vector as a function of e
	xj = 1 / ( np.power(e, 2) + 3)
	xSol = np.array([xj, xj, xj], dtype=np.float64)
	# print(xSol)
	# print(np.array([xj, xj, xj], dtype=np.float32))


	# a) Normal equations
	# using cholesky factorization
	L = np.linalg.cholesky( np.dot(np.transpose(mxA), mxA) )
	z = np.linalg.solve( L, np.dot(np.transpose(mxA), mxB))
	xNormal = np.linalg.solve( np.transpose(L), z)
	# print(xNormal)
	errNormal = np.sum(np.abs( np.divide(np.subtract(xNormal, xSol), xSol) ))


	# b) Augmented system
	# create augmented A matrix
	bmxA = np.zeros( ((m+n), (m+n)), dtype=np.float32)
	bmxA[0:m,0:m] = np.identity(4)
	bmxA[0:m,m:(m+n)] = mxA
	bmxA[m:(m+n),0:m] = np.transpose(mxA)
	# create augmented b vector
	bmxB = np.zeros( (m+n), dtype=np.float32 )
	bmxB[0:m] = mxB
	# solve & extract resulting x vector
	rxAug = np.linalg.solve(bmxA, bmxB)
	xAugmented = rxAug[m:(m+n)]
	# print(xAugmented)
	errAugmented = np.sum(np.abs( np.divide(np.subtract(xAugmented, xSol), xSol) ))


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
	# print((hQ.shape, hR.shape))
	xHouseholder = np.linalg.solve(hR, np.dot(np.transpose(hQ), mxB))
	# print(xHouseholder)
	errHouseholder = np.sum(np.abs( np.divide(np.subtract(xHouseholder, xSol), xSol) ))


	# d) Givens QR
#TODO: this
	errGivens = 0

	# e) Classical G-S
	cgsQ, cgsR = origGramSchmidt(mxA)
	# print((cgsQ.shape, cgsR.shape))
	xGSClassic = np.linalg.solve(cgsR, np.dot(np.transpose(cgsQ), mxB))
	# print(xGSClassic)
	errGSClass = np.sum(np.abs( np.divide(np.subtract(xGSClassic, xSol), xSol) ))


	# f) Modified G-S
	mgsQ, mgsR = modGramSchmidt(mxA)
	# print((cgsQ.shape, cgsR.shape))
	xGSModified = np.linalg.solve(mgsR, np.dot(np.transpose(mgsQ), mxB))
	# print(xGSModified)
	errGSMod = np.sum(np.abs( np.divide(np.subtract(xGSModified, xSol), xSol) ))


	# g) iterative G-S

	# first iteration
	igsQ, igsR = origGramSchmidt(mxA)
	newA = igsR
	# print((igsQ.shape, mxB.shape))
	newB = np.dot(np.transpose(igsQ), mxB)
	# print((newA.shape, newB.shape))
	# newB = np.zeros((mxB.shape))
	# newB[0:mxA.shape[1]] = np.dot(np.transpose(igsQ), mxB)

	# successive iterations
	for iterRun in range(3) :
		igsQ, igsR = origGramSchmidt(newA)
		newA = igsR
		newB = np.dot(np.transpose(igsQ), newB)
		# newB = np.zeros((mxB.shape))
		# newB[0:newA.shape[1]] = np.dot(np.transpose(igsQ), newB)
		# print((igsQ.shape, newA.shape, mxB.shape))
	#end loop
	xGSIterative = np.linalg.solve(newA, newB)
	# print(xGSIterative)
	errGSIter = np.sum(np.abs( np.divide(np.subtract(xGSIterative, xSol), xSol) ))


	# h) Singular Value Decomposition
	# get SVD decomposition
	U, s, Vt = np.linalg.svd(mxA, full_matrices=True)
	# print("Resulting eigen values: {}".format(s))
	# create the pieces needed for pseudoinverse
	V = np.transpose(Vt)
	sigmaInv = np.zeros( (mxA.shape[1], mxA.shape[0]), dtype=np.float32 )
	for i in range(len(s)) :
		sigmaInv[i,i] = 1 / s[i]
	Ut = np.transpose(U)
	# create the pseudoinverse of A
	pseudoInvA = np.dot( np.dot( V, sigmaInv ), Ut )
	# get x from SVD
	xSVD = np.dot(pseudoInvA, mxB)
	# print(xSVD)
	errSVD = np.sum(np.abs( np.divide(np.subtract(xSVD, xSol), xSol) ))


	# # compare against built-in least squares function
	# xLstsq = np.linalg.lstsq(mxA, mxB)
	# print(xLstsq)
	# print(np.dot(mxA, xLstsq[0]))

	# output the error
	print("x as a function of epsilon: {}".format(xSol))
	print("Relative error ...")
	print("  Normal:      {:.2e} \tAugmented: {:.2e}".format(errNormal, errAugmented))
	print("  Householder: {:.2e} \tGivens:    {:.2e}".format(errHouseholder, errGivens))
	print("  G-S Classic: {:.2e} \tG-S Mod:   {:.2e}".format(errGSClass, errGSMod))
	print("  G-S Iter:    {:.2e} \tSVD:       {:.2e}".format(errGSIter, errSVD))

	with open('figs/p03errors.txt', 'a')  as fout :
		fout.write('{}\t{}\t{}\t{}\t'.format(
			e, errNormal, errAugmented, errHouseholder))
		fout.write('{}\t{}\t{}\t{}\t'.format(
			errGivens, errGSClass, errGSMod, errGSIter))
		fout.write('{}\n'.format(errSVD))
	#end with

	# print(errNormal)
	# print(np.linalg.norm(np.subtract(xSol, xNormal), ord=1))
	# print(errAugmented)
	# print(np.linalg.norm(np.subtract(xSol, xAugmented), ord=1))

#end loop

print("\nDone.\n")