import scipy.linalg as la
import numpy as np

A = np.array( [[6, 2, 1], [2, 3, 1], [1, 1, 1]] )
x0 = np.array( [0, 0, 1] )
numIters = 15


# shift A by 2
A2 = np.subtract(A, np.multiply(2,np.identity(A.shape[0])))
# NOTE: The line above results in enough error that the autograder
#	doesn't recognize it as correct. But the line below results
#	in nearly the exact same values as the numpy function.
# NOTE: The above note holds true when using la.solve_triangular(),
#	but not when using lu_factor & lu_solve ...
#A2 = np.subtract(A, np.multiply(2.4,np.identity(A.shape[0])))


# get LU factorization (Pi to fix b in Ax=b => PLUx = b)
# P, L, U = la.lu(A2)
# Pi = np.transpose(P)
LU, pivot = la.lu_factor(A2)


# 15 iterations, apply inverse power method
# xi = np.dot(Pi, x0)
xi = x0
for i in range(numIters) :
	# z = la.solve_triangular(L, xi, lower=True)
	# y = la.solve_triangular(U, z)
	y = la.lu_solve( (LU, pivot), xi )
	xi = np.divide( y, np.linalg.norm(y, ord=np.inf) )
#end loop


# Get eigval from xi, use Rayleigh Quotient
eigvec = np.divide(xi, np.linalg.norm(xi))
eigval = np.dot( np.dot(eigvec, A), eigvec)
eigval = eigval / np.dot(eigvec, eigvec)
print('eigen value: {}'.format(eigval))


# Get the numpy-computed eigval & eigvec
eigTrueSol = np.linalg.eig(A)
diff = np.inf
keepIdx = -1
for vi in range(len(eigTrueSol[0])) :
	newDiff = abs(eigTrueSol[0][vi] - eigval)
	if newDiff < diff :
		diff = newDiff
		keepIdx = vi
#end if
etVal = eigTrueSol[0][keepIdx]
etVec = eigTrueSol[1][:,keepIdx]
etVec = np.divide(etVec, np.linalg.norm(etVec))


diffval = etVal - eigval
print('difference: {}'.format(diffval))

diffvec = np.subtract(np.abs(etVec), np.abs(eigvec))
print('vector diff: {}'.format(diffvec))


# print(eigval)
# print(eigTrueSol[0])
# print(eigvec)
# print(etVec)