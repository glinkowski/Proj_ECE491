import scipy.linalg as la
import numpy as np

A = np.array( [[6, 2, 1], [2, 3, 1], [1, 1, 1]] )
x0 = np.array( [0, 0, 1] )
numIters = 15
eigTrueSol = np.linalg.eig(A)


# shift A by 2
A2 = np.subtract(A, np.multiply(2,np.identity(A.shape[0])))
print(A2)

P, L, U = la.lu(A2)
Pi = np.transpose(P)


# print(P)
# print(L)
# print(U)

# print( np.dot(P,L) )
# print( np.dot(np.dot(P,L),U) )



# xOld = np.dot(Pi,x0)
# z = la.solve_triangular(L, xOld, lower=True)
# y = la.solve_triangular(U, z)
# xNew = np.divide( y, np.linalg.norm(y, ord=np.inf) )
# print(xNew)


# 15 iterations, apply inverse power method
xi = np.dot(Pi, x0)
for i in range(numIters) :

	z = la.solve_triangular(L, xi, lower=True)
	y = la.solve_triangular(U, z)
	xi = np.divide( y, np.linalg.norm(y, ord=np.inf) )
#end loop

print(xi)
print(eigTrueSol)

# Get eigval from xi, use Rayleigh Quotient
val = np.dot( np.dot(xi, A), xi)
val = val / np.dot(xi, xi)
print(val)

# print( np.dot(xi,A) )
# print( np.dot( np.dot(xi,A), xi) )
# print( np.dot(xi, xi) )