import numpy as np
import scipy.linalg as la

# Define the matrices
A = np.array( [[2, 4, -2], [4, 9, -3], [-2, -1, 7]] )
b = np.array( [2, 8, 10] )
c = np.array( [4, 8, -6] )

# print(A)
# print(b)

# Get LU factorization
P, L, U = la.lu(A, permute_l=False, overwrite_a=False)
Pi = np.transpose(P)
# print(P)
# print(Pi)

# print(P)
# print(L)
# print(U)

# A2 = np.dot(L,U)
# print(A)
# print(A2)
# print( np.dot(P, A2) )

# Solve Ly=b and Ux=y (Note: permute b)
b2 = np.dot(Pi, b)
#print(b2)
y = la.solve_triangular(L, b2, lower=True)
sol1 = la.solve_triangular(U, y)
#print(sol1)


# Solve Ly=c and Ux=y (Note: permute c)
c2 = np.dot(Pi, c)
#print(b2)
y = la.solve_triangular(L, c2, lower=True)
sol2 = la.solve_triangular(U, y)


# Perterb A such that a_1,2 = 2

# choose u and v^T
u = np.array( [2, 0, 0] )
#u.reshape(1,3)
#print(u)
v = np.array( [0, 1, 0] )
#v.reshape(1,3)
#print(v)


# vT = v.reshape(3,1)
# #vT = np.transpose(v)
# print(vT)
# uvT = np.transpose( np.multiply(u, vT) )
# print(uvT)
# # print(np.dot(Pi, uvT))

# u = np.dot(Pi, u)
# v = np.dot(Pi, v)


# solve Az = u
temp = la.solve_triangular(L, u, lower=True)
z = la.solve_triangular(U, temp)
#z.reshape(1,3)
print(z)
# solve Ay = b
temp = la.solve_triangular(L, b2, lower=True)
y = la.solve_triangular(U, temp)
print(y)

# find x
denom = np.subtract(1, np.dot(v, z))
numer = np.dot(v, y)
step2 = np.divide(numer, denom)
step3 = np.dot(step2, z)
sol3 = np.add(y, step3)

print(sol3)