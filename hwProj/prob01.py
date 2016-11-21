# QUESTION 3.5

import numpy as np
# import scipy as sp
from scipy import linalg as sl
from scipy.sparse import linalg as spl
import matplotlib.pyplot as pt
import matplotlib.lines as mlines



######## ######## ####### #######
# PARAMETERS

# positional data
xPos = np.array(
	[1.02, 0.95, 0.87, 0.77, 0.67, 0.56, 0.44, 0.30, 0.16, 0.01] )
yPos = np.array(
	[0.39, 0.32, 0.27, 0.22, 0.18, 0.15, 0.13, 0.12, 0.13, 0.15] )

######## ######## ####### #######



######## ######## ####### #######
# ANCILLARY FUNCTIONS

#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION



# Part A ######## ####### #######
print("\n>>>> Part A >>>>")

# create the A matrix & b vector
mxA = np.ones( (len(xPos), 5) )
mxA[:,0] = np.power(yPos, 2)
mxA[:,1] = np.multiply(xPos, yPos)
mxA[:,2] = xPos
mxA[:,3] = yPos

mxB = np.power(xPos, 2)

# solve for parameters a, b, c, d, e
pmX = np.linalg.lstsq(mxA, mxB)
pmA, pmB, pmC, pmD, pmE = pmX[0]

# print the parameter values
print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	pmA, pmB, pmC, pmD, pmE))

# draw the data points & the final ellipse equation
x = np.linspace(-2, 2, 200)
y = np.linspace(0, 2, 200)
x, y = np.meshgrid(x, y)
Y = pmA * np.power(y, 2) + (pmB * x * y) + (pmC * x) + (pmD * y) + pmE
X = np.power(x, 2)

pt.contour(x, y, (X - Y), [0])
# pt.legend(['projected orbit'])
pt.scatter(xPos, yPos)
pt.legend(['original data'])
pt.xlabel('x position')
pt.xlim([-.75, 1.25])
pt.ylabel('y position')
pt.ylim([-.25, 1.5])
pt.title('Q 3.5, part A -- planet\'s orbit')
# pt.legend(['original data', 'projected orbit'])
pt.savefig('figs/p01_ptA.png')
# pt.show()



# Part B ######## ####### #######
print("\n>>>> Part B >>>>")

xPosb = np.add(xPos, np.random.uniform(-0.005, 0.005))
yPosb = np.add(yPos, np.random.uniform(-0.005, 0.005))
# print(xPos)
# print(xPosb)

# create the A matrix & b vector
mxAb = np.ones( (len(xPosb), 5) )
mxAb[:,0] = np.power(yPosb, 2)
mxAb[:,1] = np.multiply(xPosb, yPosb)
mxAb[:,2] = xPosb
mxAb[:,3] = yPosb

mxB = np.power(xPosb, 2)

# solve for parameters a, b, c, d, e
pmX = np.linalg.lstsq(mxAb, mxB)
pmA, pmB, pmC, pmD, pmE = pmX[0]

# print the parameter values
print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	pmA, pmB, pmC, pmD, pmE))

# print("Matrix rank: {}".format( np.linalg.matrix_rank(mxA) ))

# pmX = sl.lstsq(mxA, mxB)
# pmA, pmB, pmC, pmD, pmE = pmX[0]
# # print the parameter values
# print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
# 	pmA, pmB, pmC, pmD, pmE))

# draw the data points & the final ellipse equation
Yb = pmA * np.power(y, 2) + (pmB * x * y) + (pmC * x) + (pmD * y) + pmE
Xb = np.power(x, 2)

figb = pt.figure()
ax = figb.add_subplot(111)
ax.contour(x, y, (X - Y), [0], colors='b')
ax.contour(x, y, (Xb - Yb), [0], colors='r')
# ax.contour(x, y, (X - Y), [0])
ax.scatter(xPos, yPos, c='b', marker='o')
# ax.clabel('projected orbit')
ax.scatter(xPosb, yPosb, c='r', marker='x')
ax.legend(['original data', 'perturbed data'])
ax.set_xlabel('x position')
ax.set_xlim([-.75, 1.25])
ax.set_ylabel('y position')
ax.set_ylim([-.25, 1.5])
ax.set_title('Q 3.5, part B -- perturbed orbit')
figb.savefig('figs/p01_ptB.png')
# pt.show()



# Part C ######## ####### #######
print("\n>>>> Part C >>>>")


print("condition number of original matrix A: {}".format(
	np.linalg.cond(mxA) ))
q, r = np.linalg.qr(mxA)
pmX = np.linalg.solve(r, np.dot( np.transpose(q), mxB) )
# print(pmX)
pmA, pmB, pmC, pmD, pmE = pmX
print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	pmA, pmB, pmC, pmD, pmE))

for k in range(1, 6) :
	tVal = np.power(10, -k)

#TODO: find better routine for rank-deficient matrix ?

	# pmX = so.lsq_linear(mxA, mxB, tol=tVal)
	# pmA, pmB, pmC, pmD, pmE = pmX[0]
	pmX = spl.lsmr(mxA, mxB, damp=tVal, atol=tVal, btol=tVal)
	pmA, pmB, pmC, pmD, pmE = pmX[0]
	condA = pmX[6]
	print("tolerance 10^-{}, cond(A) = {}".format(
		k, condA ))
	print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	pmA, pmB, pmC, pmD, pmE))
#end loop



# Part D ######## ####### #######
print("\n>>>> Part D >>>>")

U, s, Vt = np.linalg.svd(mxA, full_matrices=True)
print("Resulting eigen values: {}".format(s))
# print(U.shape)
# print(s.shape)
# print(Vt.shape)
V = np.transpose(Vt)
sigma = np.zeros( (10, 5) )
sigmaInv = np.zeros( (5, 10) )
for i in range(len(s)) :
	sigma[i,i] = s[i]
	sigmaInv[i,i] = 1 / s[i]
# print(sigma)
# print(sigmaInv)
Ut = np.transpose(U)



# Part E ######## ####### #######
print("\n>>>> Part E >>>>")

# for kRange in range(1,6) :

# 	pmX = np.zeros( (mxA.shape[1],5) )
# 	for k in range(kRange) :
# 		temp = np.dot(U[:,k], mxB)
# 		temp = np.divide(temp, s[k])
# 		temp = np.multiply(temp, V[:,k])
# 		# print(temp)
# 		pmX[:,k] = np.add(pmX[:,k], temp)
# 	#end loop
# 	# print(pmX)
# #end loop

# # for k in range(1,6) :
# # 	print(V[:,0:k])
# # 	print(sigmaInv[0:k])
# # 	temp = np.dot( V[:,0:k], sigmaInv[0:k])
# # 	print(temp)
# # 	pmX[k] = np.dot( temp, np.transpose(U[:,0:k]))
# # 	print(pmX)
# # #end loop

pmX = np.zeros( (mxA.shape[1],5) )
for k in range(5) :
	tSigInv = np.zeros( (5, 10) )
	for i in range(k+1) :
		# print(i)
		# print(sigmaInv[i,i])
		tSigInv[i,i] = sigmaInv[i,i]
	# print(tSigInv)
	temp = np.dot( V, tSigInv )
	# print(temp)
	pseudoInvA = np.dot( temp, Ut )
	# print(pmX)
	pmX[:,k] = np.dot(pseudoInvA, mxB)
	# print(pmX)
#end loop

fige = pt.figure()
ax = fige.add_subplot(111)
ax.contour(x, y, (X - Y), [0], colors='b')


x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
x, y = np.meshgrid(x, y)
useColors = ['c', 'y', 'g', 'm', 'r']
for i in range(5) :
	pmA, pmB, pmC, pmD, pmE = pmX[:,i]
	print("Parameters for k={}:\n   a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
		i+1, pmA, pmB, pmC, pmD, pmE))

	Ye = pmA * np.power(y, 2) + (pmB * x * y) + (pmC * x) + (pmD * y) + pmE
	Xe = np.power(x, 2)
	ax.contour(x, y, (Xe - Ye), [0], colors=useColors[i])
ax.scatter(xPos, yPos, c='b', marker='o')
legData = ([ mlines.Line2D([], [], color='c', label='k = 1'),
	mlines.Line2D([], [], color='y', label='k = 2'),
	mlines.Line2D([], [], color='g', label='k = 3'),
	mlines.Line2D([], [], color='m', label='k = 4'),
	mlines.Line2D([], [], color='r', label='k = 5'),
	mlines.Line2D([], [], color='b', marker='o', label='original data')
	])
ax.legend(handles=legData)
ax.set_xlabel('x position')
ax.set_xlim([-5, 5])
ax.set_ylabel('y position')
ax.set_ylim([-1, 7])
ax.set_title('Q 3.5, part E -- orbit from SVD')
fige.savefig('figs/p01_ptE.png')
# pt.show()