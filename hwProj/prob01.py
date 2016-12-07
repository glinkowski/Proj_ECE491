# QUESTION 3.5

import numpy as np
from scipy import linalg as sl
from scipy import optimize as so
from scipy.sparse import linalg as spl
import matplotlib.pyplot as pt
import matplotlib.lines as mlines


######## ######## ####### #######
# PARAMETERS

# positional data (input)
xPos = np.array(
	[1.02, 0.95, 0.87, 0.77, 0.67, 0.56, 0.44, 0.30, 0.16, 0.01] )
yPos = np.array(
	[0.39, 0.32, 0.27, 0.22, 0.18, 0.15, 0.13, 0.12, 0.13, 0.15] )

# amount by which to perturb data
ptbRange = 0.005	# suggested = 0.005
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# use input data to create system matrices
def createABMatrices(xCoords, yCoords) :
	A = np.ones( (len(xCoords), 5) )
	A[:,0] = np.power(yCoords, 2)
	A[:,1] = np.multiply(xCoords, yCoords)
	A[:,2] = xCoords
	A[:,3] = yCoords

	B = np.power(xCoords, 2)

	return A, B
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

# Part A ######## ####### #######
print("\n>>>> Part A >>>>")

# create the A matrix & b vector
mxA, mxB = createABMatrices(xPos, yPos)

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

# plot the results & save figure
pt.contour(x, y, (X - Y), [0])
pt.scatter(xPos, yPos)
pt.legend(['original data'])
pt.xlabel('x position')
pt.xlim([-.75, 1.25])
pt.ylabel('y position')
pt.ylim([-.25, 1.5])
pt.title('Q 3.5, part A -- planet\'s orbit')
pt.savefig('figs/p01_ptA.png')


# Part B ######## ####### #######
print("\n>>>> Part B >>>>")

# create A matrix & b vector from perturbed data
xPosb = np.add(xPos, np.random.uniform(-ptbRange, ptbRange, len(xPos)))
yPosb = np.add(yPos, np.random.uniform(-ptbRange, ptbRange, len(yPos)))
mxAb, mxBb = createABMatrices(xPosb, yPosb)

# solve for parameters a, b, c, d, e
pmX = np.linalg.lstsq(mxAb, mxBb)
pmA, pmB, pmC, pmD, pmE = pmX[0]

# print the parameter values
print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	pmA, pmB, pmC, pmD, pmE))

# draw the data points & the final ellipse equation
Yb = pmA * np.power(y, 2) + (pmB * x * y) + (pmC * x) + (pmD * y) + pmE
Xb = np.power(x, 2)

# plot the results & save figure
figb = pt.figure()
ax = figb.add_subplot(111)
ax.contour(x, y, (X - Y), [0], colors='b', width=2 )
ax.contour(x, y, (Xb - Yb), [0], colors='r')
ax.scatter(xPos, yPos, c='b', marker='o')
ax.scatter(xPosb, yPosb, c='r', marker='x')
ax.legend(['original data', 'perturbed data'])
ax.set_xlabel('x position')
ax.set_xlim([-.75, 1.25])
ax.set_ylabel('y position')
ax.set_ylim([-.25, 1.5])
ax.set_title('Q 3.5, part B -- perturbed orbit')
figb.savefig('figs/p01_ptB.png')


# Part C ######## ####### #######
print("\n>>>> Part C >>>>")

# check the condition number of original matrix
print("condition number of original matrix A: {}".format(
	np.linalg.cond(mxA) ))
q, r = np.linalg.qr(mxA)
pmX = np.linalg.solve(r, np.dot( np.transpose(q), mxB) )
pmA, pmB, pmC, pmD, pmE = pmX
print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	pmA, pmB, pmC, pmD, pmE))

# compare solving with different tolerance values
k_range = [-12, -6, -3, -1, 1, 3]
# for kVal in range(-1, 4) :
	# tVal = np.power(10, -kVal)
for kVal in k_range :
	tVal = np.power(10, kVal)

#TODO: find a worse routine for rank-deficient matrix ?

	# pmX = spl.lsmr(mxA, mxB, damp=tVal, atol=tVal, btol=tVal)
	# print(pmX)
	# pmA, pmB, pmC, pmD, pmE = pmX[0]
	# condA = pmX[6]
	# rankA = np.linalg.matrix_rank(A)
	# print("tolerance 10^-{}, cond(A) = {}".format(k, condA ))
	# # print("tolerance 10^-{}, rank(A) = {}".format(k, condA ))
	# print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	# pmA, pmB, pmC, pmD, pmE))

	# fitY = np.dot(mxA, pmX)
	# relErr = np.abs(np.divide(np.subtract(fitY, yPos), yPos))
	# print("  Relative error of Y observed & calc: {}".format(relErr))


#NOTE: Trying the svd approach, can set a tolerance
#  for singular values
#HOWEVER, no matter what I set my tolerance,
#  it finds the same eigenvalues
	U, s, Vt = spl.svds(mxA, k=(min(mxA.shape)-1), tol=tVal)
	V = np.transpose(Vt)
	sigmaInv = np.zeros( (len(s), len(s)) )
	for i in range(len(s)) :
		sigmaInv[i,i] = 1 / s[i]
	Ut = np.transpose(U)
	# create the pseudoinverse of A
	temp = np.dot( V, sigmaInv )
	pseudoInvA = np.dot( temp, Ut )
	# get parameters from SVD
	pmX = np.dot(pseudoInvA, mxB)
	pmA, pmB, pmC, pmD, pmE = pmX

	rankA = np.linalg.matrix_rank(pseudoInvA)
	print("tolerance 10^{}, rank(A^+) = {}".format(kVal, rankA ))
	# print("Parameters: a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
	# pmA, pmB, pmC, pmD, pmE))

	fitY = np.dot(mxA, pmX)
	relErr = np.abs(np.divide(np.subtract(fitY, yPos), yPos))
	print("  Mean Rel. error of Y observed vs calc: {:.3f}".format(
		np.mean(relErr)))
#end loop


# Part D ######## ####### #######
print("\n>>>> Part D >>>>")

# get SVD decomposition
U, s, Vt = np.linalg.svd(mxA, full_matrices=True)
print("Resulting eigen values: {}".format(s))

# convert into the pieces needed for part E
V = np.transpose(Vt)
# sigma = np.zeros( (10, 5) )
sigmaInv = np.zeros( (5, 10) )
for i in range(len(s)) :
	# sigma[i,i] = s[i]
	sigmaInv[i,i] = 1 / s[i]
Ut = np.transpose(U)


# Part E ######## ####### #######
print("\n>>>> Part E >>>>")

# find parameters using only first k eigen values
pmX = np.zeros( (mxA.shape[1],5) )

for k in range(5) :
	# keep only first k eigen values (zero out rest)
	tSigInv = np.zeros( (5, 10) )
	for i in range(k+1) :
		tSigInv[i,i] = sigmaInv[i,i]
	# create the pseudoinverse of A
	temp = np.dot( V, tSigInv )
	pseudoInvA = np.dot( temp, Ut )
	# get parameters from SVD
	pmX[:,k] = np.dot(pseudoInvA, mxB)
#end loop

# plot the results & save figure
fige = pt.figure()
ax = fige.add_subplot(111)
ax.contour(x, y, (X - Y), [0], colors='b')

xe = np.linspace(-10, 10, 200)
ye = np.linspace(-10, 10, 200)
xe, ye = np.meshgrid(xe, ye)
useColors = ['c', 'y', 'g', 'm', 'r']
print("Parameters for ... ")
# add a contour for each set of k eigenvalues
for i in range(5) :
	pmA, pmB, pmC, pmD, pmE = pmX[:,i]
	print("k={}:    a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
		i+1, pmA, pmB, pmC, pmD, pmE))

	Ye = pmA * np.power(ye, 2) + (pmB * xe * ye) + (pmC * xe) + (pmD * ye) + pmE
	Xe = np.power(xe, 2)
	ax.contour(xe, ye, (Xe - Ye), [0], colors=useColors[i])
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


# Part F ######## ####### #######
print("\n>>>> Part F >>>>")

# create the A matrix & b vector from perturbed data
xPosf = np.add(xPos, np.random.uniform(-ptbRange, ptbRange, len(xPos)))
yPosf = np.add(yPos, np.random.uniform(-ptbRange, ptbRange, len(yPos)))
mxAf, mxBf = createABMatrices(xPosf, yPosf)

# get SVD
U, s, Vt = np.linalg.svd(mxAf, full_matrices=True)
print("Resulting eigen values: {}".format(s))
# convert the parts needed for next step
V = np.transpose(Vt)
sigmaInv = np.zeros( (5, 10) )
for i in range(len(s)) :
	sigmaInv[i,i] = 1 / s[i]
Ut = np.transpose(U)

# get params using only so many eigen values at a time
pmX = np.zeros( (mxAf.shape[1],5) )
for k in range(5) :
	# select only k first eigen values
	tSigInv = np.zeros( (5, 10) )
	for i in range(k+1) :
		tSigInv[i,i] = sigmaInv[i,i]
	# get pseudoinverse of A
	temp = np.dot( V, tSigInv )
	pseudoInvA = np.dot( temp, Ut )
	# get parameters
	pmX[:,k] = np.dot(pseudoInvA, mxBf)
#end loop

# plot the resulting ellipses & save figure
figf = pt.figure()
ax = figf.add_subplot(111)
ax.contour(x, y, (X - Y), [0], colors='b')

useColors = ['c', 'y', 'g', 'm', 'r']
print("Parameters for ... ")
# add a contour for each set of k eigenvalues
for i in range(5) :
	pmA, pmB, pmC, pmD, pmE = pmX[:,i]
	print("k={}:    a={:.5}, b={:.5}, c={:.5}, d={:.5}, e={:.5}".format(
		i+1, pmA, pmB, pmC, pmD, pmE))

	Yf = pmA * np.power(ye, 2) + (pmB * xe * ye) + (pmC * xe) + (pmD * ye) + pmE
	Xf = np.power(xe, 2)
	ax.contour(xe, ye, (Xf - Yf), [0], colors=useColors[i])
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
ax.set_title('Q 3.5, part F -- perturbed orbit from SVD')
figf.savefig('figs/p01_ptF.png')



# Part G ######## ####### #######
print("\n>>>> Part G >>>>")

def applyTotalLeastSquares(fA, fB) :
	n = fA.shape[1]

	# Apply SVD to the augmented matrix [A, B]
	newB = np.reshape(fB, (len(fB), 1))
	augMx = np.hstack( (fA, newB) )
	U, s, Vt = np.linalg.svd(augMx, full_matrices=True)
	V = np.transpose(Vt)

	# Parameters = -V12 / V22 (blocks of V)
	V12 = V[0:n,n:V.shape[1]]
	V22 = V[n:V.shape[0],n:V.shape[1]]
	params = np.divide( np.multiply( -1, V12), V22)
	params = params.reshape( (len(params),))

	return params
#end def ####### ####### ########

# use TLS on original data, draw contour
pmXg = applyTotalLeastSquares(mxA, mxB)
pmAg, pmBg, pmCg, pmDg, pmEg = pmXg
Yg = pmAg * np.power(y, 2) + (pmBg * x * y) + (pmCg * x) + (pmDg * y) + pmEg
Xg = np.power(x, 2)
# print the parameter values
print("Applying TLS to original data ...")
print("Parameters: a={:.5f}, b={:.5f}, c={:.5f}, d={:.5f}, e={:.5f}".format(
	pmAg, pmBg, pmCg, pmDg, pmEg))

# use TLS on perturbed data, draw contour
pmXgp = applyTotalLeastSquares(mxAb, mxBb)
pmAgp, pmBgp, pmCgp, pmDgp, pmEgp = pmXgp
Ygp = pmAgp * np.power(y, 2) + (pmBgp * x * y) + (pmCgp * x) + (pmDgp * y) + pmEgp
Xgp = np.power(x, 2)
# print the parameter values
print("\nApplying TLS to perturbed data from part B ...")
print("Parameters: a={:.5f}, b={:.5f}, c={:.5f}, d={:.5f}, e={:.5f}".format(
	pmAgp, pmBgp, pmCgp, pmDgp, pmEgp))

# plot the resulting ellipses & save figure
figg = pt.figure()
pt.contour(x, y, (X - Y), [0], colors='b')
pt.contour(x, y, (Xg - Yg), [0], colors='y', linewidths=5)
pt.contour(x, y, (Xb - Yb), [0], colors='r')
pt.contour(x, y, (Xgp - Ygp), [0], colors='k', linewidths=1.5)
pt.scatter(xPos, yPos, c='b', marker='o')
pt.scatter(xPosb, yPosb, c='r', marker='x')
legData = ([ 
	mlines.Line2D([], [], color='b', marker='o', label='original data'),
	mlines.Line2D([], [], color='y', label='Total Lst Sqr'),
	mlines.Line2D([], [], color='r', marker='x', label='perturbed data'),
	mlines.Line2D([], [], color='k', label='Perturbed TLS')
	])
pt.legend(handles=legData)
pt.xlabel('x position')
pt.xlim([-.75, 1.25])
pt.ylabel('y position')
pt.ylim([-.25, 1.5])
pt.title('Q 3.5, part G -- planet\'s orbit')
pt.savefig('figs/p01_ptG.png')


pt.show()


print("\n")