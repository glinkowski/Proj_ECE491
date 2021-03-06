import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D


######## ######## ####### #######
# PARAMETERS

# boundaries
xMin = 0
xMax = 1.0
tMin = 0
tMax = 0.06
# tMax = 0.18

# boundary conditions
bc_xMin = 0
bc_xMax = 0

# step sizes
dx = 0.05
dt_Orig = 0.0012
dt_Odd = 0.0013
dt_Big = 0.005
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# apply the boundary conditions to u(t,x)
def applyBoundCond(fu) :
	xLen, tLen = fu.shape
	fu[0,:] = np.repeat( bc_xMin, tLen )
	fu[xLen-1,:] = np.repeat( bc_xMax, tLen)
	return
#end def ####### ####### ########

# apply the initial conditions to u(t,x)
def applyInitCond(fu, fx) :
	for i in range(len(fx)) :
		if fx[i] <= 0.5 :
			fu[i,0] = 2 * fx[i]
		else :
			fu[i,0] = 2 - (2 * fx[i])
	return
#end def ####### ####### ########

# apply conditions & solve for u(t,x)
#	using the desired method
def solve_u(fSolver, fdt, fdx) :
	ft = np.arange(tMin, tMax+dt, dt)
	fx = np.arange(xMin, xMax+dx, dx)
	tLen = len(ft)
	xLen = len(fx)
	u_tx = np.zeros( (xLen, tLen) )

	applyBoundCond(u_tx)
	applyInitCond(u_tx, fx)

	fSolver(u_tx, fdt, fdx)
	return u_tx
#end def ####### ####### ########

# Plot the results for u(t,x) as 3d mesh
def plotMeshGrid(fFig, fu, fdt, fdx, fTitle) :
	ft = np.arange(tMin, tMax+dt, dt)
	fx = np.arange(xMin, xMax+dx, dx)
	mt, mx = np.meshgrid(ft, fx)
	fFig.plot_wireframe(mx, mt, fu, cstride=6)
	fFig.set_xlabel('x-axis')
	fFig.set_xlim([xMin, xMax])
	fFig.set_ylabel('time')
	fFig.set_ylim([tMin, tMax])
	fFig.set_zlabel('u(t,x)')
	fFig.set_title(fTitle)
	return
#end def ####### ####### ########

# apply Finite Differences to u(t,x)
def applyFiniDiff(fu, fdt, fdx) :
	xLen, tLen = fu.shape
	a = 1
	b = xLen-1

	# each u^k+1 col based on combo of u^k values
	const = fdt / (fdx * fdx)
	for ti in range(tLen-1) :
		k1Col = np.copy(fu[a:b,ti])

		coef1 = np.multiply(const, fu[(a+1):(b+1),ti])
		coef2 = np.multiply( (-2 * const), k1Col )
		coef3 = np.multiply(const, fu[(a-1):(b-1),ti])

		k1Col = np.add(k1Col, coef1)
		k1Col = np.add(k1Col, coef2)
		k1Col = np.add(k1Col, coef3)

		# place into u^k+1
		fu[a:b,(ti+1)] = k1Col

	return
#end def ####### ####### ########

# apply Backwards Euler to u(t,x)
def applyBackEuler(fu, fdt, fdx) :
	xLen, tLen = fu.shape
	a = 1
	b = xLen-1

	# Will solve the system u^k = A * u^(k+1)
	const = fdt / (fdx * fdx)
	A = np.multiply( (1 + (2*const)), np.eye(xLen) )
	A = np.add(A, np.multiply( -const, np.eye((xLen), k=1) ))
	A = np.add(A, np.multiply( -const, np.eye((xLen), k=-1) ))
	
	# Solve each column as a linear system
	for ti in range(tLen-1) :

		# b vector is just u^k
		# solve & place into k+1 column (first & last are untouched)
		kPlus1Sol = np.linalg.solve(A, fu[:,ti])
		fu[a:b,(ti+1)] = kPlus1Sol[a:b]

	return
#end def ####### ####### ########

# apply Backwards Euler to u(t,x)
def applyCrankNic(fu, fdt, fdx) :
	xLen, tLen = fu.shape
	a = 1
	b = xLen-1

	# Will solve the system u^k = A * u^(k+1)
	const = fdt / (2 * fdx * fdx)
	A = np.multiply( (1 + (2*const)), np.eye(xLen-2) )
	A = np.add(A, np.multiply( -const, np.eye((xLen-2), k=1) ))
	A = np.add(A, np.multiply( -const, np.eye((xLen-2), k=-1) ))

	# Solve each column as a linear system
	for ti in range(tLen-1) :

		# b vector is weighted combo of the u^k entries
		uk = np.multiply( (1 - (2 * const)), fu[a:b,ti] )
		uk = np.add(uk, np.multiply( const, fu[(a-1):(b-1),ti]))
		uk = np.add(uk, np.multiply( const, fu[(a+1):(b+1),ti]))

		# solve & place into k+1 column (first & last are untouched)
		kPlus1Sol = np.linalg.solve(A, uk)
		fu[a:b,(ti+1)] = kPlus1Sol

	return
#end def ####### ####### ########

def applySemiDiscreteSys(fu, fdt, fdx) :
	xLen, tLen = fu.shape
	a = 1
	b = xLen-1

	# each row will be solved with ode solver
	u0 = fu[a:b,0]
	# create the time steps
	ft = np.arange(tMin, tMax+fdt, fdt)
	# solve each line in time
	u_x = sci.odeint( f_SemiDisc, u0, ft, args=(fdx,) )
	u_x_T = np.transpose(u_x)

	fu[a:b,:] = u_x_T
	return
#end def ####### ####### ########

# The system of ODEs to solve u' = ...
def f_SemiDisc(fu_x, ft0, fdx) :
	tLen = len(fu_x)

	denom = (fdx * fdx)
	A = np.multiply( -2, np.eye(tLen) )
	A = np.add(A, np.eye((tLen), k=1))
	A = np.add(A, np.eye((tLen), k=-1))
	u_prime = np.dot(A, fu_x)
	u_prime = np.divide(u_prime, denom)

	return u_prime
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION
print("")

# Prepare figure for plotting
fig = pt.figure(figsize=(11, 18))


# Part 1: use Finite Diff w/ delta t = 0.0012 ------------
dt = dt_Orig
u_Part1 = solve_u(applyFiniDiff, dt, dx)

# Figure 1: Finite Diff plot
ax1 = fig.add_subplot(321, projection='3d')
figTit = '... using Finite Diff, dt = {}'.format(dt)
plotMeshGrid(ax1, u_Part1, dt, dx, figTit)


# Part 2: use Finite Diff w/ delta t = 0.0013 ------------
dt = dt_Odd
u_Part2 = solve_u(applyFiniDiff, dt, dx)

# Figure 2: FD w/ different tDelta plot
ax2 = fig.add_subplot(322, projection='3d')
figTit = '... using Finite Diff, dt = {}'.format(dt)
plotMeshGrid(ax2, u_Part2, dt, dx, figTit)


# Part 3: use Backward Euler -----------------------------
dt = dt_Big
u_Part3 = solve_u(applyBackEuler, dt, dx)

# Figure 3
ax3 = fig.add_subplot(323, projection='3d')
figTit = '... using Backward Euler, dt = {}'.format(dt)
plotMeshGrid(ax3, u_Part3, dt, dx, figTit)


# Part 4: use Crank-Nicolson -----------------------------
dt = dt_Big
u_Part4 = solve_u(applyCrankNic, dt, dx)

# Figure 4
ax4 = fig.add_subplot(324, projection='3d')
figTit = '... using Crank-Nicolson, dt = {}'.format(dt)
plotMeshGrid(ax4, u_Part4, dt, dx, figTit)


# Part 5: use Semi-Discrete w/ delta t = 0.005 -----------
dt = dt_Big
u_Part5 = solve_u(applySemiDiscreteSys, dt, dx)

# Figure 5
ax5 = fig.add_subplot(325, projection='3d')
figTit = '... using Semi Discrete, dt = {}'.format(dt)
plotMeshGrid(ax5, u_Part5, dt, dx, figTit)


fig.suptitle("The Heat Equation")
pt.show()


print("\nDone.\n")