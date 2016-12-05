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
tMax = 2.0

# boundary conditions
bc_xMin = 0
bc_xMax = 0

# step sizes
dx_reasonable = 0.05
dx_range = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]#, 0.008, 0.002, 0.001]
# dx_range = [0.002, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.5]
dt = 0.05

# prevData = (np.ones(2), 2)
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# Create u(t,x) & partial du/dt from init conds
def initializeU(fdt, fdx) :
	ft = np.arange(tMin, tMax+fdt, fdt)
	fx = np.arange(xMin, xMax+fdx, fdx)
	tLen = len(ft)
	xLen = len(fx)
	u_tx = np.zeros((tLen, xLen))

	# Apply the initial conditions
	u_tx[0,:] = np.sin( np.multiply(np.pi, fx))
	u_tx[1,:] = u_tx[0,:]

	# Apply the boundary conditions
	u_tx[:,0] = np.repeat( bc_xMin, tLen )
	u_tx[:,xLen-1] = np.repeat( bc_xMax, tLen)

	# # Define partial du/dt
	# u_tPrime = np.zeros( u_tx.shape )

	return u_tx#, u_tPrime
#end def ####### ####### ########

# The exact solution (for error comparison)
def getTrueSol(fdt, fdx) :
	ft = np.arange(tMin, tMax+fdt, fdt)
	fx = np.arange(xMin, xMax+fdx, fdx)
	term1 = np.cos( np.multiply(np.pi, ft) )
	term1 = np.reshape(term1, (len(term1), 1) )
	term2 = np.sin( np.multiply(np.pi, fx) )
	uRet = np.multiply(term1, term2)
	# Apply the boundary conditions
	uRet[:,0] = np.repeat( bc_xMin, len(ft) )
	uRet[:,len(fx)-1] = np.repeat( bc_xMax, len(ft))
	return uRet
#end def ####### ####### ########

# Apply the "method of lines"
def applySemiDiscreteSys(fu, fdt, fdx) :
	uRet = np.copy(fu)
	tLen, xLen = uRet.shape
	a = 1
	b = xLen-1

	# each column will be solved with ode solver
	u0 = fu[0,a:b]
	# u0 = fu[0,:]

	# global prevData
	# prevData = [fu[0:2,a:b], (0.0, fdt)]
	prevData = [u0, 0.0, u0, 0.0]
	# prevData = [np.zeros(b-a), 0.0]

	# u0 = fu[1,a:b]
	# u0 = fu[:,1]
	# create the time steps
	ft = np.arange(tMin, tMax+fdt, fdt)
	# print(ft)
	# solve each line in time
	u_t = sci.odeint( f_SemiDisc, u0, ft, args=(fdt, fdx, prevData) )
	# u_x_T = np.transpose(u_x)

	# print(u_x)

	uRet[:,a:b] = u_t
	# uRet[:,:] = u_t
	return uRet
#end def ####### ####### ########


# First solve u_xx, then get ut_k+1, average for ut_k
# The system of ODEs to solve u' = ...
def f_SemiDisc(u_k, t_k, fdt, fdx, prevData) :
	xLen = len(u_k)

	# get the ut_k-1 & dt
	ut_kMin = prevData[0]
	t_kMin = prevData[1]
	dt = t_k - t_kMin
	if t_k == 0 :
		ut_kMin = np.zeros(xLen)

	# Solve for u_xx
	denom = (fdx * fdx)
	A = np.multiply( -2, np.eye(xLen) )
	A = np.add(A, np.eye((xLen), k=1))
	A = np.add(A, np.eye((xLen), k=-1))
	u_xx = np.dot(A, u_k)
	u_xx = np.divide(u_xx, denom)
	#NOTE: u_xx == u_tt

	# Find ut_k+1 using u_xx & ut_k-1
	ut_kPlus = np.multiply(u_xx, (2 * dt))
	ut_kPlus = np.add(ut_kPlus, ut_kMin)

	# Averge the two to get ut_k
	ut_k = np.add(ut_kPlus, ut_kMin)
	ut_k = np.divide(ut_k, 2)

	# Save this round's data for the next one
	prevData[0] = ut_k
	prevData[1] = t_k

	return ut_k
#end def ####### ####### ########

# # First solve u_k+1, then solve u_t
# # The system of ODEs to solve u' = ...
# def f_SemiDisc(fu_t, fti, fdt, fdx, prevData) :
# 	xLen = len(fu_t)
# 	# print(fu_x)
# 	# print(ft0)

# 	# if fti == 0 :
# 	# 	return np.zeros(xLen)

# 	overwrite = True

# 	# global prevData
# 	u_kMin = prevData[2]
# 	t_kMin = prevData[3]
# 	# dt = (fti - t_kMin)
# 	if fti == t_kMin :
# 		u_kMin = prevData[0]
# 		t_kMin = prevData[1]
# 		overwrite = False
# 		# dt = (fti - t_kMin)
# 	# print(dt)
# 	dt = fti - t_kMin
# 	if dt == 0 :
# 		print(dt, t_kMin, fti)

# 	# Solve for u_xx
# 	denom = (fdx * fdx)
# 	A = np.multiply( -2, np.eye(xLen) )
# 	A = np.add(A, np.eye((xLen), k=1))
# 	A = np.add(A, np.eye((xLen), k=-1))
# 	u_xx = np.dot(A, fu_t)
# 	u_xx = np.divide(u_xx, denom)
# 	#NOTE: u_xx == u_tt

# 	# u_t_kPlus1 = np.multiply(u_xx, (2 * dt))
# 	# u_t_kPlus1 = np.add(u_t_kPlus1, u_t_kMin[0,:])

# 	# Solve u_tt = u_xx for u_k+1
# 	#	u_k+1 = u_xx * dt^2 - u_k-1 + 2*u_k
# 	# print(u_xx)
# 	u_kPlus1 = np.multiply(u_xx, (dt * dt))
# 	# print(u_kPlus1)
# 	# u_t_kPlus1 = np.add(u_t_kPlus1, u_t_kMin[0,:])
# 	u_kPlus1 = np.add(u_kPlus1, np.multiply(fu_t, 2))
# 	u_kPlus1 = np.add(u_kPlus1, np.multiply(u_kMin, -1))
# 	# print((u_kPlus1.shape, u_kMin.shape))
	
# 	# store previous 2 steps of data
# 	if overwrite :
# 		prevData[0] = prevData[2]
# 		prevData[1] = prevData[3]
# 	prevData[2] = fu_t
# 	prevData[3] = fti

# 	if fti == 0.0 :
# 		u_t = np.zeros(xLen)
# 	else :
# 		u_t = np.subtract(u_kPlus1, u_kMin)
# 		u_t = np.divide(u_t, (2 * dt))

# 	# return u_t
# 	return u_t
# #end def ####### ####### ########

# # NOTE: This one almost works
# # The system of ODEs to solve u' = ...
# def f_SemiDisc(fu_t, fti, fdt, fdx, prevData) :
# 	xLen = len(fu_t)
# 	# print(fu_x)
# 	# print(ft0)

# 	# if fti == 0 :
# 	# 	return np.zeros(xLen)

# 	# global prevData
# 	if fti == 0.0 :
# 		u_t_kMin1 = np.zeros(xLen)
# 		t_kMin1 = 0.0
# 	else :
# 		u_t_kMin1 = prevData[0]
# 		t_kMin1 = prevData[1]
# 	dt = fti - t_kMin1
# 	# print(dt, t_kMin1, fti)

# 	denom = (fdx * fdx)
# 	A = np.multiply( -2, np.eye(xLen) )
# 	A = np.add(A, np.eye((xLen), k=1))
# 	A = np.add(A, np.eye((xLen), k=-1))
# 	u_xx = np.dot(A, fu_t)
# 	u_xx = np.divide(u_xx, denom)
# 	#NOTE: u_xx == u_tt

# 	u_t_k = np.multiply(u_xx, dt)
# 	u_t_k = np.add(u_t_k, u_t_kMin1)
# 	# u_t_k = np.multiply(u_xx, dt)
# 	# u_t_k = np.add(u_t_k)

# 	# prevData = (u_t_k, fti)
# 	prevData[0] = u_t_k
# 	prevData[1] = fti

# 	# return u_t
# 	return u_t_k
# #end def ####### ####### ########

# # The system of ODEs to solve u' = ...
# def f_SemiDisc(fu_x, fti, fdt, fdx) :
# 	xLen = len(fu_x)
# 	# print(fu_x)
# 	# print(ft0)

# 	denom = (fdx * fdx)
# 	A = np.multiply( -2, np.eye(xLen) )
# 	A = np.add(A, np.eye((xLen), k=1))
# 	A = np.add(A, np.eye((xLen), k=-1))
# 	u_xx = np.dot(A, fu_x)
# 	u_xx = np.divide(u_xx, denom)
# 	#NOTE: u_xx == u_tt

# 	# ft = np.arange(tMin, tMax+fdt, fdt)
# 	# tLen = len(ft)

# 	# u_t = np.multiply( np.add(u_tMin1, u_xx), fdt)
# 	A2 = np.multiply(-1, np.eye(xLen, k=-1))
# 	A2 = np.add(A2, np.eye(xLen, k=1))
# 	A2 = np.divide( A2, (2 * fdt) )
# 	# print(A2)
# 	u_t = np.linalg.solve(A2, u_xx)
# 	# print(u_t)
# 	# u_t[0,:] = 0

# 	# # u_tPrime = np.zeros(len(fu_x))
# 	# # return u_tPrime

# 	# # expected solution
# 	# u_t = np.multiply( -np.pi, np.sin(np.multiply(np.pi, ft0)))
# 	# u_t = np.multiply( u_t, np.ones((len(fu_x))))
# 	# # u_t = np.multiply(u_t, fu_x)

# 	# return u_t
# 	return u_xx
# #end def ####### ####### ########

# Plot the results for u(t,x) as 3d mesh
def plotMeshGrid(fFig, fu, fdt, fdx, fTitle) :
	ft = np.arange(tMin, tMax+dt, dt)
	fx = np.arange(xMin, xMax+dx, dx)
	mx, mt = np.meshgrid(fx, ft)
	# print(fu.shape)
	# print(mt, mx)
	fFig.plot_wireframe(mt, mx, fu, cstride=1)
	fFig.set_xlabel('x-axis')
	fFig.set_xlim([tMin, tMax])
	fFig.set_ylabel('time')
	fFig.set_ylim([xMin, xMax])
	fFig.set_zlabel('u(t,x)')
	fFig.set_title(fTitle)
	return
#end def ####### ####### ########

def getMaxRelError(fbase, fcalc) :
	# Calculate the error
	err = np.subtract(fcalc, fbase)
	err = np.abs(err)
	# Extract the single maximum error b/t calc & base
	mErr = np.amax(err)

	# remove the boundaries (identically zero, also no error)
	a = 1
	b = fbase.shape[1] - 1
	# tFinal = fbase.shape[0]-1
	ubase = fbase[:,a:b]
	ucalc = fcalc[:,a:b]
	# Calculate the relative error
	relErr = np.divide(np.subtract(ucalc, ubase), ubase)
	relErr = np.abs(relErr)
	# Extract the single maximum relative error b/t calc & base
	mRelErr = np.amax(relErr)

	# # Print result to terminal
	# print("at dx = {}, max error = {:.5f}".format(dx, maxErr))

	return mErr, mRelErr
#end def ####### ####### ########




######## ######## ####### #######
# PRIMARY FUNCTION
print("")

# Prepare figure for plotting
fig = pt.figure(figsize=(10, 10))


# Draw the surface (calculated vs expected)
#	for a 'reasonable' dx
dx = dx_reasonable
u_tx = initializeU(dt, dx)
u_sol = applySemiDiscreteSys(u_tx, dt, dx)
u_true = getTrueSol(dt, dx)

# # Print result to terminal
# maxErr, maxRelErr = getMaxRelError(u_true, u_sol)
# print("at dx = {}, max error = {:.5f}".format(dx, maxErr))
# print("     max relative error = {:.5f}".format(maxRelErr))

# Plot the two surfaces
ax1 = fig.add_subplot(221, projection='3d')
figTit = '... calculated w/ dx = {}'.format(dx)
plotMeshGrid(ax1, u_sol, dt, dx, figTit)

ax2 = fig.add_subplot(222, projection='3d')
figTit = '... true sol'
plotMeshGrid(ax2, u_true, dt, dx, figTit)


# Compare error while varying dx
y1Plot = list()
y2Plot = list()
print("")
for dx in dx_range :
	# print("calculating error at dx = {}".format(dx))

	u_tx = initializeU(dt, dx)
	u_sol = applySemiDiscreteSys(u_tx, dt, dx)
	u_true = getTrueSol(dt, dx)
	maxErr, maxRelErr = getMaxRelError(u_true, u_sol)

	# Print result to terminal
	print("at dx = {},  max error = {:.5f}".format(dx, maxErr))
	print("  max relative error = {:.1e}\n".format(maxRelErr))

	y1Plot.append(maxErr)
	y2Plot.append(maxRelErr)

#end loop

# Plot the resulting errors as function of dx
ax3 = fig.add_subplot(223)
ax3.plot(dx_range, y1Plot)
ax3.set_xlabel('delta x')
ax3.set_xscale('log')
ax3.invert_xaxis()
ax3.set_ylabel('maximum error')
ax3.set_yscale('log')
ax3.set_title('max Raw error (true - calculated)')

ax4 = fig.add_subplot(224)
ax4.plot(dx_range, y2Plot)
ax4.set_xlabel('delta x')
ax4.set_xscale('log')
ax4.invert_xaxis()
ax4.set_ylabel('maximum relative error')
ax4.set_yscale('log')
ax4.set_title('max Relative error (true - calc)/true' +
	'\n very high where true ~ 0 ')
# ax4.set_subtitle('very high where true ~ 0')



fig.suptitle('Wave Equation, dt={}'.format(dt))
pt.show()