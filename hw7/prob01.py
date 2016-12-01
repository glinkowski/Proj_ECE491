import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
import matplotlib.pyplot as pt


######## ######## ####### #######
# PARAMETERS

# boundaries (a < t < b)
a = 0
b = 1

# boundary conditions
uofa = 0
uofb = 1

# desired n values
nValsFD = [1, 3, 7, 15]
nValsCo = [3, 4, 5, 6]
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

def createTHforFiniteDiff(fn) :
	h = 1 / (fn + 1.0)
	t = np.arange(a, b+h, h)

	return t, h
#end def ####### ####### ########

def f_FinDif(fu, ft, h) :
	# force the boundary conditions
	fu[0] = uofa
	fu[len(fu)-1] = uofb

	# solve: u'' - 10u^3 - 3u - t^2 (= 0)
	# 	create matrix for u''
	M = np.multiply(-2, np.eye( len(fu) ))
	M = np.add(M, np.eye(len(fu), k=1))
	M = np.add(M, np.eye(len(fu), k=-1))
	M = np.divide( M, (h**2) )
	# add component for -3u
	M = np.add(M, np.multiply(-3, np.eye( len(fu) )))
	# multiply by y vector & subtract 10 u^3
	f_ut = np.dot(M, fu)
	f_ut = np.subtract(f_ut, np.multiply(10, np.power(fu, 3)))
	# subtract t^2
	f_ut = np.subtract(f_ut, np.power(ft, 2))

	# Force the endpoints = 0
	#	b/c looking for where solution is a zero vector
	#	and can't determine endpoints w/o +1/-1 data points
	f_ut[0] = 0
	f_ut[len(f_ut)-1] = 0

	return f_ut
#end def ####### ####### ########

def createTHforFCollocation(fn) :
	if fn <= 1 :
		fn = 2
	h = 1 / (fn - 1.0)
	t = np.arange(a, b+h, h)

	return t, h
#end def ####### ####### ########

def getVofXT(fx, ft) :

	# Get u(t)
	A = np.zeros( (len(ft), len(fx)) )
	for col in range(len(fx)) :
		A[:,col] = np.power(ft, col)

	v_xt = np.dot(A, fx)

	return v_xt
#end def ####### ####### ########

def f_Colloc(fx, ft) :
	# first element of x is known
	fx[0] = uofa

	v_xt = getVofXT(fx, ft)

	# Force boundary conditions on u(t)
	v_xt[0] = uofa
	v_xt[len(v_xt)-1] = uofb

	# Back-solve for new x vector
	A = np.zeros( (len(ft), len(fx)) )
	for col in range(len(fx)) :
		A[:,col] = np.power(ft, col)
	result = np.linalg.lstsq(A, v_xt)
	fx = result[0]

	v_xt = getVofXT(fx, ft)

	# Get u''(t)
	B = np.zeros( (len(ft), len(ft)) )
	for col in range(2, len(ft)) :
		B[:,col] = np.multiply( np.math.factorial(col), np.power(t, col-2))
	fu_dp = np.dot(B, fx)


	# Calculate solution to equation
	f_ut = np.subtract(fu_dp, np.multiply(3.0, v_xt))
	f_ut = np.subtract(f_ut, np.multiply(10.0, np.power(v_xt, 3.0)))
	f_ut = np.subtract(f_ut, np.power(ft, 2.0))

	# Force the endpoints = 0
	#	b/c looking for where solution is a zero vector
	#	and can't determine endpoints w/o +1/-1 data points
	f_ut[0] = 0
	f_ut[len(f_ut)-1] = 0

	return f_ut
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION
print("")


# Apply the Finite Difference method
print("Applying the Finite Difference method ...")
plotFD = list()
for n in nValsFD :
	t, h = createTHforFiniteDiff(n)
	y0 = np.copy(t)

	# use solver with finite diff function
	uSol = fsolve(f_FinDif, y0, args=(t, h))
	fofyt = f_FinDif(uSol, t, h)
	print("n = {:2d}, accuracy at t_i = {:.2e}".format( n,
		np.linalg.norm(np.subtract(np.zeros(len(fofyt)), fofyt))))

	# store results to plot
	plotFD.append( (t, uSol) )
#end loop
print("")


# Apply the Collocation method
print("Applying the Collocation method ...")
plotCo = list()
for n in nValsCo :
	t, h = createTHforFCollocation(n)
	x0 = np.copy(t)

	# use solver with collocation function
	xSol = fsolve(f_Colloc, x0, args=(t))
	fofut = f_Colloc(xSol, t)
	print("n = {:2d}, accuracy at t_i = {:.2e}".format( n,
		np.linalg.norm(np.subtract(np.zeros(len(fofut)), fofut))))

	# store results to plot
	delta = 0.01
	tPlot = np.arange(a, b+delta, delta)
	uSol = getVofXT(xSol, tPlot)
	plotCo.append( (tPlot, uSol, n) )
#end loop


# Plot the desired figures
fig = pt.figure(figsize=(8, 12))

# Figure 1: Finite Diff plots
ax1 = fig.add_subplot(211)
legendHandles = list()
for item in plotFD[::-1] :
	ax1.plot( item[0], item[1], '-s' )
	legendHandles.append('n = {:d}'.format( len(item[0])-2 ))
ax1.set_title("using Finite Difference")
ax1.set_xlabel("time (t)")
ax1.set_ylabel("u(t)")
ax1.legend( legendHandles, loc=0 )

# Figure 2: Collocation plots
ax2 = fig.add_subplot(212)
legendHandles = list()
for item in plotCo :
	ax2.plot( item[0], item[1] )
	legendHandles.append('n = {:d}'.format( item[2] ))
ax2.set_title("using Collocation")
ax2.set_xlabel("time (t)")
ax2.set_xlim([a,b])
ax2.set_ylabel("u(t)")
legendHandles.append('u(t) = 1')
ax2.plot( item[0], np.ones(len(item[0])), ':k' )
ax2.legend( legendHandles, loc=0 )

pt.show()

print("\n")