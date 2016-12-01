# QUESTION 6.16

import math
import numpy as np
import numpy.linalg as npl


######## ######## ####### #######
# PARAMETERS

# input data
v0Vals = [0.024, 0.036, 0.053, 0.060, 0.064]
sVals = [2.5, 5.0, 20.0, 15.0, 20.0]
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# Newton Method: one iteration
def iterNewton(bfr, bJf, xk) :
	q, r = npl.qr( bJf(xk) )
	sk = npl.solve(r, np.dot( np.transpose(q), bfr(xk)) )

	xk1 = np.add(xk, sk)

	return xk1
#end def ####### ####### ########

def solveNewton(bfr, bJf, x0) :

	# stopping criteria
	errTol = 1e-15
	numIters = 32

	# Newton Method: first iteration
	i = 0
	xNew = iterNewton(bfr, bJf, x0)
	xDiff = 2

	# Newton Method: successive iterations
	while(xDiff > errTol) :
		i += 1

		# Check how much the new x vector changes
		xOld = xNew
		xNew = iterNewton(bfr, bJf, xNew)
		xDiff = npl.norm( np.subtract(xNew, xOld), ord=2)

		if i >= numIters :
			break
	#end loop

	print("Newton iterations: {}, delta: {:.2e}".format(i, xDiff))
	return xNew
#end def ####### ####### ########

# The nonlinear function v0(S)
def f_v0ofS(fx) :
	V, Km = fx

	fv = np.zeros( len(sVals) )
	fv = np.add(1.0, np.divide(Km, sVals))
	fv = np.divide(V, fv)

	return fv
#end def ####### ####### ########

# Residual defined as (y_true - f(x))
def f_residual_v0ofS(fx) :
	fr = np.subtract(v0Vals, f_v0ofS(fx))

	return fr
#end def ####### ####### ########

# The Jacobian matrix of first f(x)
def Jf_v0ofX(fx) :
	V, Km = fx

	J11 = np.divide( sVals, np.add(sVals, Km) )

	J12 = np.multiply( (-V), sVals )
	J12 = np.divide( J12, np.power( np.add(sVals, Km), 2.0) )

	Jf = np.zeros( (len(sVals), len(fx)) )
	Jf[:,0] = J11
	Jf[:,1] = J12
	return Jf
#end def ####### ####### ########

# Calculate relative error
def getRelErr(base, value) :
	err = np.abs( np.divide( np.subtract(base, value), base))
	return err
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("\n\n>>>> Part A >>>>")

x0 = [2, 2]

# Apply Gauss-Newton iterative solver
print("\nNewton results -----------------------")
xFinal = solveNewton(f_residual_v0ofS, Jf_v0ofX, x0)
v0_approx = f_v0ofS(xFinal)

# Display and compare results
print("  using x0 = {}".format(x0))
print("  found x  = [{:.5f}, {:.5f}]".format(xFinal[0], xFinal[1]))
print("  final v0(S) = {}".format(v0_approx))
print("  diff b/t this and expected: {:.3e}".format(
	npl.norm( np.subtract(v0_approx, v0Vals), ord=2) ))
print("final:\n  V = {}\n Km = {}".format(xFinal[0], xFinal[1]))



print("\n\n>>>> Part B >>>>")

# The Lineweaver & Burk approximation ------------
bLB = np.divide(1.0, v0Vals)
ALB = np.ones( (len(v0Vals), 2) )
ALB[:,1] = np.divide(1.0, sVals)

# solve linear system
xlstsq = npl.lstsq(ALB, bLB)
xLB = xlstsq[0]
VLB = 1.0 / xLB[0]
KmLB = xLB[1] / xLB[0]

# display results
print("\nLineweaver & Burk rearrangement --------")
print("  V = {:.5f},  error = {:.2e}".format(VLB, getRelErr(xFinal[0], VLB)))
print(" Km = {:.5f},  error = {:.2e}".format(KmLB, getRelErr(xFinal[0], KmLB)))


# The Dixon approximation ------------------------
bDx = np.divide(sVals, v0Vals)
ADx = np.ones( (len(v0Vals), 2) )
ADx[:,1] = sVals

# solve linear system
xlstsq = npl.lstsq(ADx, bDx)
xDx = xlstsq[0]
VDx = 1.0 / xDx[1]
KmDx = xDx[0] / xDx[1]

# display results
print("\nDixon rearrangement --------------------")
print("  V = {:.5f},  error = {:.2e}".format(VDx, getRelErr(xFinal[0], VDx)))
print(" Km = {:.5f},  error = {:.2e}".format(KmDx, getRelErr(xFinal[0], KmDx)))


# The Eadie & Hofstee approximation --------------
bEH = v0Vals
AEH = np.ones( (len(v0Vals), 2) )
AEH[:,1] = -np.divide(v0Vals, sVals)

# solve linear system
xlstsq = npl.lstsq(AEH, bEH)
xEH = xlstsq[0]
VEH = xEH[0]
KmEH = xEH[1]

# display results
print("\nEadie & Hofstee rearrangement ----------")
print("  V = {:.5f},  error = {:.2e}".format(VEH, getRelErr(xFinal[0], VEH)))
print(" Km = {:.5f},  error = {:.2e}".format(KmEH, getRelErr(xFinal[0], KmEH)))


print("\n")