import numpy as np


######## ######## ####### #######
# PARAMETERS
t = np.array( [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] )
	# the time data
y = np.array( [6.8, 3.00, 1.5, 0.75, 0.48, 0.25, 0.2, 0.15] )
	# the results data
x0 = [1, 1]
	# starting vector for Gauss-Newton
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

def f(t,x) :
	y = x[0] * np.exp(x[1] * t)
	return y
#end def ####### ####### ########
def Jf(t,x) :
	j11 = np.exp(x[1] * t)
	j12 = x[0] * t * np.exp(x[1] * t)
	return np.array( [j11, j12] )
#end def ####### ####### ########

def applyf(t,x) :
	fv = np.zeros( (len(t), 1) )

	for i in range(len(t)) :
		fv[i,:] = f(t[i], x)
	#end loop

	return fv
#end def ####### ####### ########
def applyJf(t,x) :
	J = np.zeros( (len(t), len(x)) )

	for i in range(len(t)) :
		J[i,:] = Jf(t[i], x)
	#end loop

	return J
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("")

# print(applyJf(t,x0))
# print(applyf(t,x0))

# The NON-Linear implementation, using QR factorization
xk = x0
diff = 1
while diff > 0 :

	Jk = applyJf(t, xk)

	fk = applyf(t, xk)
	rk = np.subtract(y.reshape((len(y),1)), fk)

	q, r = np.linalg.qr(Jk)
	sk = np.linalg.solve(r, np.dot( np.transpose(q), rk) )

	xk1 = xk + sk.reshape((2,))

	diff = np.linalg.norm( np.subtract(xk, xk1) )

	xk = xk1
#end loop
nl_x = np.array( [xk1[0], xk1[1]] )

print("Non-Linear solution: {}".format(nl_x))


# The Log-Linearized least squares calculation
b = np.log(y)
A = np.ones( (len(t), 2) )
A[:,1] = t

xLog = np.linalg.lstsq(A, b)[0]
l_x = np.array( [np.exp(xLog[0]), xLog[1]] )

# print(xLog)
print("Linear log solution: {}".format(l_x))