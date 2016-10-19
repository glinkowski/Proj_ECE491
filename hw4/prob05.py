import numpy as np
import math


######## ######## ####### #######
# PARAMETERS
t = np.array( [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] )
	# the time data
y = np.array( [6.8, 3.00, 1.5, 0.75, 0.48, 0.25, 0.2, 0.15] )
	# the results data
x0 = [1, 1]
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


xk = x0
diff = 1
# for r in range(1) :
while diff > 0 :

	Jk = applyJf(t, xk)
	fk = applyf(t, xk)
	# print(fk)
	# print(fk)
	rk = np.subtract(y.reshape((len(y),1)), fk)
	# print(rk)

	# print(Jk)
	# print(rk)

	# A = np.hstack((Jk, rk))
	q, r = np.linalg.qr(Jk)
	# print(q)
# 	print(r)
# #	print(np.dot(np.transpose(q), rk))
	# print(np.linalg.solve(r,np.dot(np.transpose(q),rk)))
	# print(np.dot( np.transpose(q), rk) )
	# print(np.transpose(q))
	# print(rk)
	sk = np.linalg.solve(r, np.dot( np.transpose(q), rk) )
	# print(sk)

# #	sk = np.linalg.solve(Jk, rk)
# 	sk = np.linalg.lstsq(Jk, rk)
# 	print(sk)
	xk1 = xk + sk.reshape((2,))
	# print(xk)
	# print(xk1)
	diff = np.linalg.norm( np.subtract(xk, xk1) )

	xk = xk1
	# print('iteration')
#end loop
nl_x = np.array( [xk1[0], xk1[1]] )

print(nl_x)