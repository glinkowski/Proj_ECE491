from scipy.integrate import ode
from scipy.integrate import odeint
import numpy as np



######## ######## ####### #######
# PARAMETERS

c = 1
d = 5
params = [c, d]
	# function parameters

y10 = 95
y20 = 5
y30 = 0
yinits = [y10, y20, y30]
	# initial conditions

t0 = 0
tf = 1
	# integration range

y1 = np.zeros( (3,) )
	# output array
y = [y10, y20, y30]


######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# The system of equations to pass to solver
def fsys(y, t, params) :
	# unpack the inputs
	fy1, fy2, fy3 = y
	fc, fd = params

	fd1 = -fc * fy1 * fy2
	fd2 = (fc * fy1 * fy2) - (fd * fy2)
	fd3 = fd * fy2

	return [fd1, fd2, fd3]
#end def ####### ####### ########


######## ######## ####### #######
# PRIMARY FUNCTION


tArray = np.arange(t0, tf, 0.05)

solution = odeint(fsys, yinits, tArray, args=(params,))

print(solution)