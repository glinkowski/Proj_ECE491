from scipy.integrate import ode
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as pt



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
tInval = 0.005
tArray = np.arange(t0, tf+tInval, tInval)
	# integration range

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

# Apply the ode solver to the equations
solution = odeint(fsys, yinits, tArray, args=(params,))
lenArray = len(tArray)

# print(solution)
# print(tArray)

# The final solution at t=1
y1 = solution[lenArray-1, :]
print("At time t=1, solution: {}".format(y1))

# Draw the plots
fig = pt.figure()
ax = fig.add_subplot(111)
ax.plot(tArray, solution[:,0], tArray, solution[:,1], tArray, solution[:,2])
ax.set_xlabel('time')
ax.set_ylabel('solutions to y')
ax.set_title('the Kermack-McKendrick model')
ax.legend(['sol. of y1', 'sol. of y2', 'sol. of y3'])

pt.show()