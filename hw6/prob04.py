from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as pt
import math



######## ######## ####### #######
# PARAMETERS

eValues = [0.0, 0.5, 0.9]
	# the initial value parameter e

t0 = 0
tf = (2 * math.pi) * 4
tInval = 0.05
tArray = np.arange(t0, tf+tInval, tInval)
	# time interval & step

######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

def setInitConds(e) :
	fy10 = 1 - e
	fy20 = 0
	fy30 = 0
	fy40 = math.pow( ((1 + e) / (1 - e)), 0.5 )

	return [fy10, fy20, fy30, fy40]
#end def ####### ####### ########

def rCalc(y) :
	fy1, fy2, fy3, fy4 = y

	inside = math.pow(fy1, 2) + math.pow(fy3, 2)
	rFull = math.pow(inside, 0.5)

#	print(rFull)
	return float(rFull)
#end def ####### ####### ########

def conservEqs(y) :
	fy1, fy2, fy3, fy4 = y

	energy = (math.pow(fy2, 2) + math.pow(fy4, 2)) / 2.0
	energy = energy - 1 / rCalc(y)

	momentum = (fy1 * fy4) - (fy3 * fy2)

	return [energy, momentum]
#end def ####### ####### ########

def calcConserv(sol) :

	length, width = sol.shape
	consVals = np.zeros( (length, 2) )

	for i in range(length) :
		consVals[i,:] = conservEqs(sol[i,:])

	return consVals
#end def ####### ####### ########

# The system of equations to pass to solver
def fsys(y, t) :
	# unpack the inputs
	fy1, fy2, fy3, fy4 = y

	# The derivatives
	fd1 = fy2
	fd2 = - fy1 / math.pow(rCalc(y), 3)
	fd3 = fy4
	fd4 = - fy3 / math.pow(rCalc(y), 3)

	return [fd1, fd2, fd3, fd4]
#end def ####### ####### ########


######## ######## ####### #######
# PRIMARY FUNCTION

# Apply the ode solver to the equations
sol0 = odeint(fsys, setInitConds(eValues[0]), tArray)
#conv0 = calcConserv(sol0)
sol1 = odeint(fsys, setInitConds(eValues[1]), tArray)
#conv1 = calcConserv(sol1)
sol2 = odeint(fsys, setInitConds(eValues[2]), tArray)
conv2 = calcConserv(sol2)


# Plot the desired figures
fig = pt.figure(figsize=(14,16))

# First axis: x vs t
ax1 = fig.add_subplot(321)
ax1.plot(tArray, sol0[:,0], tArray, sol1[:,0], tArray, sol2[:,0])
ax1.set_title('x vs t')
ax1.set_ylabel('x')
ax1.set_xlabel('time')
ax1.set_xlim([0, tf])
ax1.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# Second axis: y vs t
ax2 = fig.add_subplot(322)
ax2.plot(tArray, sol0[:,2], tArray, sol1[:,2], tArray, sol2[:,2])
ax2.set_title('y vs t')
ax2.set_ylabel('y')
ax2.set_xlabel('time')
ax2.set_xlim([0, tf])
ax2.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# Third axis: y vs x
ax3 = fig.add_subplot(323)
ax3.plot(sol0[:,0], sol0[:,2], sol1[:,0], sol1[:,2], sol2[:,0], sol2[:,2])
ax3.set_title('y vs x')
ax3.set_ylabel('y')
ax3.set_xlabel('x')
ax3.set_xlim([-2.5, 2.5])
ax3.set_ylim([-1.5, 1.5])
ax3.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# Fourth axis: conservation of energy
ax4 = fig.add_subplot(325)
#ax4.plot(tArray, conv0[:,0], tArray, conv1[:,0], tArray, conv2[:,0])
ax4.plot(tArray, conv2[:,0])
ax4.set_title('Conservation of Energy, e = 0.9')
ax4.set_ylabel('energy')
ax4.set_xlabel('time')
ax4.set_xlim([0, tf])
#ax4.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# Fourth axis: conservation of energy
ax5 = fig.add_subplot(326)
#ax5.plot(tArray, conv0[:,1], tArray, conv1[:,1], tArray, conv2[:,1])
ax5.plot(tArray, conv2[:,1])
ax5.set_title('Conservation of Momentum, e = 0.9')
ax5.set_ylabel('angular momentum')
ax5.set_xlabel('time')
ax5.set_xlim([0, tf])
#ax5.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

pt.tight_layout()
pt.show()