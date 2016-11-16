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
fig = pt.figure(figsize=(12,12))

# e = 0.0
# First row: x vs t
ax11 = fig.add_subplot(431)
ax11.plot(tArray, sol0[:,0])
ax11.set_title('x position, e = 0.0')
ax11.set_ylabel('x')
ax11.set_xlabel('time')
ax11.set_xlim([0, tf])
# First row: y vs t
ax12 = fig.add_subplot(432)
ax12.plot(tArray, sol0[:,2])
ax12.set_title('y position, e = 0.0')
ax12.set_ylabel('y')
ax12.set_xlabel('time')
ax12.set_xlim([0, tf])
# First row: x vs t
ax13 = fig.add_subplot(433)
ax13.plot(sol0[:,0], sol0[:,2])
ax13.set_title('y vs x, e = 0.0')
ax13.set_ylabel('y')
ax13.set_xlabel('x')
ax13.set_xlim([-2, 2])
ax13.set_ylim([-2, 2])

# e = 0.5
# Second row: x vs t
ax21 = fig.add_subplot(434)
ax21.plot(tArray, sol1[:,0])
ax21.set_title('x position, e = 0.5')
ax21.set_ylabel('x')
ax21.set_xlabel('time')
ax21.set_xlim([0, tf])
# Second row: y vs t
ax22 = fig.add_subplot(435)
ax22.plot(tArray, sol1[:,2])
ax22.set_title('y position, e = 0.5')
ax22.set_ylabel('y')
ax22.set_xlabel('time')
ax22.set_xlim([0, tf])
# Second row: x vs t
ax23 = fig.add_subplot(436)
ax23.plot(sol1[:,0], sol1[:,2])
ax23.set_title('y vs x, e = 0.5')
ax23.set_ylabel('y')
ax23.set_xlabel('x')
ax23.set_xlim([-2, 2])
ax23.set_ylim([-2, 2])

# e = 0.9
# Third row: x vs t
ax31 = fig.add_subplot(437)
ax31.plot(tArray, sol2[:,0])
ax31.set_title('x position, e = 0.9')
ax31.set_ylabel('x')
ax31.set_xlabel('time')
ax31.set_xlim([0, tf])
# Third row: y vs t
ax32 = fig.add_subplot(438)
ax32.plot(tArray, sol2[:,2])
ax32.set_title('y position, e = 0.9')
ax32.set_ylabel('y')
ax32.set_xlabel('time')
ax32.set_xlim([0, tf])
# Third row: x vs t
ax33 = fig.add_subplot(439)
ax33.plot(sol2[:,0], sol2[:,2])
ax33.set_title('y vs x, e = 0.9')
ax33.set_ylabel('y')
ax33.set_xlabel('x')
ax33.set_xlim([-2, 2])
ax33.set_ylim([-2, 2])

# Fourth row: conservation of energy
# ax41 = fig.add_subplot(427)
ax41 = fig.add_subplot(4,3,10)
ax41.plot(tArray, conv2[:,0])
ax41.set_title('Conservation of Energy, e = 0.9')
ax41.set_ylabel('energy')
ax41.set_xlabel('time')
ax41.set_xlim([0, tf])
# Fourth row: conservation of energy
# ax42 = fig.add_subplot(428)
ax42 = fig.add_subplot(4,3,12)
ax42.plot(tArray, conv2[:,1])
ax42.set_title('Conservation of Momentum, e = 0.9')
ax42.set_ylabel('angular momentum')
ax42.set_xlabel('time')
ax42.set_xlim([0, tf])

pt.tight_layout()
pt.show()


# NOT USED: Share figures with multiple plots
#	looks nicer, but not what is asked for

# # First axis: x vs t
# ax1 = fig.add_subplot(321)
# ax1.plot(tArray, sol0[:,0], tArray, sol1[:,0], tArray, sol2[:,0])
# ax1.set_title('x vs t')
# ax1.set_ylabel('x')
# ax1.set_xlabel('time')
# ax1.set_xlim([0, tf])
# ax1.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# # Second axis: y vs t
# ax2 = fig.add_subplot(322)
# ax2.plot(tArray, sol0[:,2], tArray, sol1[:,2], tArray, sol2[:,2])
# ax2.set_title('y vs t')
# ax2.set_ylabel('y')
# ax2.set_xlabel('time')
# ax2.set_xlim([0, tf])
# ax2.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# # Third axis: y vs x
# ax3 = fig.add_subplot(323)
# ax3.plot(sol0[:,0], sol0[:,2], sol1[:,0], sol1[:,2], sol2[:,0], sol2[:,2])
# ax3.set_title('y vs x')
# ax3.set_ylabel('y')
# ax3.set_xlabel('x')
# ax3.set_xlim([-2.5, 2.5])
# ax3.set_ylim([-1.5, 1.5])
# ax3.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# # Fourth axis: conservation of energy
# ax4 = fig.add_subplot(325)
# #ax4.plot(tArray, conv0[:,0], tArray, conv1[:,0], tArray, conv2[:,0])
# ax4.plot(tArray, conv2[:,0])
# ax4.set_title('Conservation of Energy, e = 0.9')
# ax4.set_ylabel('energy')
# ax4.set_xlabel('time')
# ax4.set_xlim([0, tf])
# #ax4.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# # Fourth axis: conservation of energy
# ax5 = fig.add_subplot(326)
# #ax5.plot(tArray, conv0[:,1], tArray, conv1[:,1], tArray, conv2[:,1])
# ax5.plot(tArray, conv2[:,1])
# ax5.set_title('Conservation of Momentum, e = 0.9')
# ax5.set_ylabel('angular momentum')
# ax5.set_xlabel('time')
# ax5.set_xlim([0, tf])
# #ax5.legend(['e = 0.0', 'e = 0.5', 'e = 0.9'])

# pt.tight_layout()
# pt.show()