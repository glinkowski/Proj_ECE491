import numpy as np
import math
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad


######## ######## ####### #######
# PARAMETERS

intgXA = -1
intgXB = 1
intgYA = -1
intgYB = 1
	# the area of integration

phiXA = 2
phiXB = 10
phiYA = 2
phiYB = 10
	# the domain of the function phi

numMPPlot = 1e4
	# number of ponts to use in one dimension
# numMPOne = 1e5
# 	# number of points to use for single calculation
numGrid = 32
	# number of x, y input points to plot

######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

# the function being integrated at point (10,10)
def f10(x, y) :

	arg1 = math.pow( (10 - x), 2)
	arg2 = math.pow( (10 - y), 2)
	final = math.pow( (arg1 + arg2), -0.5)

	return final
#end def ####### ####### ########
# the function being integrated at point (2,2)
def f2(x, y) :

	arg1 = math.pow( (2 - x), 2)
	arg2 = math.pow( (2- y), 2)
	final = math.pow( (arg1 + arg2), -0.5)

	return final
#end def ####### ####### ########

# Monte Carlo applied to the integral
def phiMonteCarlo(xHat, yHat, MCAccuracy) :

	area = (intgXB - intgXA) * (intgYB - intgYA)

	x = np.random.uniform(intgXA, intgXB, int(MCAccuracy))
	y = np.random.uniform(intgYA, intgYB, int(MCAccuracy))

	arg1 = np.power( np.subtract(xHat, x), 2 )
	arg2 = np.power( np.subtract(yHat, y), 2 )
	funcVal = np.power( np.add(arg1, arg2), -0.5 )

	volume = np.multiply(funcVal, area)
	avgVal = np.mean(volume)

	return avgVal
#end def ####### ####### ########


######## ######## ####### #######
# PRIMARY FUNCTION

print("")

# Calculate phi(10,10) using built-in library
pot_10_10 = dblquad(f10, intgXA, intgXB, lambda x:intgYA, lambda x:intgYB)[0]
# pot_2_2 = dblquad(f2, intgXA, intgXB, lambda x:intgYA, lambda x:intgYB)[0]


# Create the grid over which to plot phi(x,y)
xPoints = np.linspace(2, 10, numGrid)
yPoints = np.linspace(2, 10, numGrid)
X, Y = np.meshgrid(xPoints, yPoints)


# Calculate phi values using Monte Carlo method
phiVals = np.zeros( (len(xPoints), len(yPoints)) )
for xi in range(len(xPoints)) :
	for yi in range(len(yPoints)) :
		phiVals[xi, yi] = phiMonteCarlo(xPoints[xi], yPoints[yi], numMPPlot)
#end with
print("phi(10,10) using scipy library: {:.6f}".format(pot_10_10))
print("phi(10,10) w/ loose Monte Carlo: {:.6f}".format(phiVals[len(xPoints)-1,len(yPoints)-1]))


# Plot the function phi over X, Y
fig = pt.figure(figsize=(8,12))
ax1 = fig.add_subplot(211, projection='3d')
ax1.plot_surface(X, Y, phiVals)
ax1.set_xlabel('x-axis')
ax1.set_ylabel('y-axis')
ax1.set_zlabel('phi(x,y)')
ax1.set_title('surface of Phi, y=(2,10), x=(2,10)')

ax2 = fig.add_subplot(212)
cax = ax2.imshow(phiVals, extent=(2,10,10,2))
ax2.set_xlabel('y-axis')
ax2.set_ylabel('x-axis')
ax2.set_title('surface of Phi, y=(2,10), x=(2,10)')
cbar = fig.colorbar( cax,
	ticks=np.linspace(np.amin(phiVals), np.amax(phiVals), 9) )

pt.show()