import numpy as np
# import scipy.integrate as sci
import matplotlib.pyplot as pt
# from mpl_toolkits.mplot3d import Axes3D


######## ######## ####### #######
# PARAMETERS

# number of points in a dimension
n = 27
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION
print("")


# Prepare figure for plotting
fig = pt.figure(figsize=(12, 3))
# fig = pt.figure()
# useCmap = 'PuOr'
useCmap = 'bwr'
# useCmap = 'coolwarm'

# set up the parameters
# h = 1 / (n-1)
# x = np.arange(0, 1+h, h)
# y = np.copy(x)


# part one: one dimension
n = 16

matrixA = np.multiply(np.eye(n), -2)
matrixA = np.add(matrixA, np.eye(n, k=1))
matrixA = np.add(matrixA, np.eye(n, k=-1))


ax1 = fig.add_subplot(131)
topVal = np.amax(np.abs(matrixA))
cax1 = ax1.imshow(matrixA, interpolation='None',
	cmap=useCmap, vmin=(-topVal), vmax=topVal)
ax1.set_title('one dimension, k={}'.format(n))
cbar1 = fig.colorbar( cax1,
	ticks = np.linspace(np.amin(matrixA), np.amax(matrixA), 4) )



# part two: two dimensions
# n2 = n**2
n1 = n
n2 = np.sqrt(float(n1))
# print(n2)
n2 = int(n2)

n2 = 5
n1 = n2**2

matrixB = np.multiply(np.eye(n1), -4)
matrixB = np.add(matrixB, np.eye(n1, k=1))
matrixB = np.add(matrixB, np.eye(n1, k=-1))
matrixB = np.add(matrixB, np.eye(n1, k=n2))
matrixB = np.add(matrixB, np.eye(n1, k=-n2))
# matrixB = -np.log(matrixB)



ax2 = fig.add_subplot(132)
minB = np.amin(matrixB)
maxB = np.amax(matrixB)
topVal = np.amax(np.abs([minB, maxB]))
cax2 = ax2.imshow(matrixB, interpolation='None',
	cmap=useCmap, vmin=(-topVal), vmax=topVal)
ax2.set_title('two dimensions, k={}'.format(n2))
# cbar2 = fig.colorbar( cax2,
# 	ticks = np.linspace(-topVal, topVal, (topVal*2 + 1)) )
cbar2 = fig.colorbar( cax2,
	ticks = np.linspace(minB, maxB, (maxB - minB + 1)) )


# part three: three dimensions
# n2 = n**2
n1 = n
n3 = np.power(float(n1), (1/3))
n2 = n3 * n3
# print(n2)
# print(n3)
n2 = int(n2)
n3 = int(n3)

n3 = 4
n2 = n3**2
n1 = n2 * n3


matrixC = np.multiply(np.eye(n1), -6)
matrixC = np.add(matrixC, np.eye(n1, k=1))
matrixC = np.add(matrixC, np.eye(n1, k=-1))
matrixC = np.add(matrixC, np.eye(n1, k=n2))
matrixC = np.add(matrixC, np.eye(n1, k=-n2))
matrixC = np.add(matrixC, np.eye(n1, k=n3))
matrixC = np.add(matrixC, np.eye(n1, k=-n3))
# matrixC = -np.log(matrixC)




ax3 = fig.add_subplot(133)
minC = np.amin(matrixC)
maxC = np.amax(matrixC)
topVal = np.amax(np.abs([minC, maxC]))
cax3 = ax3.imshow(matrixC, interpolation='None',
	cmap=useCmap, vmin=(-topVal), vmax=topVal)
ax3.set_title('three dimensions, k={}'.format(n3))
# cbar2 = fig.colorbar( cax2,
# 	ticks = np.linspace(-topVal, topVal, (topVal*2 + 1)) )
cbar3 = fig.colorbar( cax3,
	ticks = np.linspace(minC, maxC, (maxC - minC + 1)) )






pt.savefig('prob06.png')
pt.show()

print('\n')