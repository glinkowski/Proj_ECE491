import numpy as np
import matplotlib.pyplot as pt


# the min/max size of the matrices to examine
minSize = 2
maxSize = 24


# Create a bunch of matrices
#	maList size = (index + 2)
maList = list()
for size in range(minSize, maxSize) :
	# main diag is all 1, upper = -1, lower = 0
	matrix = np.triu( np.multiply(-1, np.ones((size,size))), k=1 )
	matrix = np.add(matrix, np.identity(size))
	maList.append(matrix)
#end loop

# print(maList[4])


# Get sigma max/min ratio for each matrix
#	after SVD, max is entry 0, min is last entry (-1)
xVals = range(minSize, maxSize)
yVals = list()
for matrix in maList :
	s = np.linalg.svd(matrix, compute_uv=False)
	ratio = s[0] / s[-1]
	# print("{}, {}, {}".format(s[0], s[-1], ratio))
	yVals.append(ratio)
#end loop


# Plot the results
pt.plot(xVals, yVals)


pt.show()