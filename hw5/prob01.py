import numpy as np
import matplotlib.pyplot as pt
import math



######## ######## ####### #######
# PARAMETERS
year = np.array( [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980] )
	# the time data
pop = np.array( [76212168,
				92228496,
				106021537,
				123202624,
				132164569,
				151325798,
				179323175,
				203302031,
				226542199] )
	# the population data
######## ######## ####### #######



######## ######## ####### #######
# ANCILLARY FUNCTIONS

def applyBasis(t, subVal, divVal) :
	dim = len(t)
	# place the time data into the matrix
	matrix = np.repeat( t.reshape((dim,1)), 9, axis=1)
	# print(matrix[0:3,0:3])
	# create the expnonent (j - 1)
	jmin1 = np.array(range(dim))
	# calculate the array
	matrix = np.subtract(matrix, float(subVal))
	# print(matrix[0:2,0:2])
	matrix = np.divide(matrix, float(divVal))
	# print(matrix[0:2,0:2])
	matrix = np.power(matrix, jmin1)
	# print(matrix[0:3,0:3])

	return matrix
#end def ####### ####### ########

def applyHorners(xVals, coeff) :
	yVals = np.zeros( (len(xVals)), dtype=np.float64)

	revCoeff = coeff[::-1]

	for ix in range(len(xVals)) :
		x = xVals[ix]
		temp = revCoeff[0]
		for ic in range(1, len(revCoeff)) :
			temp = revCoeff[ic] + (x * temp)
		yVals[ix] = temp

	return yVals
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("")

# 1) Create matrices from defined basis functions
mat1 = applyBasis(year, 0, 1)
mat2 = applyBasis(year, 1900, 1)
mat3 = applyBasis(year, 1940, 1)
mat4 = applyBasis(year, 1940, 40)

# print(mat1[0:2,0:2])

# 2) Comput condition number of each matrix
cond1 = np.linalg.cond(mat1)
cond2 = np.linalg.cond(mat2)
cond3 = np.linalg.cond(mat3)
cond4 = np.linalg.cond(mat4)

print("Condition numbers: {:.3e}, {:.3e}, {:.3e}, {:.3e}".format(
	cond1, cond2, cond3, cond4))
print("Using matrix 'mat4'...")
#Note: use mat4 for the following ...

# 3) compute & plot polynomial interpolant
coeffs = np.linalg.solve(mat4, pop)
# print(coeffs)

# Define the range of years (x-axis)
pStart = year[0]# - 1
pStop = year[len(year)-1]# + 1
plotX = np.linspace(pStart, pStop, (pStop - pStart + 1))
# plotX = year
plotX = np.divide( np.subtract(plotX, 1940), float(40))
# plotX = year
# year = np.divide( np.subtract(year, 1940), float(40))
# plotY = np.zeros( (plotX.shape), dtype=np.float64 )

# print(plotX)
# print(year)

# # troubleshoot horners
# coeffs = [3, 2, 1]
# plotX = [-1, 0, 1, 2]
# plotY = np.zeros( (len(plotX)) )

# calculate the corresponding populations (y-axis)
# revCoeff = coeffs[::-1]
# print(revCoeff)
# for ix in range(len(plotX)) :
# # for ix in range(1,2) :
# 	x = plotX[ix]
# 	# print(x)
# 	temp = revCoeff[0]
# 	for ic in range(1, len(revCoeff)) :
# 		temp = revCoeff[ic] + (x * temp)
# 		# print("  {}, {}".format(revCoeff[ic], temp))
# 	plotY[ix] = temp
# #end loop

# plotX = year

# Use Horner's Nested evaluation scheme
plotY = applyHorners(plotX, coeffs)
# print(plotY)
plotX = np.add( np.multiply(plotX, 40), 1940)


# for ix in range(len(plotX)) :
# 	temp = coeffs[0]
# 	for ic in range(1, len(coeffs)) :
# 		temp = temp + (coeffs[ic] * math.pow(plotX[ix], ic))
# 	plotY[ix] = temp
# #end loop
# print(plotY)


# Plot the interpolant function w/ data
pt.plot(plotX, plotY)
#pt.scatter(year, pop)
pt.plot(year, pop, 'sr')
pt.legend(['interpolant', 'original data'])
# pt.plot(plotX, plotY)
# pt.plot(year, pop)
pt.title('Plotting the Interpolant')
pt.ylabel('population')
pt.xlabel('year')
pt.xticks(year, year)
# xlabels = pt.get_xticklabels()
# print(xlabels)
pt.show()
