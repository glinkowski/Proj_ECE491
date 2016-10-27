import numpy as np
import matplotlib.pyplot as pt
# import scipy as sc
import scipy.interpolate as si



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

	# reverse the coefficient order
	revCoeff = coeff[::-1]

	# calculate at each x value
	for ix in range(len(xVals)) :
		x = xVals[ix]

		#each round = (next_coeff + x*(prev_value))
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


# 2) Comput condition number of each matrix
cond1 = np.linalg.cond(mat1)
cond2 = np.linalg.cond(mat2)
cond3 = np.linalg.cond(mat3)
cond4 = np.linalg.cond(mat4)

print("Condition numbers: {:.3e}, {:.3e}, {:.3e}, {:.3e}".format(
	cond1, cond2, cond3, cond4))
print("Using matrix 'mat4'...")
#Note: use mat4 for the following ...


# 3) Evaluate interpolant using Horner's
coeffs = np.linalg.solve(mat4, pop)
# print(coeffs)

# Define the range of years (x-axis)
pStart = year[0]# - 1
pStop = year[len(year)-1]# + 1
plotX = np.linspace(pStart, pStop, (pStop - pStart + 1))
# scale the years before evaluating
plotXScaled = np.divide( np.subtract(plotX, 1940), float(40))

# Use Horner's Nested evaluation scheme
plotYa = applyHorners(plotXScaled, coeffs)
# # un-scale the years
# plotX = np.add( np.multiply(plotX, 40), 1940)


# 4) monotone Hermite cubic interpolant
pchip = si.PchipInterpolator(year, pop, extrapolate=True)
plotYb = pchip.__call__(plotX)


# 5) cubic spline interpolant
unvspline = si.UnivariateSpline(year, pop)
plotYc = unvspline.__call__(plotX)


# 6) Extrapolate each method to 1990, get error
exYear = 1990
exPop = 248709873
popa = applyHorners( [((exYear - 1940) / float(40))], coeffs)[0]
popb = pchip.__call__(exYear)
popc = unvspline.__call__(exYear)
print("Projected to 1990: {:,.0f}; {:,.0f}; {:,.0f}".format(
	popa, float(popb), float(popc)))

err_polynomial = np.abs(popa - exPop) / exPop
err_hermite = np.abs(popb - exPop) / exPop
err_spline = np.abs(popc - exPop) / exPop
print("Relative error: {:.3f}; {:.3f}; {:.3f}".format(
	err_polynomial, err_hermite, err_spline))


# 7) Round the data & get resulting error




# Plot the interpolant function w/ data
pt.plot(plotX, plotYa)
pt.plot(plotX, plotYb)
pt.plot(plotX, plotYc)
pt.plot(year, pop, 'sr')

pt.legend(['scaled monotonic', 'Hermite cubic',
	'cubic spline', 'original data'], loc=0)

pt.title('Plotting the Interpolants')
pt.ylabel('population')
pt.xlabel('year')
pt.xticks(year, year)
pt.show()
