import numpy as np



######## ######## ####### #######
# PARAMETERS
year = np.array( [1900.0, 1910.0, 1920.0, 1930.0, 1940.0, 1950.0, 1960.0, 1970.0, 1980.0] )
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

# def basis1(t, j)
# 	phi = t ** (j - 1)
# 	return phi
# #end def ####### ####### ########
# def basis2(t, j)
# 	phi = (t - 1900) ** (j - 1)
# 	return phi
# #end def ####### ####### ########
# def basis3(t, j)
# 	phi = (t - 1940) ** (j - 1)
# 	return phi
# #end def ####### ####### ########
# def basis4(t, j)
# 	phi = ((t - 1940) / 40) ** (j - 1)
# 	return phi
# #end def ####### ####### ########

# def applyBasis1(year) :
# 	dim = len(year)
# 	# place the time data into the matrix
# 	matrix = np.repeat( year.reshape((dim,1)), 9, axis=1)
# 	# create the expnonent (j - 1)
# 	jmin1 = np.array(range(dim))
# 	# calculate the array
# 	matrix = np.power(matrix, jmin1)

# 	return matrix
# #end def ####### ####### ########
# def applyBasis2(year) :
# 	dim = len(year)
# 	# place the time data into the matrix
# 	matrix = np.repeat( year.reshape((dim,1)), 9, axis=1)
# 	# create the expnonent (j - 1)
# 	jmin1 = np.array(range(dim))
# 	# calculate the array
# 	matrix = np.subtract(matrix, 1900)
# 	matrix = np.power(matrix, jmin1)

# 	return matrix
# #end def ####### ####### ########
# def applyBasis3(year) :
# 	dim = len(year)
# 	# place the time data into the matrix
# 	matrix = np.repeat( year.reshape((dim,1)), 9, axis=1)
# 	# create the expnonent (j - 1)
# 	jmin1 = np.array(range(dim))
# 	# calculate the array
# 	matrix = np.subtract(matrix, 1940)
# 	matrix = np.power(matrix, jmin1)

# 	return matrix
# #end def ####### ####### ########
# def applyBasis4(year) :
# 	dim = len(year)
# 	# place the time data into the matrix
# 	matrix = np.repeat( year.reshape((dim,1)), 9, axis=1)
# 	# create the expnonent (j - 1)
# 	jmin1 = np.array(range(dim))
# 	# calculate the array
# 	matrix = np.subtract(matrix, 1940)
# 	matrix = np.divide(matrix, 40.0)
# 	matrix = np.power(matrix, jmin1)

# 	return matrix
# #end def ####### ####### ########
def applyBasis(t, subVal, divVal) :
	dim = len(t)
	# place the time data into the matrix
	matrix = np.repeat( t.reshape((dim,1)), 9, axis=1)
	# create the expnonent (j - 1)
	jmin1 = np.array(range(dim))
	# calculate the array
	matrix = np.subtract(matrix, subVal)
	matrix = np.divide(matrix, float(divVal))
	matrix = np.power(matrix, jmin1)

	return matrix
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("")

mat1 = applyBasis(year, 0, 1)
mat2 = applyBasis(year, 1900, 1)
mat3 = applyBasis(year, 1940, 1)
mat4 = applyBasis(year, 1940, 40)