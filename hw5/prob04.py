import numpy as np
import matplotlib.pyplot as pt
import math


######## ######## ####### #######
# PARAMETERS
h = np.linspace(1, 250, 50)
h = np.round(h)
	# step/panel size
a = 0
b = 1
	# function range
truAns = np.pi
	# the expected answer of the integral
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

def f(x) :
	return (4 / (1 + (x*x)))
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("")

# Store sums of quadrature rules for each h
#	Midpoint, Trapazoid, Simpson
M = np.zeros( (len(h)) )
T = np.zeros( (len(h)) )
S = np.zeros( (len(h)) )


for ih in range(len(h)) :

	Msum = 0
	Tsum = 0
	Ssum = 0

	width = (b-a) / h[ih]
	xi = np.linspace(a, b, h[ih]+1)
	# print(len(xi))
	# print(width)

	for i in range(1, len(xi)) :
		Msum += width * f( (xi[i-1] + xi[i]) / 2 )
		Tsum += width * (f(xi[i-1]) + f(xi[i])) / 2
		SMidVal = 4 * f( (xi[i-1] + xi[i]) / 2 )
		Ssum += width * ( f(xi[i-1]) + SMidVal + f(xi[i]) ) / 6
	#end loop

#	print("Q results: {:0.3f}, {:0.3f}, {:0.3f}".format(Msum, Tsum, Ssum))

	M[ih] = Msum
	T[ih] = Tsum
	S[ih] = Ssum
#end loop

# print(xi)
# print(h)
# print(M)
# print(T)
# print(S)
# print(truAns)



# Implement Romberg method

# create arrays to hold data
numIters = 8
R = np.zeros( (numIters, numIters) )
RNRange = np.zeros( (numIters+1) )
RHRange = np.zeros( (numIters+1) )

# calculate first cell
RNRange[1] = 2
RHRange[1] = (b - a) / RNRange[1]
R[0,0] = RHRange[1] * ( f(a) + f(b) )

# calc successive rows
for i in range(1, numIters) :
	RNRange[i+1] = RNRange[i] * 2
	RHRange[i+1] = RHRange[i] / float(2)

	# first col: half of prev cell + trap rule on intermediate points
	RZSum = 0
	# sum rule over intermed pts, ie: (.25, .75), (.125, .375, .625, .875), ...
	for z in range(1, int(RNRange[i])) :
		point = a + (2*z-1) * RHRange[i]
		if point > b :
			break
		RZSum += f(point)
		# print(point)
	# print(RZSum)
	# print((R[(i-1),0] / 2) + RZSum / 2)
	R[i,0] = (R[(i-1),0] / 2) + RZSum * RHRange[i]

	# calc rest of row, based on previous cells
	for j in range(1, (i+1)) :
		R[i,j] = R[i,(j-1)]
		R[i,j] += (R[i,(j-1)] - R[(i-1),(j-1)]) / (math.pow(4, j) - 1)
#end loop
RFinals = np.diag(R)

# print(R[numIters-1,numIters-1])
# print(R)
# print(RNRange)
# print(RHRange)


# Calculate the relative error of each Quadrature rule
Merr = np.abs(np.divide( np.subtract(M, truAns), truAns))
Terr = np.abs(np.divide( np.subtract(T, truAns), truAns))
Serr = np.abs(np.divide( np.subtract(S, truAns), truAns))
stepSize = np.divide( (b - a), h)

Rerr = np.abs(np.divide( np.subtract(RFinals, truAns), truAns))


# Plot the error of each approach
pt.plot(h, Merr, h, Terr, h, Serr, RNRange[1:len(RNRange)], Rerr)
pt.legend(['Midpoint', 'Trapezoid', 'Simpson', 'Romberg'], loc=1)
pt.title('Error of quadrature functions')
pt.xlabel('number of intervals')
pt.ylabel('log10( relative error )')
pt.yscale('log')
pt.show()