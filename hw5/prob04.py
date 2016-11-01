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



# # Implement Romberg method
# hR = 2 * float(b - a)	# the step size (will halve each time)
# numIters = 8
# evaluations = np.zeros( (numIters, numIters) )
# i = -1
# for step in range(numIters) :
# 	i += 1
# 	hR = hR / 2.0
# 	points = np.arange(a, (b+hR), hR)

# 	for ip in range(len(points)) :



# # try again
# numIters = 4
# evaluations = np.zeros( (numIters, numIters) )
# rh = 2 * (b - a)
# for rn in range(numIters) :
# 	rh = rh / float(2)

# 	if rn == 0 :

# 	for j in range(rn) :
# ###


def f2(x) :
	return f(x)
	# return 1 / float(x)
##########
# a = 1
# b = 2



# try again
numIters = 8
R = np.zeros( (numIters, numIters) )
RNRange = np.zeros( (numIters+1) )
RHRange = np.zeros( (numIters+1) )
RNRange[1] = 2
RHRange[1] = (b - a) / RNRange[1]
R[0,0] = RHRange[1] * ( f2(a) + f2(b) )
for i in range(1, numIters) :
	RNRange[i+1] = RNRange[i] * 2
	RHRange[i+1] = RHRange[i] / float(2)

	# RZSum = 0
	# for z in range(1, int(math.pow(2, (i-1)))) :
	# 	RZSum += f2(a + (RHRange[i] * ((2*z) - 1)))
	# R[i,0] = (0.5 * R[(i-1),0]) + (RHRange[i] * RZSum)

	RZSum = 0
	print("1 to 2^n-1 ... {}".format(RNRange[i]))
	for z in range(1, int(RNRange[i])) :
		point = a + (2*z-1) * RHRange[i]
		if point > b :
			break
		RZSum += f2(point)
		print(point)
	# print(RZSum)
	# print((R[(i-1),0] / 2) + RZSum / 2)
	R[i,0] = (R[(i-1),0] / 2) + RZSum * RHRange[i]
	# R[i,0] = (R[(i-1),0] / 2) + RZSum * RHRange[i+1]


	for j in range(1, (i+1)) :
		R[i,j] = R[i,(j-1)]
		R[i,j] += (R[i,(j-1)] - R[(i-1),(j-1)]) / (math.pow(4, j) - 1)

#end loop
RFinals = np.diag(R)


print(R[numIters-1,numIters-1])
print(R)
print(RNRange)
print(RHRange)




# Calculate the relative error of each Quadrature rule
Merr = np.abs(np.divide( np.subtract(M, truAns), truAns))
Terr = np.abs(np.divide( np.subtract(T, truAns), truAns))
Serr = np.abs(np.divide( np.subtract(S, truAns), truAns))
stepSize = np.divide( (b - a), h)

Rerr = np.abs(np.divide( np.subtract(RFinals, truAns), truAns))


# Plot the error of each approach
# pt.plot(stepSize, Merr, stepSize, Terr, stepSize, Serr)
# pt.plot(h, Merr, h, Terr, h, Serr)
# pt.legend(['Midpoint', 'Trapezoid', 'Simpson'], loc=1)
pt.plot(h, Merr, h, Terr, h, Serr, RNRange[1:len(RNRange)], Rerr)
pt.legend(['Midpoint', 'Trapezoid', 'Simpson', 'Romberg'], loc=1)
pt.title('Error of quadrature functions')
pt.xlabel('number of intervals')
# pt.xlabel('step size')
pt.ylabel('log10( relative error )')
pt.yscale('log')
pt.show()
