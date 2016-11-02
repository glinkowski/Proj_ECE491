import numpy as np
import matplotlib.pyplot as pt
import math


######## ######## ####### #######
# PARAMETERS
a = 0
b = 1
	# function range
truAns = np.pi
	# the expected answer of the integral
numRIters = 10
	# number of iterations for Romberg method
h = np.linspace((b - a), math.pow(2, numRIters), 50)
h = np.round(h)
	# (1 / step/panel size) for first three methods
maxMC = 7.0
######## ######## ####### #######


######## ######## ####### #######
# ANCILLARY FUNCTIONS

def f(x) :
	return (4 / (1 + (x*x)))
#end def ####### ####### ########



######## ######## ####### #######
# PRIMARY FUNCTION

print("")

# Newman: Store sums of quadrature rules for each h
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

	# Newman: Sum over n points for quad rules
	for i in range(1, len(xi)) :
		Msum += width * f( (xi[i-1] + xi[i]) / 2 )
		Tsum += width * (f(xi[i-1]) + f(xi[i])) / 2
		SMidVal = 4 * f( (xi[i-1] + xi[i]) / 2 )
		Ssum += width * ( f(xi[i-1]) + SMidVal + f(xi[i]) ) / 6
	#end loop
#	print("Q results: {:0.3f}, {:0.3f}, {:0.3f}".format(Msum, Tsum, Ssum))

	# Newman: store final sum for n points
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
R = np.zeros( (numRIters, numRIters) )
RNRange = np.zeros( (numRIters+1) )
RHRange = np.zeros( (numRIters+1) )

# calculate first cell
RNRange[1] = 2
RHRange[1] = (b - a) / RNRange[1]
R[0,0] = RHRange[1] * ( f(a) + f(b) )

# calc successive rows
for i in range(1, numRIters) :
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

# print(R[numRIters-1,numRIters-1])
# print(R)
# print(RNRange)
# print(RHRange)



# Implement Monte Carlo method
# MCn = np.arange(1, 9e6+1, 5e5)
MCn = np.logspace(1.0, maxMC, num=np.ceil(maxMC))
# print(MCn)
area = (b - a)
# Store the sums for the Monte Carlo integration
MC = np.zeros( len(MCn) )
for ni in range(len(MCn)) :
	# choose n random points for evaluation
	MCpoints = np.random.uniform(a, b, int(MCn[ni]))
	MCEvals = np.divide(4, np.add(1, np.power(MCpoints, 2)))
	# MC[ni] = np.mean( MCEvals )
	MC[ni] = np.mean( np.multiply(MCEvals, area) )
#end loop



# Calculate the relative error of each Quadrature rule
Merr = np.abs(np.divide( np.subtract(M, truAns), truAns))
Terr = np.abs(np.divide( np.subtract(T, truAns), truAns))
Serr = np.abs(np.divide( np.subtract(S, truAns), truAns))
# stepSize = np.divide( (b - a), h)

Rerr = np.abs(np.divide( np.subtract(RFinals, truAns), truAns))
# RStep = np.divide( (b - a), RNRange[1:len(RNRange)] )

MCerr = np.abs(np.divide( np.subtract(MC, truAns), truAns))



# Plot the results as two plots on the same figure
fig = pt.figure(figsize=(6,8))
pt1 = fig.add_subplot(2,1,1)
pt2 = fig.add_subplot(2,1,2)

# Plot the error of first four approaches
pt1.plot(h, Merr, h, Terr, h, Serr, RNRange[1:len(RNRange)], Rerr)
# pt1.plot(stepSize, Merr, stepSize, Terr, stepSize, Serr, RStep, Rerr)
pt1.legend(['Midpoint', 'Trapezoid', 'Simpson', 'Romberg'], loc=1)
pt1.set_title('Error of Newton & Romberg quad. func.s')
# pt1.set_xlabel('number of n evaluation points (NOTE: step size h = 1/n)')
# # pt1.xlabel('1 / step_size (ie: 1/h = n)')
pt1.set_xlim([0, math.pow(2, numRIters)])
pt1.set_ylabel('log10( relative error )')
pt1.set_yscale('log')

# lbl1 = [item.get_text() for item in pt1.get_xticklabels()]
lbl1 = pt1.get_xticks().tolist()
# print(lbl1)
lbl1[0] = '1'
for il in range(1, len(lbl1)) :
	lbl1[il] = '1/{:d}'.format(int(lbl1[il]))
pt1.set_xticklabels(lbl1)


# Plot the error of the Monte Carlo approach
pt2.plot(MCn, MCerr)
pt2.set_title('Error of Monte Carlo quadrature function')
pt2.set_yscale('log')
pt2.set_ylabel('log10( relative error )')
pt2.set_xlabel('step size h\n(h = 1 / n evaluation points)')
# pt2.set_xlabel('1 / step_size \n (NOTE: number of eval points n = 1/h)')
# pt2.set_xlabel('number of evaluation points \n (NOTE: step size h = 1/n)')
pt2.set_xscale('log')

# lbl2 = np.log10(pt2.get_xticks().tolist())
# # lbl2 = [item for item in pt2.get_xticklabels()]
# print(lbl2)
# for il in range(len(lbl2)) :
# 	lbl2[il] = '1e-{:d}'.format(int(lbl2[il]))
# pt2.set_xticklabels(lbl2)

lbl2 = pt2.get_xticks().tolist()
lbl2 = np.divide(1, lbl2)
pt2.set_xticklabels(lbl2)

pt.show()



# Print an answer to the question:
#	Print one or two sentences explaining why the error
#	stops improving for some of the methods.
print("For the Simpson's and Romberg rules, the error effectively bottoms \n"
	+ "out just under the 1e-15 range. In these cases I believe the program \n"
	+ "is running up against the limitations of floating-point arithmetic.")
print("That is, as smaller and smaller numbers are added, some of the \n"
	+ "digits of the mantissa will fall outside the represntable range \n"
	+ "and will be lost. Similarly, when multiplied and divided, the \n"
	+ "resulting mantissa may need to be truncated.")