import numpy as np
import matplotlib.pyplot as pt



def createHilbertArray() :
	# Generate hilbert matrices
	hilbMaster = np.zeros((13,13), dtype=np.float64)
	for i in range(13) :
		for j in range(13) :
			hilbMaster[i,j] = 1 / ( (i+1) + (j+1) - 1)
	#end loop
	#print(hilbMaster)

	hilbert = list()
	for n in range(2, 13) :
		hilbert.append( hilbMaster[0:n,0:n] )
	#end loop
	#print(hilbert[0])

	return hilbert
#end def ######## ####### #######


def origGramSchmidt(B) :

	A = np.copy(B)
	(numRows, numCols) = A.shape

	# The arrays to return
	Q = np.zeros( A.shape, dtype=np.float64 )
	R = np.zeros( A.shape, dtype=np.float64 )

	for k in range(numCols) :
		Q[:,k] = A[:,k]
		for j in range(k) :
			R[j,k] = np.dot( Q[:,j], A[:,k] )
			Q[:,k] = np.subtract( Q[:,k], np.multiply(R[j,k], Q[:,j]) )
			# r = np.dot( Q[:,j], A[:,k] )
			# Q[:,k] = np.subtract( Q[:,k], np.multiply(r, Q[:,j]) )
		#end loop
		R[k,k] = np.linalg.norm( Q[:,k] )
		Q[:,k] = np.divide( Q[:,k], R[k,k] )
		# r = np.linalg.norm( Q[:,k] )
		# Q[:,k] = np.divide( Q[:,k], r )
	#end loop

	return Q, R
#end def ######## ####### #######


def modGramSchmidt(B) :

	A = np.copy(B)
	(numRows, numCols) = A.shape

	# The arrays to return
	Q = np.zeros( A.shape, dtype=np.float64 )
	R = np.zeros( A.shape, dtype=np.float64 )

	for k in range(numCols) :
		R[k,k] = np.linalg.norm(A[:,k])
		Q[:,k] = np.divide(A[:,k], R[k,k])
		for j in range( (k+1), numCols ) :
			R[k,j] = np.dot(Q[:,k],A[:,j])
			A[:,j] = np.subtract( A[:,j], np.multiply(R[k,j], Q[:,k]))
	#end loop

	return Q, R
#end def ######## ####### #######


def getError(Q) :

	E = -1 * np.log10( np.linalg.norm(np.identity(Q.shape[0]) - np.dot( np.transpose(Q), Q)))
	E = max(E, 0)

	return E
#end def ######## ####### #######


def Householder(B) :

	A = np.copy(B)
	(numRows, numCols) = A.shape
	# print("A")
	# print(A)

#	H = np.zeros(A.shape, dtype=np.float64)
	H = np.identity(numCols)
	e = np.identity(numCols)

	for n in range(numCols) :
		alpha = np.linalg.norm(A[:,n])
		# print(alpha)
		v = A[:,n] - np.multiply(-alpha, e[:,n])
		vT = v.reshape(len(v), 1)
#		print(v)
		H_i = e - np.multiply( (2 / np.dot(v, vT)), np.multiply(vT, v) )
		# print("H_{}".format(n))
		# print(H_i)
		A = np.dot(H_i, A)
		# print("H_i * A")
		# print(A)
		H = np.dot(H_i, H)
	#end loop
#	print(H)

	Q = np.transpose(H)
	return Q
#end def ######## ####### #######


######## ######## ####### #######
# PRIMARY FUNCTION

hilbert = createHilbertArray()

testIdx = 6

# #print(hilbert[testIdx])

# Q, R = origGramSchmidt(hilbert[testIdx])
# print(Q)

# Q, R = modGramSchmidt(hilbert[testIdx])
# print(Q)

# Q1, R1 = np.linalg.qr(hilbert[testIdx])
# print(Q1)

# # # Q^T * Q should ~= I
# print( np.dot( np.transpose(Q), Q ) )
# # print( np.linalg.norm(np.identity(testIdx + 2) - np.dot( np.transpose(Q), Q)) )
# # #print(R)

# E = -1 * np.log10( np.linalg.norm(np.identity(testIdx + 2) - np.dot( np.transpose(Q), Q)))
# print(E)


Q, R = origGramSchmidt(hilbert[testIdx])
print(Q)
print( np.dot( np.transpose(Q), Q ) )
print(getError(Q))

Q = Householder(hilbert[testIdx])
print(Q)
print( np.dot( np.transpose(Q), Q ) )
E = getError(Q)
print(E)



x = list()
y_origGS = list()
y_modGS = list()
y_selfGS = list()
y_HH = list()

entry = 2
for matrix in hilbert :

	x.append(entry)
	entry += 1

	Q, R = origGramSchmidt(matrix)
	E = getError(Q)
	y_origGS.append(E)

	Q, R = origGramSchmidt(Q)
	E = getError(Q)
	y_selfGS.append(E)

	Q, R = modGramSchmidt(matrix)
	E = getError(Q)
	y_modGS.append(E)

	Q = Householder(matrix)
	E = getError(Q)
	y_HH.append(E)
#end loop

# Plot the stuff
pt.plot(x, y_origGS, x, y_modGS, x, y_selfGS, x, y_HH)
pt.xlim([0,12])
pt.xlabel('Hilbert matrix size nxn')
pt.ylabel('digits of accuracy')
pt.legend(['classical Gram-Schmidt', 'modified Gram-Schmidt',
		'Gram-Schmidt on self', 'Householder transform'])
pt.title('Digits of Accuracy vs Matrix Dimensions')
pt.savefig('linkows2-hw2Prob5.png')
pt.show()