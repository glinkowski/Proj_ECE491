import numpy as np



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

def origGramSchmidt(A) :

	# numCols = A.shape[1]
	# numRows = A.shape[0]
	(numRows, numCols) = A.shape
	print(numRows)

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

#def modGramSchmidt(A) :






######## ######## ####### #######
# PRIMARY FUNCTION

hilbert = createHilbertArray()

testIdx = 2

#print(hilbert[testIdx])

Q, R = origGramSchmidt(hilbert[testIdx])
print(Q)

Q1, R1 = np.linalg.qr(hilbert[testIdx])
print(Q1)

# # Q^T * Q should ~= I
print( np.dot( np.transpose(Q), Q ) )
# print( np.linalg.norm(np.identity(testIdx + 2) - np.dot( np.transpose(Q), Q)) )
# #print(R)

E = -1 * np.log10( np.linalg.norm(np.identity(testIdx + 2) - np.dot( np.transpose(Q), Q)))
print(E)

