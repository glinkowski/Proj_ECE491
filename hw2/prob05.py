import numpy as np


# Generate hilbert matrices
hilbMaster = np.zeros((13,13))
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