import numpy as np
import numpy.linalg as nl



A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])



print("\n1) A^-1 = \n{}".format(nl.inv(A)))
print("\n2) det(A) = {}".format(nl.det(A)))
print("\n3) rank(A) = {}".format(nl.matrix_rank(A)))
