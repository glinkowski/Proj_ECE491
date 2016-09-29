import numpy as np


# A = np.matrix('0.1, 0.2, 0.3; 0.4, 0.5, 0.6; 0.7, 0.8, 0.9')
# B = np.matrix('0.1; 0.3; 0.5')

# A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
# B = np.array([[0.1], [0.3], [0.5]])

A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
B = np.array([0.1, 0.3, 0.5])

print(A)
print(B)

# solve Ax=B
solution = np.linalg.solve(A, B)
print("Ax=b ... x= \n{}".format(solution))

condition = np.linalg.cond(A)
print("cond(A)= {}".format(condition))

print("expected correct digits: {}".format( max(0, (16 - np.log10(condition))) ))
correct = 0