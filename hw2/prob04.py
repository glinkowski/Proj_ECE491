import numpy as np

A = np.array( [	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1],
				[1, -1, 0, 0],
				[1, 0, -1, 0],
				[1, 0, 0, -1],
				[0, 1, -1, 0],
				[0, 1, 0, -1],
				[0, 0, 1, -1] ], dtype=np.float64 )

b = np.array( [2.95, 1.74, -1.45, 1.32, 1.23, 4.45, 1.61, 3.21, 0.45, -2.75] )

x_actual = np.array( [2.95, 1.74, -1.45, 1.32] )

sol = np.linalg.lstsq(A, b)


x = sol[0]
resid = sol[1]
rank = sol[2]
singval = sol[3]

rel_errors = np.divide( np.abs(np.subtract(x, x_actual)), np.abs(x))

#print(A)
#print(b)
print(x)
print(rel_errors)