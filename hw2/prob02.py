import numpy as np
import math

A = np.array( [	[1, math.exp(1)],
				[2, math.exp(2)],
				[3, math.exp(3)] ], dtype=np.float64 )

b = np.array( [2, 3, 5] )

sol = np.linalg.lstsq(A, b)


x = sol[0]
resid = sol[1]
rank = sol[2]
singval = sol[3]

print(A)
print(b)
print(x)