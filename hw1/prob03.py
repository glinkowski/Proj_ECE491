import numpy as np
import math


# Fill these arrays with the correct values
positive = np.ones(5)
negative = np.ones(5)


printFlag = True


xValPos = [1, 5, 10, 15, 20]


if printFlag :
    print("Solving for the negative values ...")
for i in range(len(xValPos)) :
#for i in range(1) :
    x = xValPos[i]
    preVal = 0
    curVal = 1
    exp = 1
    while abs(curVal - preVal) > 0 :
        preVal = curVal
        incr = math.pow(x, exp) / float( math.factorial(exp) )
        curVal = curVal + incr
        exp += 1
    #end loop
    clcSol = np.divide(1, curVal)
    truSol = np.exp(-x)
    if printFlag :
        print("  e^-{} ~ {} vs true= {}".format(x, clcSol, truSol))
    err = abs(truSol - clcSol) / truSol
    negative[i] = err
#end loop
if printFlag :
    print("Error = {}".format(negative))