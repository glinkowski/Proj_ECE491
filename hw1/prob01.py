import numpy as np
import math

# Fill these arrays with the correct values
positive = np.ones(5)
negative = np.ones(5)

xValPos = [1, 5, 10, 15, 20]
xValNeg = np.multiply(xValPos, -1)

print("Solving for the positive values ...")
for i in range(len(xValPos)) :
#for i in range(1) :
    x = xValPos[i]
    preVal = 0
    curVal = 1
    exp = 1
#    print("using x={}".format(x))
    while abs(curVal - preVal) > 0 :
        preVal = curVal
        incr = math.pow(x, exp) / float( math.factorial(exp) )
#        print("    {} / {}".format(math.pow(x, exp)), math.factorial(exp))
        curVal = curVal + incr
        exp += 1
#        print("  curVal= {}, incr= {}".format(curVal, incr))
    #end loop
    
    truSol = np.exp(x)
    print("  e^{} ~ {} vs true= {}".format(x, curVal, truSol))
    err = (truSol - curVal) / truSol
    positive[i] = err
#end loop
print("Error = {}".format(positive))


print("Solving for the negative values ...")
for i in range(len(xValNeg)) :
#for i in range(1) :
    x = xValNeg[i]
    preVal = 0
    curVal = 1
    exp = 1
#    print("using x={}".format(x))
    while abs(curVal - preVal) > 0 :
        preVal = curVal
        incr = math.pow(x, exp) / float( math.factorial(exp) )
#        print("    {} / {}".format(math.pow(x, exp)), math.factorial(exp))
        curVal = curVal + incr
        exp += 1
#        print("  curVal= {}, incr= {}".format(curVal, incr))
    #end loop
    
    truSol = np.exp(x)
    print("  e^{} ~ {} vs true= {}".format(x, curVal, truSol))
    err = (truSol - curVal) / truSol
    negative[i] = err
#end loop
print("Error = {}".format(negative))