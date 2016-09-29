import numpy as np
import math


sequence = np.arange(-2,8,1)
#bad_sequence = np.linspace(42000042.00001, 42000042.9, num=10)
bad_sequence = np.linspace(42000042.00001, 42000045.00001, num=1001)

print("Calculating for seq {}".format(sequence))

# Two-pass method
# 1) Get the mean:
meanSeq = 0
numSeq = len(sequence)
for x in sequence :
	meanSeq += x
meanSeq = meanSeq / numSeq

#2) calculate std
stdFront = 1 / float(numSeq - 1)
stdSum = 0
for i in range(numSeq) :
	x = sequence[i]
	stdSum += math.pow( (x - meanSeq), 2)
#end loop
stdProd = stdFront * stdSum
var_seq_tp = math.pow(stdProd, 0.5)


# One-pass method
stdFront = 1 / float(numSeq - 1)
meanSeq = 0
stdSum = 0
for i in range(numSeq) :
	x = sequence[i]
	meanSeq += x
	stdSum += math.pow(x,2)
#end loop
meanSeq = meanSeq / numSeq
stdProd = stdFront * (stdSum - (numSeq * math.pow(meanSeq, 2) ))
var_seq_op = math.pow(stdProd, 0.5)

print("  Mean result: mean= {}".format(meanSeq))
print("  StD results: tp= {}, op= {}".format(var_seq_tp, var_seq_op))



#print("Calculating for seq {}".format(bad_sequence))
print("Calculating for the bad sequence".format(bad_sequence))

# Two-pass method
# 1) Get the mean:
meanSeq = 0
numSeq = len(bad_sequence)
for x in bad_sequence :
	meanSeq += x
meanSeq = meanSeq / numSeq

#2) calculate std
stdFront = 1 / float(numSeq - 1)
stdSum = 0
for i in range(numSeq) :
	x = bad_sequence[i]
	stdSum += math.pow( (x - meanSeq), 2)
#end loop
stdProd = stdFront * stdSum
var_bs_tp = math.pow(stdProd, 0.5)


# One-pass method
stdFront = 1 / float(numSeq - 1)
meanSeq = 0
stdSum = 0
for i in range(numSeq) :
	x = bad_sequence[i]
	meanSeq += x
	stdSum += math.pow(x,2)
#end loop
meanSeq = meanSeq / numSeq
stdProd = stdFront * (stdSum - (numSeq * math.pow(meanSeq, 2) ))
var_bs_op = math.pow(stdProd, 0.5)

print("  Mean result: mean= {}".format(meanSeq))
print("  StD results: tp= {}, op= {}".format(var_bs_tp, var_bs_op))