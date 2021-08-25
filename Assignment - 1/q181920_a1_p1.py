import numpy as np
import sys

print()
q18 =  np.diag(1+np.arange(8,5,-1), k = -1)
print("Create a 4X4 matrix and the values just below the diagonal is 9 8 7: \n", q18)
print()

q19 = np.zeros ((8,8), dtype=int)
q19[1::2, ::2]= 1
q19[::2, 1::2] = 1
print ("Create a check board pattern using numpy: \n",q19)
print()

q20_1 = np.dtype(np.int32) 
q20_2 = np.dtype(np.float64) 
print("Print the dtype of int32 and float64 data type: \n", q20_1, q20_2)
print()