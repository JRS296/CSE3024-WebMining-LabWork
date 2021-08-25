import numpy as np
import sys

print()
q1 = np.empty((2,3))
print("An empty NumPy array: \n", q1)
print()

q2 = np.zeros((6),dtype=int)
print ("NumPy array filled with all zeros: \n",q2)
print()

q3 = np.arange(20).reshape([4, 5])
print("Check a NumPy array has a particular row: \n")
print(q3)
print([0, 1, 2, 3, 4] in q3.tolist())
print([0, 1, 2, 3, 5] in q3.tolist())
print([15, 16, 17, 18, 19] in q3.tolist())
print()

print("Delete elements (+ve) values from a NumPy array: \n")
q4 = np.array([4,5,-6,7,-8,9,-10,-11,-12,13,14,-15,16,-17])
print("orginal array: \n",q4)
q4 = q4[q4 >= 0]
print("Array after removal of +ve numbers: \n",q4)
print()