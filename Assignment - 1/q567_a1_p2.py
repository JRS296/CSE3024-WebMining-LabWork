import numpy as np
import sys

print()
q5 = arr = np.array([[2, 8, 9, 4], [9, 4, 9, 4],[4, 5, 9, 7],[2, 9, 4, 3]])
count = repr(arr).count("9, 4")
print("Find count of a particular sequence has occurred in a NumPy Array: \n", q5)
print("Count of sequence 9,4: ",count)
print()

print ("Search a maximum no of element that is there in a NumPy array: ")
q6 = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3, ])
print("Array: \n",q6)
x = np.bincount(q6)
maximum = max(x)
for i in range(len(x)):
    if x[i] == maximum:
        print("Most frequent value in above array: ",i)
print()

q7_1 = np.arange(1, 10).reshape(3, 3)
q7_2 = np.arange(10, 19).reshape(3, 3)
# axis = 0 ----> row-wise
q7 = np.concatenate((q7_1, q7_2), axis=0)
print("Is it possible to merge two 2 dimensional array (NumPy)?Do then: \n", q7)
print()
