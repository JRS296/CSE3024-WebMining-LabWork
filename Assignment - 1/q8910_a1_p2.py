import numpy as np
import sys

print()
q8 = np.ones((3,3))
print("Original array: \n", q8)
q8 = np.pad(q8, pad_width=1, mode='constant', constant_values=0)
print("Can you add a border of a NumPy array? Do then: \n", q8)
print()

arr1 = np.array([[1, 2, 3], [3, 4, 5]])
arr2 = np.array([[1, 2, 3], [3, 4, 5]])
arr3 = np.array([[-45, 34, 4], [13, -4, -35]])
comparison1 = arr1 == arr2
comparison2 = arr2 == arr3
comparison3 = arr3 == arr1
q9_1 = comparison1.all()
q9_2 = comparison2.all()
q9_3 = comparison3.all()
print("Comparison Result of arr1 and arr2: ",q9_1)
print("Comparison Result of arr2 and arr3: ",q9_2)
print("Comparison Result of arr3 and arr1: ",q9_3)
print()

q10 = np.matrix('[6, 2; 3, 4]')
print("Find diagonal of 2D NumPy array: \n",(q10.diagonal()))
