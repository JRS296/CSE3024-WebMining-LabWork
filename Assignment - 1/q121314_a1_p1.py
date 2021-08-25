import numpy as np
from numpy import random
import sys

print()
q12 = np.zeros(20)
q12[6] = 0
print("Create a null vector of size 20 where the 6th value should be 0 \n", q12)
print()

q13 = np.arange(1,101,1)
print("Orginal array of size 100 uisng numpy:", q13)
q13_method1  = q13[::-1]
q13_method2  = np.flipud(q13)
print("Reverse an array of size 100 using numpy (Method 1): \n", q13_method1)
print("Reverse an array of size 100 using numpy (Method 2): \n", q13_method2)
print()
  
q14 = np.arange(1,401,1).reshape(20, 20)
print("Find the minimum and maximum values of a 20X20 array using numpy ", np.min(q14), np.max(q14))
print()