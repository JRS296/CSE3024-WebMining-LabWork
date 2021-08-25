import numpy as np
from numpy import random
import sys

print()
q9 = random.rand()
print("Random number between 0 to 1: ", q9)
print()

q10 = np.linspace(1, 10, 10) #linespace defined by third parameter = 10
print("10 points that are space linearly from each: ", q10)
print()
  
arr1 = np.array([[1, 2, 3], [3, 4, 5]])
arr2 = np.array([[1, 2, 3], [3, 4, 5]])
arr3 = np.array([[-45, 34, 4], [13, -4, -35]])
  
comparison1 = arr1 == arr2
comparison2 = arr2 == arr3
comparison3 = arr3 == arr1
q11_1 = comparison1.all()
q11_2 = comparison2.all()
q11_3 = comparison3.all()
  
print("Comparison Result of arr1 and arr2: ",q11_1)
print("Comparison Result of arr2 and arr3: ",q11_2)
print("Comparison Result of arr3 and arr1: ",q11_3)