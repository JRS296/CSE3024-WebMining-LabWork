import numpy as np
from numpy import random
import sys
import math

print()
q15 = np.random.rand(50)#use radint for whole numbers (i.e. integers)
print("Randomly generated array: \n", q15)
print("Mean value of a randomly generated array of size 50: ", np.mean(q15))
print()

q16 = np.zeros(400).reshape(20, 20)
for i in range(0,20):
    for j in range(0,20):
        if (i==0 or j==0 or i == 19 or j ==19):
            q16[i,j] = 1
print("Create a 20 X20 array filled with zeros at all borders and all 1’s inside: \n", q16)
print()

n = math.nan
q17 = np.zeros(100).reshape(10, 10)
for i in range(0,10):
    for j in range(0,10):
        if (i==10 or 10==0):
            q17[i,j] = n
print("Create an array of size 10X10 with 10 element valued as nan: \n", q17)
print()