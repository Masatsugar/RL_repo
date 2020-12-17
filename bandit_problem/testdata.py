import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

lam = 0.1
x = rd.exponential(0.01/lam, size=100)

y = []
for elem in x:
    s = str(elem)
    y.append(s)
print(", ".join(y))
