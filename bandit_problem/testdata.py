import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

lam = 0.1   # 1分あたり0.1回発生する。
x = rd.exponential(0.01/lam, size=50)
print(x)
