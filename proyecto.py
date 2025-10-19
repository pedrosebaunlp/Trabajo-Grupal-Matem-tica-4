#import calcs.multiple          as multiple
import calcs.simple            as simple
import matplotlib.pyplot       as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Dataset obtenido desde: https://archive.ics.uci.edu/dataset/242/energy+efficiency
energy_efficiency = fetch_ucirepo(id=242)

# data (as pandas dataframes)
X = energy_efficiency.data.features["X1"]
Y = energy_efficiency.data.targets["Y1"]
#n = X.size
simple.calculate(X,Y,X.size)
#print(X)
# print(X.sum())
# print((X**2).sum())
# print(X.sum()**2)
