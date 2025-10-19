#import calcs.multiple          as multiple
import calcs.simple            as simple
import calcs.intervalos        as intervalos
import matplotlib.pyplot       as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Dataset obtenido desde: https://archive.ics.uci.edu/dataset/242/energy+efficiency
energy_efficiency = fetch_ucirepo(id=242)

# data (as pandas dataframes)
# X = energy_efficiency.data.features["X1"]
Y = energy_efficiency.data.targets["Y1"]
#n = X.size
#simple.calculate(X,Y,X.size)
print("========== (Y1) ==========")
print(Y, "\n\n")
for i in range(1,9):
    Xi = f"X{i}"
    Xi_data = energy_efficiency.data.features[Xi]
    print(f"========== ({Xi}) ==========")
    simple.calculate(Xi_data, Y, Xi_data.size)
    intervalos.IC_B1(Xi_data, Y, 0.05)
    intervalos.IC_B0(Xi_data, Y, 0.05)
    intervalos.ICM_Y(Xi_data, Y, 0.05)
    intervalos.IP_Y(Xi_data, Y, 0.05)
#print(X)
# print(X.sum())
# print((X**2).sum())
# print(X.sum()**2)
