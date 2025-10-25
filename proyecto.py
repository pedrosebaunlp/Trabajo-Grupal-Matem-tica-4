import calcs.multiple          as multiple
import calcs.simple            as simple
import calcs.intervalos        as intervalos
import calcs.multiple as multiple
import matplotlib.pyplot       as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Dataset obtenido desde: https://archive.ics.uci.edu/dataset/242/energy+efficiency
energy_efficiency = fetch_ucirepo(id=242)

# data (as pandas dataframes)
Y = energy_efficiency.data.targets["Y1"]
X = energy_efficiency.data.features["X1"]



print("========== REGRESIÓN SIMPLE ==========")
print("========== (Y) ==========")
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

print("========== REGRESIÓN MULTIPLE==========")
betas = multiple.form_reg_mult(energy_efficiency.data.features, Y)

acc = betas[0]

for i in range(1,9):
    acc += betas[i] * energy_efficiency.data.features[f"X{i}"][0]

print(betas)
print(acc)
print(Y[0])
print(simple.beta_1_estim(X, Y, X.size) * X[0] + simple.beta_0_estim(X, Y, X.size))
