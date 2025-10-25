import numpy as np
from ucimlrepo import fetch_ucirepo
import multiple  #Las funciones para calculas r,r2,r2a y sigma al cuadrado
# Dataset obtenido desde: https://archive.ics.uci.edu/dataset/242/energy+efficiency
energy_efficiency = fetch_ucirepo(id=242)

Y = energy_efficiency.data.targets["Y1"]

X = energy_efficiency.data.features.to_numpy()


m, n = X.shape
X = np.column_stack([np.ones((m, 1)), X])  # Le agrega los 1 a la matriz


beta = np.zeros(n + 1)
alpha = 0.000003
pasos = 1000 

# Intento de calculo del descenso del gradiente para estimar la reg mult
#grad_norm_history = []
for _ in range(pasos):
    y_pred = X @ beta
    error = y_pred - Y
    gradient = (1/m) * (X.T @ error)
    beta -= alpha * gradient
    #grad_norm = np.linalg.norm(gradient)  
    #grad_norm_history.append(grad_norm)


print("Coeficientes β:")
for i, b in enumerate(beta):
    print(f"β{i} = {b:.4f}")

b1 = beta[0]
for i in range(1,9):
    b1+= beta[i] * energy_efficiency.data.features[f"X{i}"][0]
print("Estimacion de Y para el y1",b1) #Para ver si da un numero acorde 


print("r ",multiple.r(energy_efficiency.data.features,Y))
print("R2 ",multiple.R2(energy_efficiency.data.features,Y))
print("R2a ",multiple.R2a(energy_efficiency.data.features,Y))
print("sigma 2",multiple.sigma2(energy_efficiency.data.features,Y))


#import matplotlib.pyplot as plt
#plt.plot(grad_norm_history)
#plt.title("Norma del gradiente durante el entrenamiento")
#plt.xlabel("Iteraciones")
#plt.ylabel("||∇J(β)||")
#plt.savefig("reg2.png")