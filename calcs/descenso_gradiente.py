import numpy as np
from ucimlrepo import fetch_ucirepo

# Dataset obtenido desde: https://archive.ics.uci.edu/dataset/242/energy+efficiency
energy_efficiency = fetch_ucirepo(id=242)

Y = energy_efficiency.data.targets["Y1"]

X = energy_efficiency.data.features.to_numpy()

# Agregamos una columna de 1s para el intercepto (β0)
m, n = X.shape
X = np.column_stack([np.ones((m, 1)), X])  # ahora X tiene (m, n+1)

# Inicialización
beta = np.zeros(n + 1)
alpha = 0.000003
#0.0000003
#0.000001
epochs = 1000

# Descenso del gradiente
grad_norm_history = []
for _ in range(epochs):
    y_pred = X @ beta
    error = y_pred - Y
    gradient = (1/m) * (X.T @ error)
    beta -= alpha * gradient
    grad_norm = np.linalg.norm(gradient)  # magnitud del gradiente
    grad_norm_history.append(grad_norm)




import matplotlib.pyplot as plt
plt.plot(grad_norm_history)
plt.title("Norma del gradiente durante el entrenamiento")
plt.xlabel("Iteraciones")
plt.ylabel("||∇J(β)||")
plt.savefig("reg2.png")

print("Coeficientes β:")
for i, b in enumerate(beta):
    print(f"β{i} = {b:.4f}")

b1 = beta[0]
for i in range(1,9):
    b1+= beta[i] * energy_efficiency.data.features[f"X{i}"][0]
print(b1)


#Prod = beta @ X

"""
plt.plot(cost_history)
plt.title("Descenso del gradiente - Evolución del costo")
plt.xlabel("Iteraciones")
plt.ylabel("Costo J(β)")
plt.savefig("reg.png")"""
