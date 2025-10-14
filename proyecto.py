import calcs.multiple          as multiple
import calcs.simple            as simple
import matplotlib.pyplot       as plt

from ucimlrepo import fetch_ucirepo

# Dataset obtenido desde: https://archive.ics.uci.edu/dataset/242/energy+efficiency
energy_efficiency = fetch_ucirepo(id=242)

# data (as pandas dataframes)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# Definimos como Var. Respuesta a Y1
print(simple)

