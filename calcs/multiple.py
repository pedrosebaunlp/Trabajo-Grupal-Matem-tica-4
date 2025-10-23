import numpy  as np
import pandas as pd

def x_matriz(x):
    n = x["X1"].size # Todas tienen el mismo largo
    return np.column_stack((
        np.ones(n),
        x["X1"],
        x["X2"],
        x["X3"],
        x["X4"],
        x["X5"],
        x["X6"],
        x["X7"],
        x["X8"])
    )

def form_reg_mult(x,y):
    X_mat = x_matriz(x)
    XT = X_mat.T
    return np.linalg.inv(XT @ X_mat) @ XT @ y

def pred_mod(X,Y):
    beta = form_reg_mult(X,Y)
    X_mat = x_matriz(X)
    return  X_mat @ beta

def syy(y):
    y_sum = y.sum()
    return ((y - y_sum/y.size)**2).sum()

def ssr(x,y):
    return ((y - pred_mod(x,y))**2).sum()

def R2(x,y):
    return 1-ssr(x,y)/syy(y)

def R2a(x,y):
    k = 8 # Hay 8 variables independientes
    n = x.size
    return 1-(1-R2(x,y))*((n-k)/(n-k-1))

def r(x,y):
    return np.sqrt(R2a(x,y))
def sigma2(x,y):
    return ssr(x,y)/(x.size-4)





