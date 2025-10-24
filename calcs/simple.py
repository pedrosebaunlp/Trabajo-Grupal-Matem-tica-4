import numpy as np

def sxx(x,n):
    sum_sq = (x**2).sum()
    sq_sum = (x.sum())**2
    return sum_sq - (sq_sum/n)

def syy(y,n):
    sum_sq = (y**2).sum()
    sq_sum = (y.sum())**2
    return sum_sq - (sq_sum/n)

def sxy(x,y,n):
    return ((x*y).sum())-(x.sum()*y.sum()/n)

def ssr(x,y,n):
    return syy(y,n)-((sxy(x,y,n)**2)/sxx(x,n))

def beta_1_estim(x,y,n):
    return sxy(x,y,n)/sxx(x,n)

def beta_0_estim(x,y,n):
    return y.mean()-(beta_1_estim(x,y,n)*x.mean())

def varianza_estim(x,y,n):
    return ssr(x,y,n)/(n-2)

def R_cuadrado(x,y,n):
    return 1 - (ssr(x,y,n)/syy(y,n))

def r(x,y,n):
    rsqr = R_cuadrado(x,y,n)
    return np.sqrt(R_cuadrado(x,y,n))

def y_estim(x,y,n):
    return (beta_1_estim(x,y,n)*x)+beta_0_estim(x,y,n)

def calculate(x,y,n):
    print(f"""
        \n Sxx:{sxx(x,n)},\n
        \n Syy:{syy(y,n)},\n
        \n Sxy:{sxy(x,y,n)},\n
        \n B1:{beta_1_estim(x,y,n)},\n
        \n B0:{beta_0_estim(x,y,n)},\n
        \n y:{beta_1_estim(x,y,n)}x+({beta_0_estim(x,y,n)}),\n
        \n SSr:{ssr(x,y,n)},\n
        \n sigma^2:{varianza_estim(x,y,n)},\n
        \n R2:{R_cuadrado(x,y,n)},\n
        \n r:{r(x,y,n)}\n
    """)
