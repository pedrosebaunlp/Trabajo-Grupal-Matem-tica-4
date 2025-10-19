from scipy.stats import t
import numpy as np
from . import simple

def IC_B1(x, y, alpha):
    b1 = simple.beta_1_estim(x,y,x.size)
    
    t_crit = t.ppf(1 - (alpha/2), x.size - 2) # t_(a/2;n-2)
    
    aux = t_crit * np.sqrt(simple.varianza_estim(x, y, x.size)/simple.sxx(x, x.size))
    
    print(f"IC(B1) = ({b1-aux} ; {b1+aux})")


def IC_B0(x, y, alpha):
    b0 = simple.beta_0_estim(x,y,x.size)
    
    t_crit = t.ppf(1 - (alpha/2), x.size - 2) # t_(a/2;n-2)
    
    in_sqrt = (1/x.size)+(((x.sum()/x.size)**2)/simple.sxx(x, x.size))
    
    aux = t_crit * np.sqrt(simple.varianza_estim(x, y, x.size)*in_sqrt)
    
    print(f"IC(B0) = ({b0-aux} ; {b0+aux})")


def ICM_Y(x, y, alpha):
    b0 = simple.beta_0_estim(x,y,x.size)
    b1 = simple.beta_1_estim(x,y,x.size)
    
    x_barra = x.sum()/x.size
    
    x_estrella = x_barra # ToDo: DECIDIR...
    
    t_crit = t.ppf(1 - (alpha/2), x.size - 2) # t_(a/2;n-2)
    
    in_sqrt = (1/x.size)+((x_estrella-x_barra)**2/simple.sxx(x, x.size))
    
    aux = t_crit * np.sqrt(simple.varianza_estim(x, y, x.size)*in_sqrt)
    
    print(f"ICM(Y) = ({b0+b1*x_estrella-aux} ; {b0+b1*x_estrella+aux})")


def IP_Y(x, y, alpha):    
    y_estrella = 0 # ToDo: No sé aun que es
    
    x_barra = x.sum()/x.size
    
    x_estrella = x_barra # ToDo: DECIDIR...
    
    t_crit = t.ppf(1 - (alpha/2), x.size - 2) # t_(a/2;n-2)
    
    in_sqrt = 1+(1/x.size)+((x_estrella-x_barra)**2/simple.sxx(x, x.size))
    
    aux = t_crit * np.sqrt(simple.varianza_estim(x, y, x.size)*in_sqrt)
    
    print(f"IP(Y) = ({y_estrella-aux} ; {y_estrella+aux})")