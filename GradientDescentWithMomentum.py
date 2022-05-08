import math
import numpy as np
import matplotlib.pyplot as plt
def grad(x):
    return 2*x + 10 * np.cos(x)
def cost(x):
    return x**2 + 10*np.sin(x)
def myGD(eta, x0):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, i)
def myGDM(eta, gamma, x0):
    x = [x0]
    v_old = 0
    for i in range(100):
        v_new = v_old*gamma + eta*grad(x[-1])
        x_new = x[-1] - v_new
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
        v_old = v_new
    return (x, i)

def myNAG(eta, gamma, x0):
    x = [x0]
    v_old = 0
    for i in range(100):
        v_new = v_old*gamma + eta*grad(x[-1] - gamma*v_old)
        x_new = x[-1] - v_new
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
        v_old = v_new
    return (x, i)

(x1, it1) = myGD(.1, -5)
(x2, it2) = myGD(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))

(x1, it1) = myGDM(.1, .9, -5)
(x2, it2) = myGDM(.1, .9, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))

(x1, it1) = myNAG(.1, .9, -5)
(x2, it2) = myNAG(.1, .9, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))