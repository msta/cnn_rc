import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(x):
    a = []
    for item in x:

        a.append(1/(1+math.exp(-item)))
    return a

def dif_sigmoid(x):
    sig_res = sigmoid(x)
    return [d * (1 - d) for d in sig_res]


def relu(x):
    return [max(xx, 0) for xx in x]

def dif_relu(x):
    difs = []
    for r in x:
        if r > 0:
            difs.append(1)
        else:
            difs.append(0)
    return difs


def tanh(x):
    a = []
    x = [2*xx for xx in x]
    x = sigmoid(x)
    for item in x:
        a.append( 2 * item - 1) 
    return a


def dif_tanh(x):
    tans = tanh(x)
    a = []
    for item in tans:
        a.append(1 - (item*item))
    return a

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

x = np.arange(-10, 10., 0.2)

plt.figure(1)

plt.subplot(131)

plt.plot(x,sigmoid(x), x, dif_sigmoid(x))
plt.axis([-5, 5, 0, 1])
plt.legend([r"$S(x) = \frac{1}{1 + e^{-x}}$", r"$S'(x) = S(x) * (1 - S(x)) $"])


plt.subplot(133)

plt.plot(x, relu(x), x, dif_relu(x))
plt.axis([-1, 1, -0.1, 1.2])
plt.legend([r"$ReLU(x) = max(0, x) $", r"$ ReLU'(x) = \big\{ 1 x > 0 \\ 0 \leq 0 $"])

plt.subplot(132)

plt.plot(x, tanh(x), x, dif_tanh(x))
plt.axis([-5, 5, 0, 1])
plt.legend([r"$tanh(x) = 2 * S(2*x) - 1$", r"$S'(x) = 1 - tanh(x)^2 $"])

plt.show()

