import numpy as np
import sympy as sym
from sympy import *

def  Degenerate_Fred_II (leftIntegralBorder, rightIntegralBorder, Lambda, KernelFunc1, KernelFunc2, rightEquationPart):
    size = len(KernelFunc1)
    column = np.matrix(np.zeros((size, 1)))
    squareMatrix = np.matrix(np.zeros((size, size)))
    for i in range (size):
        column[i] = integrate(KernelFunc2[i] * rightEquationPart, (x, leftIntegralBorder, rightIntegralBorder))
        for j in range (size):
            squareMatrix[i, j] = -Lambda * integrate(KernelFunc1[j] * KernelFunc2[i], (x, leftIntegralBorder, rightIntegralBorder))
    #print("column :\n", column)
    for i in range (size):
        squareMatrix[i, i] = 1 + squareMatrix[i, i]
    #print("square matrix :\n", squareMatrix)
    c = (squareMatrix ** (-1)) * column
    #print("c ;\n", c)
    res = 0;
    for i in range (size):
        res += c[i] * KernelFunc1[i]
    result = res * Lambda + rightEquationPart;
    return result


x = sym.Symbol('x')
a = 0
b = 0.5
lambd = 1
f = 1
for n in range(5):
    alpha = np.array([(-x**2)**i for i in range(n+1)])
    beta = np.array([((x**2)**i)/factorial(i) for i in range(n+1)])
    y=Degenerate_Fred_II(a, b, lambd, alpha, beta, f)
    print("Solution for n = ", n, " : ", y)
