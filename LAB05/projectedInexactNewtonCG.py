# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# from Kavya Jayaramaiah
# IDM ID: iz81eniq

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA
import copy

from projectionInBox import projectionInBox
from simpleValleyObjective import simpleValleyObjective

def matrnr():
    # set your matriculation number here
    matrnr = 23398482
    return matrnr


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp = P.project(x0)

    # INCOMPLETE CODE STARTS
    n_k = min(0.5, np.sqrt(np.linalg.norm(xp-P.project(xp-f.gradient(xp))))) * np.linalg.norm(xp-P.project(xp-f.gradient(xp)))

    while np.linalg.norm(xp-P.project(xp-f.gradient(xp))) > eps:

        x_j = copy.deepcopy(xp)
        r_j = f.gradient(xp)

        d_j= -copy.deepcopy(r_j)

        while np.linalg.norm(r_j) > n_k:

            d_a = PHA.projectedHessApprox(f,P,xp,d_j)

            rho = d_j.T @ d_a

            if rho <= eps*np.linalg.norm(d_j)**2:
                break
            t_j = (np.linalg.norm(r_j)**2)/ rho
            x_j = x_j + t_j*d_j

            rold = copy.deepcopy(r_j)
            r_j = rold + t_j*d_a

            beta_j = (np.linalg.norm(r_j)**2) / (np.linalg.norm(rold)**2)
            d_j = -r_j + beta_j*d_j

        if rho <= eps*np.linalg.norm(d_j)**2:
            gradx = f.gradient(xp)
            dk = -gradx
            
        else:
            
            dk = x_j - xp

        t_k = PB.projectedBacktrackingSearch(f, P, xp, dk)
        xp = P.project(xp + t_k * dk)
        n_k = min(0.5, np.sqrt(np.linalg.norm(xp-P.project(xp-f.gradient(xp))))) * np.linalg.norm(xp-P.project(xp-f.gradient(xp)))

        countIter += 1


        
    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp

p = np.array([[1], [1]])
myObjective = simpleValleyObjective(p)
a = np.array([[1], [1]])
b = np.array([[2], [2]])
myBox = projectionInBox(a, b)
x0 = np.array([[2], [2]], dtype=float)
eps = 1.0e-3
xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
print(xmin)
