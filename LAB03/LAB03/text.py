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
    grad_xp = f.gradient(xp)
    n_k = min(0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - grad_xp)))) * np.linalg.norm(xp - P.project(xp - grad_xp))

    while np.linalg.norm(xp - P.project(xp - grad_xp)) > eps:
        x_j = copy.deepcopy(xp)
        r_j = grad_xp
        d_j = -r_j

        while np.linalg.norm(r_j) > n_k:
            d_a = PHA.projectedHessApprox(f, P, x_j, d_j)
            rho = d_j.T @ d_a

            if rho <= eps * np.linalg.norm(d_j)**2:
                break

            t_j = (r_j.T @ r_j) / rho
            x_j = P.project(x_j + t_j * d_j)

            r_j_old = r_j
            r_j = f.gradient(x_j)

            if np.linalg.norm(r_j) <= n_k:
                break

            beta_j = (r_j.T @ r_j) / (r_j_old.T @ r_j_old)
            d_j = -r_j + beta_j * d_j

        if np.linalg.norm(r_j) <= n_k:
            d_k = -grad_xp
        else:
            d_k = x_j - xp

        t_k = PB.projectedBacktrackingSearch(f, P, xp, d_k)
        xp = P.project(xp + t_k * d_k)
        grad_xp = f.gradient(xp)
        n_k = min(0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - grad_xp)))) * np.linalg.norm(xp - P.project(xp - grad_xp))

        countIter += 1

    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', stationarity)

    return xp

# Example test case
p = np.array([[1], [1]])
myObjective = simpleValleyObjective(p)
a = np.array([[1], [1]])
b = np.array([[2], [2]])
myBox = projectionInBox(a, b)
x0 = np.array([[2], [2]], dtype=float)
eps = 1.0e-3
xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
print(xmin)
