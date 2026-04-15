# Optimization for Engineers

This repository contains my lab assignments for the course **Optimization for Engineers** by Dr. Johannes Hild.
The work focuses on implementing core numerical optimization algorithms from scratch using Python.

The project covers both **unconstrained and constrained optimization**, including Newton-type methods, line search strategies, conjugate gradient solvers, and advanced techniques for least-squares and constrained problems.

---

## Implemented Topics

### LAB00 – Setup & Linear Algebra Foundations

* Environment setup and validation
* Understanding numerical routines:

  * Incomplete Cholesky decomposition
  * LLT-based linear system solver

---

### LAB01 – Newton Methods & Linear Solvers

* Implemented **Preconditioned Conjugate Gradient (PCG) Solver**
* Implemented **Newton Descent Method**
* Solved linear systems using Hessian information
* Verified with `Check01.py`

---

### LAB02 – Line Search & Quasi-Newton Methods

* Implemented **Wolfe-Powell Line Search**
* Implemented **BFGS Descent (Quasi-Newton Method)**
* Focus on global convergence behavior
* Verified with `Check02.py`

---

### LAB03 – Constrained Optimization (Box Constraints)

* Implemented **Projected Backtracking Line Search**
* Implemented **Projected Inexact Newton-CG Method**
* Handled box constraints using projection operators
* Verified with `Check03.py`

---

### LAB04 – Least-Squares Optimization

* Implemented **Least-Squares Model Formulation**
* Implemented **Levenberg–Marquardt Algorithm**
* Combined Gauss-Newton and gradient descent strategies
* Verified with `Check04.py`

---

### LAB05 – Constrained Optimization (Augmented Lagrangian)

* Implemented **Augmented Lagrangian Objective**
* Implemented **Augmented Lagrangian Descent Method**
* Solved equality-constrained optimization problems with box constraints
* Verified with `Check05.py`

---

## Technical Details

* **Language:** Python
* **Libraries:** NumPy, SciPy
* **Focus:** Numerical Optimization Algorithms
* Implementations follow algorithmic formulations from lecture notes
* Emphasis on:

  * Convergence behavior
  * Numerical stability
  * Efficient linear system solving

---

## Key Learning

This project builds a strong foundation in optimization by implementing algorithms from first principles, providing a deeper understanding of how modern optimization methods used in machine learning and engineering systems operate internally.

---

## Author

Kavya J
