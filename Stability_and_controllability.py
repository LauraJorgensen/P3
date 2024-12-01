#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:41:40 2024

@author: laurajorgensen
"""

import numpy as np

# Define the system parameters (Appendix B)
mc = 6.28
mp = 0.25
l = 0.3325
g = 9.82
Fc_c = 3.2
Fc_p = 4.1e-3
alpha = 0.5e-3

# Define matrix A
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0,(mp*g)/mc, -Fc_c/mc, -(Fc_p+alpha)/(l*mc)],
    [0, (g*(mp+mc))/(l*mc), -Fc_c/(l*mc), -((Fc_p+alpha)*(mc+mp))/(l**2*mc*mp)]
])

# Define matrix B
B = np.array([
    [0],
    [0],
    [1/mc],
    [1/(l*mc)]
])

# Calculate C, eigenvalues of A and rank of C
C = np.hstack([B, A @ B, A @ (A @ B), A @ A @ A @ B])
eigenvalues = np.linalg.eigvals(A)
rank_C = np.linalg.matrix_rank(C)

print("\nEigenvalues of A:\n",eigenvalues, "\n\nControllability Matrix C:\n", C, "\n\nRank of C:",rank_C)