#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:22:21 2024

@author: laurajorgensen
"""

# Import package
import numpy as np
from scipy.integrate import solve_ivp
import control
from Plot_function import create_plot_sim       

# Define the system parameters (Appendix B)
mc = 6.28        # kg
mp = 0.25        # kg
l = 0.3325       # m
g = 9.82         # m/s^2
Fcp = 0.0041     # Nm
alpha = 0.0005   # Nms
Fcc = 3.2        # N
KT = 0.0934      # Nm/A
r = 0.028        # m
d_maks = 0.385   # m
i_maks = 78.9    # A
u_maks = 263.18  # Force

# Initial conditions
x_1 = [0, np.pi / 8, 0, 0]
x_2 = [0, np.pi / 10, 0, 0]
x_3 = [0, np.pi / 12, 0, 0]

# Linear model
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, (mp * g) / mc, -Fcc / mc, -(Fcp + alpha) / (l * mc)], 
              [0,( g * (mp + mc)) / (l * mc), -Fcc / (l * mc), -(Fcp + alpha) * (mc + mp) / (mc * mp * l**2)]])

B = np.array([[0],
              [0],
              [1 / mc],
              [1 / (mc * l)]])

# Define Q and R
Q = np.diag([1 / d_maks**2, 0, 0, 0])
R = np.diag([1 / u_maks**2])

# Feedback gain
def K_matrix(Q_matrix,R_matrix):
    K, _, _ = control.lqr(A, B, Q_matrix, R_matrix)
    K=-K
    K = K.flatten()
    return K

# Non-linear model
def equations(t, z, K):
    z1, z2, z3, z4 = z
    u = K @ np.array([z1, z2, z3, z4])  
    d = mc + mp * (np.cos(z2))**2
    Ffc = np.tanh(z3)*Fcc
    Ffp = np.tanh(z4)*Fcp+alpha*z4
    
    dz1 = z3
    dz2 = z4
    dz3 = (
        mp * g * np.sin(z2) * np.cos(z2)
        - Ffp / l * np.cos(z2)
        - mp * l * z4**2 * np.sin(z2)
        - Ffc
        + u
    ) / d
    dz4 = (
        mp * g * np.sin(z2)*np.cos(z2)**2
        - Ffp / l * np.cos(z2)**2
        - mp * l * z4**2 * np.sin(z2) * np.cos(z2)
        - Ffc*np.cos(z2)
        + u * np.cos(z2) 
    )/ (l * d) + g / l * np.sin(z2) - Ffp/(mp*l**2)
    return [dz1, dz2, dz3, dz4]

# Function to simulate the system
def simulate_system(initial_condition, t, K):
    sol = solve_ivp(
        equations,
        [t[0], t[-1]],
        initial_condition,
        t_eval=t,
        args=(K,)
    )
    return sol.t, sol.y

# Function to calculate input and convert to Ampere
def calculate_input(state, K):
    u = np.dot(K, state) 
    return u / (KT / r) 

# Time vector
T = np.linspace(0, 2, 1000)

# Simulate the system 
time_1, state_1 = simulate_system(x_1, T, K_matrix(Q,R))
time_2, state_2 = simulate_system(x_2, T, K_matrix(Q,R))
time_3, state_3 = simulate_system(x_3, T, K_matrix(Q,R))

# Calculate input
u_1 = calculate_input(state_1, K_matrix(Q,R))
u_2 = calculate_input(state_2, K_matrix(Q,R))  
u_3 = calculate_input(state_3, K_matrix(Q,R))

# Create plots 
def plot_og_print(K):
    create_plot_sim('Time [s]', 'Displacement [m]', [time_1, time_2, time_3], [state_1[0], state_2[0], state_3[0]], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-0.42,0.42], guidelines=[d_maks, -d_maks])
    create_plot_sim('Time [s]', 'Angle [rad]', [time_1, time_2, time_3],[state_1[1], state_2[1], state_3[1]], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-np.pi/6, np.pi/6])
    create_plot_sim('Time [s]', 'Velocity [m/s]', [time_1, time_2, time_3], [state_1[2], state_2[2], state_3[2]],labels=['$x_1$', '$x_2$', '$x_3$'],  y_limits=[-2.5, 1.5]) 
    create_plot_sim('Time [s]', 'Angular velocity [rad/s]', [time_1, time_2, time_3],[state_1[3], state_2[3], state_3[3]], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-6, 4]) 
    create_plot_sim('Time [s]', 'Current [A]', [time_1, time_2, time_3],[u_1, u_2, u_3], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-85, 85], guidelines=[i_maks, -i_maks])
    
    K = K.reshape(1, 4)
    print('Feedback gain: K =',K,'\nEigenvalues of (A+BK):',np.linalg.eigvals(A + B @ K))
    print('Initial conditions:', [x_1, x_2, x_3])
    print('Maximum absolute displacements (m):', [ np.max(np.abs(state_1[0])), np.max(np.abs(state_2[0])),  np.max(np.abs(state_3[0]))])
    print('Maximum absolute currents (A):', [ np.max(np.abs(u_1[0])), np.max(np.abs(u_2[0])),  np.max(np.abs(u_3[0]))])

if __name__ == "__main__":
    plot_og_print(K_matrix(Q,R))
