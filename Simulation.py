#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:29:32 2024

@author: laurajorgensen
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control


# System parameters
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
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (mp * g) / mc, -Fcc / mc, -(Fcp + alpha) / (l * mc)], 
    [0,( g * (mp + mc)) / (l * mc), -Fcc / (l * mc), -(Fcp + alpha) * (mc + mp) / (mc * mp * l**2)]
])

B = np.array([
    [0],
    [0],
    [1 / mc],
    [1 / (mc * l)]
])

# Q and R
Q = np.diag([1 / d_maks**2, 0, 0, 0])
#R = np.diag([1 / i_maks**2])
R = np.diag([1 / u_maks**2])

# Feedback gain
K, _, _ = control.lqr(A, B, Q, R)
K=-K
K = K.flatten()

# Non-linear model
def equations(t, z, K):
    k_1 = 1000   # Approksimationsvariable
    z1, z2, z3, z4 = z
    u = K @ np.array([z1, z2, z3, z4])  
    d = mc + mp * (np.cos(z2))**2
    Ffc = np.tanh(k_1*z3)*Fcc
    Ffp = np.tanh(k_1*z4)*Fcp+alpha*z4
    
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

# Simulate the system
def simulate_system(initial_condition, t, K):
    sol = solve_ivp(
        equations,
        [t[0], t[-1]],
        initial_condition,
        t_eval=t,
        args=(K,)
    )
    return sol.t, sol.y

def calculate_control_input(state, K):
    u = np.dot(K, state)# MINUS ELLER PLUS HER??!??!?! (alle laver minus men jeg forstår ikke hvor det minus kommer fra :(
    return u / (KT / r) #Konversion: gå fra force til current


# Time vector
T = np.linspace(0, 2, 1000)

# Apply initial conditions
time_1, state_1 = simulate_system(x_1, T, K)
time_2, state_2 = simulate_system(x_2, T, K)
time_3, state_3 = simulate_system(x_3, T, K)

u_1 = calculate_control_input(state_1, K)
u_2 = calculate_control_input(state_2, K)  
u_3 = calculate_control_input(state_3, K)

# Function that determines max initial condition
def find_max_initial_condition(step=0.01):
    initial_angle = 0.0
    while True:
        # Create initial condition
        initial_condition = [0, initial_angle, 0, 0]  # Only vary the angle
        time, state = simulate_system(initial_condition, T, K)
        u = calculate_control_input(state, K)
        
        # Check constraints
        max_displacement = np.max(np.abs(state[0]))
        max_current = np.max(np.abs(u))
        
        if max_displacement > d_maks or max_current > i_maks:
            break  # Stop if any constraint is violated
        
        initial_angle += step  # Increase initial angle
    
    return initial_angle - step

max_angle = find_max_initial_condition()

# Plotting function
def create_plot(x_label, y_label, x_data_list, y_data_list, labels=None, y_limits=None, guidelines=None, convert_to_pi=False):
    plt.figure(figsize=(10, 4))
    
    # Iterate through all datasets
    for i, (x_data, y_data) in enumerate(zip(x_data_list, y_data_list)):
        label = labels[i] if labels else None
        plt.plot(x_data, y_data, label=label)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    if y_limits:
        plt.ylim(y_limits)
    if guidelines:
        for line in guidelines:
            plt.axhline(line, color='black', linewidth=0.8, linestyle='--')
    # if convert_to_pi:
    #     # Define nice y-ticks in terms of π
    #     ax = plt.gca()
    #     yticks = np.linspace(y_limits[0], y_limits[1], 5)  # Create 5 evenly spaced ticks
    #     yticks_pi = [
    #         f"π/{int(np.pi / tick)}" if tick != 0 else "0"
    #         for tick in yticks
    #     ]
    #     ax.set_yticks(yticks)
    #     ax.set_yticklabels(yticks_pi)
    plt.xlim([x_data[0], x_data[-1]])
    plt.legend(loc='lower right')    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Create plots 
create_plot('Time [s]', 'Displacement [m]', [time_1, time_2, time_3], [state_1[0], state_2[0], state_3[0]], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-0.42,0.42], guidelines=[d_maks, -d_maks])
create_plot('Time [s]', 'Angle [rad]', [time_1, time_2, time_3],[state_1[1], state_2[1], state_3[1]], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-np.pi/6, np.pi/6], convert_to_pi=True)
create_plot('Time [s]', 'Velocity [m/s]', [time_1, time_2, time_3], [state_1[2], state_2[2], state_3[2]],labels=['$x_1$', '$x_2$', '$x_3$'],  y_limits=[-2.5, 1.5]) #y_limits=[-1.5, 2.5]
create_plot('Time [s]', 'Angular velocity [rad/s]', [time_1, time_2, time_3],[state_1[3], state_2[3], state_3[3]], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-6, 4]) #y_limits=[-9, 3]
create_plot('Time [s]', 'Current [A]', [time_1, time_2, time_3],[u_1, u_2, u_3], labels=['$x_1$', '$x_2$', '$x_3$'], y_limits=[-85, 85], guidelines=[i_maks, -i_maks])

# # Print results
K = K.reshape(1, 4)
print('Feedback gain: K =',K,'\nEigenvalues of (A+BK):',np.linalg.eigvals(A + B @ K))
print('Initial conditions:', [x_1, x_2, x_3])
print('Maximum displacements (m):', [ np.max(state_1[0]), np.max(state_2[0]),  np.max(state_3[0])])
print('Maximum currents (A):', [ np.max(u_1[0]), np.max(u_2[0]),  np.max(u_3[0])])
print(f"Maximum initial angle (m): {max_angle:.4f} radians")