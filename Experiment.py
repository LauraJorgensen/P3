# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:55:33 2024

@author: h4sor
"""

# Import packages 
import numpy as np
import matplotlib.pyplot as plt
import os
from Simulation import simulate_system, K_matrix, d_maks, u_maks       
from Plot_function import create_plot_exp    


# Define the path to the folder
folder_path = "Data"

# Get a list of all items in the folder
filer = os.listdir(folder_path)

# Define sampletime
sampletime = 0.00667

# Extract Q and R values from filename
def extract_q_r(filename):
    try:
        parts = filename.split('Q_')
        q_value = float(parts[0]) 
        r_value = float(parts[1].split('R')[0])  
        return q_value, r_value
    except (IndexError, ValueError):
        return None, None

# Read data from each files and print results
for i in filer:
    print(i)
    file_path = os.path.join(folder_path, i)

    q_value, r_value = extract_q_r(i)

    try:
        with open(file_path, "r") as file:
            data = file.readlines()
    
        # Filter by removing whitespaces and process valid rows
        array = []
        for row in data:
            row = row.strip()  
            if len(row) > 1:
                try:
                    values = list(map(float, row.split(',')))
                    if len(values) == 5:
                        array.append(values)
                    else:
                        continue
                except ValueError as e:
                    print(f"Error processing line: {row}")
                    print(f"Reason: {e}")
        array = np.array(array)
    except Exception as e:
        print(f"Error processing file {i}: {e}")
        continue

    # Transpose the array and separate the columns into vectors
    C_position, P_position, C_velocity, P_velocity, u = array.T
    
    # Adjust C_ and P_position
    if P_position[-1] > 0.1 or P_position[-1] < -0.1:
        adjustment = 0.385 - C_position[-1]  
        C_position = C_position + adjustment  
    else:
        C_position = C_position - C_position[-1]
        P_position = P_position - P_position[-1]
        u = u - u[-1]

    # Calculate Q and R
    Q = np.diag([q_value / d_maks**2, 0, 0, 0])
    R = np.diag([r_value / u_maks**2])

    # Create a time vector
    n = np.arange(0, len(u)) * sampletime
    K=K_matrix(Q, R)
      
    # Simulate the system using same initial conditions as in the experiment
    time, state = simulate_system([C_position[0],P_position[0],C_velocity[0],P_velocity[0]], n, K)

    # Plot slide with the file name
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, f"Results of experiment using Q·{q_value} and R·{r_value}", fontsize=24, ha='center', va='center')    
    plt.axis('off')
    plt.show()
    
    # Plot results     
    create_plot_exp(n, C_position, 'Time [s]', 'Displacement [m]',x2=n, y2=state[0], legend_label1="Measured Data", legend_label2="Simulation",x_limits=[0, n[-1]], guidelines=[-0.385,0.385])
    create_plot_exp(n, P_position, 'Time [s]', 'Angle [rad]', x2=n, y2=state[1], legend_label1="Measured Data", legend_label2="Simulation",x_limits=[0, n[-1]])


