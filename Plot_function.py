#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:16:02 2024

@author: laurajorgensen
"""

# Import package
import matplotlib.pyplot as plt

# Plotting function
def create_plot_sim(x_label, y_label, x_data_list, y_data_list, labels=None, y_limits=None, guidelines=None, convert_to_pi=False, filename=None):
    plt.figure(figsize=(8, 4))
    
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
    plt.xlim(0, x_data[-1])
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}.pdf", format="pdf", dpi=300)
    plt.show()
    
    
def create_plot_exp(x1, y1, x_label, y_label, title=None, x2=None, y2=None, legend_label1=None, guidelines=None, legend_label2=None, y_limits=None, x_limits=None, filename=None):
    """Helper function to create and display a plot."""
    plt.figure(figsize=(8, 4))
    plt.plot(x1, y1, label=legend_label1)  # Målte data
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2, label=legend_label2, linestyle='--')  # Simulering
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    if title:
        plt.title(title)
    if guidelines:
        for line in guidelines:
            plt.axhline(line, color='black', linewidth=0.8, linestyle='--')
    if y_limits:
        plt.ylim(y_limits)
    if legend_label1 or legend_label2:  # Tilføj legend, hvis labels er tilgængelige
        plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if x_limits:
        plt.xlim(x_limits)
    if filename:
        plt.savefig(f"{filename}.pdf", format="pdf")
    plt.show()