# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:19:49 2024

@author: Jack Bourne
"""

import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

epsilon = 3.9
epsilon_0 = 8.85e-12
Vg = np.linspace(-1, 1, 1000)
e = 1.6e-19
d = 300e-9
mu = 1e4

n = (epsilon * epsilon_0 * Vg) / (e * d)
sigma = e * n * mu

sigma1 = np.abs(sigma)
print(len(sigma1)==len(Vg))

# Define the window size for moving average
window_size = 100 # Adjust the window size as needed

# Compute moving average
sigma_ma = moving_average(sigma1, window_size)

# Plot original and moving average data
plt.figure(figsize=(10, 5))
#plt.plot(Vg, sigma1, label='Original Data')
plt.plot(Vg, sigma_ma, label=f'Moving Average (window size = {window_size})')
plt.xlabel('Gate Voltage')
plt.ylabel('Conductance')
plt.legend()
plt.grid(True)
plt.show()

