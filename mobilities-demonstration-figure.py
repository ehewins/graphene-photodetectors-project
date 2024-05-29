"""
This file generates a figure used in the theory section of the report, illustrating two methods for finding the field-effect mobility, those being maximum differential mobility and the 3-point mobility.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


def top_hat(width):
    return np.ones(width) / width


dVg = 0.2  # Gate voltage step resolution (V)
Vsd = 10e-3  # 10 mV source-drain voltage
d = 90e-9  # dielectric thickness (m)
epsilon_r = 3.9  # relative permittivity of dielectric
epsilon_0 = 8.85e-12  # permittivity of free space (F/m)
e = 1.6e-19  # charge magnitude of a single carrier (C)

width = 25  # width of top hat function for smoothing out the gradient function

filename = "/home/ellis/documents/university/year_4/masters_project/graphene-photodetectors-project/data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max.dat"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for ax in (ax1, ax2):
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.ticklabel_format(scilimits=[-3, 6], useMathText=True)
ax1.set_ylabel("Conductivity, $\\sigma$ ($\\Omega^{-1}$□)")
ax2.set_ylabel("Resistivity, $\\rho$ ($\\Omega$□$^{-1}$)")
Vg_data, Isd_data = np.loadtxt(filename, usecols=(0, 3), unpack=True)
datapoints_per_pass = int((max(Vg_data) - min(Vg_data)) / dVg)
num_passes = int(len(Vg_data) / datapoints_per_pass)
# Extract data from the final forward Vg sweep
start_index = (num_passes - 2) * datapoints_per_pass
stop_index = (num_passes - 1) * datapoints_per_pass
Vg = Vg_data[start_index: stop_index]
Isd = Isd_data[start_index: stop_index]
sigma, rho = Isd / Vsd, Vsd / Isd
ax1.plot(Vg, sigma, label='Data')
ax2.plot(Vg, rho, label='Data')
V_dirac = Vg[np.argmax(rho)]
ax1.plot([V_dirac], [min(sigma)], 'x', color='C3')
ax1.annotate("$V_{Dirac}$", fontsize=14, xy=(V_dirac, min(sigma)), xytext=(-1.2, 1), textcoords='offset fontsize')
rho_max = np.max(rho)
ax2.plot([V_dirac], [rho_max], 'X', color='C3')
halfmax_indices = np.argsort(np.abs(rho - rho_max/2))[:2]
# ax1.plot(Vg[halfmax_indices], sigma[halfmax_indices], 'X', color='C3')
ax2.plot(Vg[halfmax_indices], rho[halfmax_indices], 'X', color='C3', label='$\\rho_{max}$ & FWHM')
ax2.annotate("", xy=(Vg[halfmax_indices[0]], rho[halfmax_indices[0]]),
             xytext=(Vg[halfmax_indices[1]], rho[halfmax_indices[1]]),
             arrowprops=dict(arrowstyle='<->', mutation_scale=20))
ax2.annotate("$\\rho_{max}$", fontsize=14, xy=(V_dirac, rho_max), xytext=(-1, 0.5), textcoords='offset fontsize')
ax2.annotate("$\\delta n$", fontsize=14, xy=(V_dirac, rho_max/2), xytext=(0, 0.5), textcoords='offset fontsize')

gradient = np.gradient(sigma, Vg)
gradient_pad = np.pad(gradient, (width-1)//2, mode='edge')
gradient_smooth = np.convolve(gradient_pad, top_hat(width), mode='valid')
gradient2 = np.gradient(np.abs(gradient_smooth), Vg)
max_grad_index_1 = np.nonzero((Vg < V_dirac) & (gradient2 > 0))[0][-1]
max_grad_index_2 = np.nonzero((Vg > V_dirac) & (gradient2 > 0))[0][-1]
near_gradients_1 = gradient[max_grad_index_1-5: max_grad_index_1+6]
near_gradients_2 = gradient[max_grad_index_2-5:
                            min(max_grad_index_2+6, len(gradient))]
# Parameters for graphical plot of chosen point and their uncertainties
m1, a1, b1 = np.mean(near_gradients_1), Vg[max_grad_index_1], sigma[max_grad_index_1]
m2, a2, b2 = np.mean(near_gradients_2), Vg[max_grad_index_2], sigma[max_grad_index_2]
c1 = b1 - m1 * a1
c2 = b2 - m2 * a2
ax1.plot([a1-2, a1+2], m1*np.array([a1-2, a1+2])+c1, '--', color='C3', linewidth=2)
ax1.plot([a2-2, a2+2], m2*np.array([a2-2, a2+2])+c2, '--', color='C3', linewidth=2)
ax1.plot([a1, a2], [b1, b2], 'o', color='C3', label='$\\mu_{max}$')
ax1.annotate("max$\\left(\\frac{d\\sigma}{dV_g}\\right)$", fontsize=14, xy=(a1, b1), xytext=(0.4, 0.4), textcoords='offset fontsize')
fig.tight_layout()

# ax1.annotate("$\\mu_{FE} = \\frac{d}{\\epsilon\\epsilon_0} \\frac{d\\sigma}{dV_g}$", fontsize=24, xy=(0, 3.5e-4))
# ax2.annotate("$\\mu_{3p} = \\frac{4}{e\\delta{}n\\rho_{max}}$", fontsize=24, xy=(10, 5000))
# fig.savefig("/home/ellis/measures-of-mu-from-graphs.png", dpi=400)

# ax1.legend(loc='upper right')
# ax2.legend(loc='upper left')
plt.show()
