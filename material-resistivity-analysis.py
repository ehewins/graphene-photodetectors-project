import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


"""
Objective 1:
Create graphs demonstrating the change in OFET resistance as a result of
functionalisation, and as a result of illumination by the laser.
Resistance calculations are done in the data-processing.py files found in
the OFET data directories.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
axes = (ax1, ax2)
devices = ("OFET3", "OFET4")
directories = ("data/ofet3_16022024/", "data/ofet4_23022024/")
conditions = ("Bare", "Dark", "Light")
graph_labels = ("Before functionalisation", "Dark conditions", "Light conditions")
for device, ax, directory in zip(devices, axes, directories):
    for condition, label in zip(conditions, graph_labels):
        filename = directory + device
        if condition == "Bare":
            filename += "-Isd-Vsd.dat"
        else:
            filename += f"F-Isd-Vsd-{condition}.dat"
        Vsd, Isd = np.loadtxt(filename, usecols=(0, 1), unpack=True)
        # Let's not plot data which looks overly messy.
        if filename != "data/ofet3_16022024/OFET3-Isd-Vsd.dat":
            Vsd = Vsd[len(Vsd)//2:]
            Isd = Isd[len(Isd)//2:]
        ax.plot(Vsd, Isd*1e9, label=label)
    ax.set_xlabel("Source-drain voltage, $V_{sd}$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (nA)")
    ax.legend()
ax1.set_title("OFET Functionalised with Perovskites")
ax2.set_title("OFET Functionalised with Quantum Dots")
fig.tight_layout()

"""
Objective 2:
Create graphs demonstrating the change in GFET resistance as a result of
functionalisation, and as a result of illumination by the laser.
Also calculate the resistivity in each case.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
axes = (ax1, ax2)
devices = ("CVD1", "CVD2")
conditions = ("Pristine", "Dark", "Light")
graph_labels = ("Pristine graphene", "Functionalised (Dark)", "Functionalised (Light)")
for device, ax in zip(devices, axes):
    for condition, label in zip(conditions, graph_labels):
        if condition == "Pristine":
            filename = f"data/cvd_prefunc_12032024/{device}-Isd-Vsd.dat"
        else:
            filename = f"data/cvd_func_15032024/{device}F-Isd-Vsd-{condition}.dat"
        Vsd, Isd = np.loadtxt(filename, usecols=(0, 1), unpack=True)
        ax.plot(Vsd*1e3, Isd*1e6, label=label)
        (m, c), cv = curve_fit(linear_fit, Vsd, Isd)
        R, R_err = 1/m, 1/m * np.sqrt(cv[0, 0]) / m  # Resistance and uncert.
        # Graphene channel is 50x50 µm square, so resistance = resistivity.
        rho = R
        # Need to propagate the uncertainty in this squreness however:
        rho_err = rho * np.sqrt((R_err/R)**2 + 2*(1/50)**2)
        print(f"Resistivity of {device} in {condition} conditions is {rho:.2e} ± {rho_err:.2e} Ω/sq")
    ax.set_xlabel("Source-drain voltage, $V_{sd}$ (mV)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ ($\\mu$A)")
    ax.legend()
ax1.set_title("Device 1, Functionalised with Quantum Dots")
ax2.set_title("Device 2, Functionalised with Perovskites")
fig.tight_layout()

plt.show()
