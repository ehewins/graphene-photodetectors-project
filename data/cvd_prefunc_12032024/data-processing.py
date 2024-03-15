import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


"""
Objective 1: Graphing results, comparing the two graphene FETs
"""

# Graphing the source-drain current versus source-drain voltage measurements
# for CVD device 1 and 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Source-drain measurements with zero gate voltage")
axes = (ax1, ax2)
files = ("CVD1-Isd-Vsd.dat", "CVD2-Isd-Vsd.dat")
for f, ax in zip(files, axes):
    Vsd_2t, Isd, Vsd_4t = np.loadtxt(f, delimiter='\t', unpack=True)
    ax.plot(Vsd_2t, Isd, label="2-terminal voltage")
    ax.plot(Vsd_4t, Isd, label="4-terminal voltage")
    ax.set_xlabel("Source-drain voltage, $V_{sd}$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    ax.legend()
ax1.set_title("CVD device 1")
ax2.set_title("CVD device 2")
fig.tight_layout()

# Graphing the source-drain current versus gate voltage measurements for CVD
# device 1 and 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Gate voltage measurements with $V_{sd} = 10$ mV")
axes = (ax1, ax2)
files = ("CVD1-Isd-Vg.dat", "CVD2-Isd-Vg.dat")
for f, ax in zip(files, axes):
    Vg, Ig, Vsd_2t, Isd, Vsd_4t = np.loadtxt(f, delimiter='\t', unpack=True)
    ax.plot(Vg, Isd)
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
ax1.set_title("CVD device 1")
ax2.set_title("CVD device 2")
fig.tight_layout()

# Graphing the source-drain current versus time for alternating dark and light conditions

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Source-drain current measurements with $V_{sd} = 10$ mV")
axes = (ax1, ax2)
files = ("CVD1-time.dat", "CVD2-time.dat")
for f, ax in zip(files, axes):
    time, Isd = np.loadtxt(f, delimiter='\t', usecols=(1, 3), unpack=True)
    ax.plot(time, Isd, '.-')
    ax.set_xlabel("Time, $t$ (s)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    print("Average time between I_{sd}(t) datapoints:",
          str(np.mean(time[1::2] - time[0:-1:2])), "seconds.")
ax1.set_title("CVD device 1")
ax2.set_title("CVD device 2")
fig.tight_layout()

"""
Objective 2: find the source-drain resistance of the two CVD devices.
For device 1, use both the 2-terminal measurements and 4-terminal measurements.
For device 2, use only the 2-terminal measurements.
"""

Vsd1_2t, Isd1, Vsd1_4t = np.loadtxt("CVD1-Isd-Vsd.dat", delimiter='\t',
                                    unpack=True)
Vsd2_2t, Isd2 = np.loadtxt("CVD2-Isd-Vsd.dat",delimiter='\t', usecols=(0, 1),
                           unpack=True )
(m1, c1), cv1 = curve_fit(linear_fit, Vsd1_2t, Isd1)
(m2, c2), cv2 = curve_fit(linear_fit, Vsd1_4t, Isd1)
(m3, c3), cv3 = curve_fit(linear_fit, Vsd2_2t, Isd2)
R1, err1 = 1/m1, 1/m1 * np.sqrt(cv1[0, 0]) / m1
R2, err2 = 1/m2, 1/m2 * np.sqrt(cv2[0, 0]) / m2
R3, err3 = 1/m3, 1/m3 * np.sqrt(cv3[0, 0]) / m3
print("Calculations of the source-drain resistance:")
print(f"For device 1, using the two terminal measurements: R = {R1:e} ± {err1:e} Ohms")
print(f"For device 1, using the four terminal measurements: R = {R2:e} ± {err2:e} Ohms")
print(f"For device 2, using the two terminal measurements: R = {R3:e} ± {err3:e} Ohms")

"""
Objective 3: Calculate the field-effect mobility and carrier concentration in
the samples using the source-drain current vs gate voltage measurements
"""

Vg1, Isd1 = np.loadtxt("CVD1-Isd-Vg.dat", delimiter='\t', usecols=(0, 3),
                       unpack=True)
Vg2, Isd2 = np.loadtxt("CVD2-Isd-Vg.dat", delimiter='\t', usecols=(0, 3),
                       unpack=True)
# Use values from V_g >= -5 V
linear_region = Vg1 >= -3
(m1, c1), cv1 = curve_fit(linear_fit, Vg1[linear_region], Isd1[linear_region])
(m2, c2), cv2 = curve_fit(linear_fit, Vg2[linear_region], Isd2[linear_region])
errs1 = np.sqrt(np.diag(cv1))
errs2 = np.sqrt(np.diag(cv2))
rel_dm1 = errs1[0] / m1
rel_dc1 = errs1[1] / c1
rel_dm2 = errs2[0] / m2
rel_dc2 = errs2[1] / c2

# THESE ARE WRONG -> CHANGE
L = 50e-6  # channel length (m)
dL = 1e-6  # uncertainty in channel length (m)
rel_dL = dL / L
W = 50e-6  # channel width (m)
dW = 1e-6  # uncertainty in channel width (m)
rel_dW = dW / W
d = 90e-9  # dielectric thickness (m)
epsilon_r = 3.9  # relative permittivity of dielectric
epsilon_0 = 8.85e-12  # permittivity of free space (F/m)
V_ds = 10e-3  # source-drain voltage (V)
e = 1.6e-19  # charge magnitude of a single carrier (C)

mu1 = m1 * L * d / (W * V_ds * epsilon_0 * epsilon_r)
mu2 = m2 * L * d / (W * V_ds * epsilon_0 * epsilon_r)
rel_dmu1 = np.sqrt(rel_dm1**2 + rel_dL**2 + rel_dW**2)
rel_dmu2 = np.sqrt(rel_dm2**2 + rel_dL**2 + rel_dW**2)
dmu1 = rel_dmu1 * mu1
dmu2 = rel_dmu2 * mu2
p1 = - c1 / m1 * (epsilon_0 * epsilon_r) / (e * d)
p2 = - c2 / m2 * (epsilon_0 * epsilon_r) / (e * d)
rel_dp1 = np.sqrt(rel_dm1**2 + rel_dc1**2)
rel_dp2 = np.sqrt(rel_dm2**2 + rel_dc2**2)
dp1 = rel_dp1 * p1
dp2 = rel_dp2 * p2
print()
print("Calculations of the field effect mobility and carrier concentrations:")
print(f"Device 1: μ = {mu1*1e4:e} ± {dmu1*1e4:e} cm^2 V^-1 s^-1, p₀ = {p1*1e-4:e} ± {dp1*1e-4:e} cm^-2")
print(f"Device 2: μ = {mu2*1e4:e} ± {dmu2*1e4:e} cm^2 V^-1 s^-1, p₀ = {p2*1e-4:e} ± {dp2*1e-4:e} cm^-2")

"""
Objective 4: Calculate the photocurrent from the Isd versus time measurements
"""

# Current measurements were taken from the graph manually
device1_I_values = np.array([1.2446, 1.2506, 1.2417, 1.2343, 1.2342, 1.2406,
                             1.2288, 1.2224, 1.2215, 1.2272, 1.2021, 1.1951,
                             1.1948, 1.2010, 1.1827, 1.1755])*1e-5
device2_I_values = np.array([7.6642, 7.721, 7.651, 7.592, 7.594, 7.652, 7.593,
                             7.532, 7.539, 7.593, 7.544, 7.482, 7.488, 7.543,
                             7.486, 7.429])*1e-6
I_err = 1e-9
magnitude_I_jumps1 = np.abs(device1_I_values[1::2] - device1_I_values[0::2])
magnitude_I_jumps2 = np.abs(device2_I_values[1::2] - device2_I_values[0::2])
I_jumps_err = np.sqrt(2) * I_err
device1_photocurrent = np.mean(magnitude_I_jumps1)
device2_photocurrent = np.mean(magnitude_I_jumps2)
device1_photocurrent_err = np.std(magnitude_I_jumps1)
device2_photocurrent_err = np.std(magnitude_I_jumps2)
print()
print("Calculations of the photocurrent for device 1 and 2:")
print(f"Device 1: {device1_photocurrent:e} ± {device1_photocurrent_err:e} A")
print(f"Device 2: {device2_photocurrent:e} ± {device2_photocurrent_err:e} A")

plt.show()
