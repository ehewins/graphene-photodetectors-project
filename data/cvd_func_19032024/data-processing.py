import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


def find_min_point_indices(x, y):
    gradient = np.abs(np.gradient(y, x))
    grad_argsort = np.argsort(gradient)
    true_minima = np.array([], dtype=int)
    for i in grad_argsort[:5]:
        # Nearby points may have lower gradient than the other minimum point
        if y[i-1] < y[i] or y[i+1] < y[i] or y[i-2] < y[i] or y[i+2] < y[i]:
            continue  # we don't want these points
        else:
            true_minima = np.append(true_minima, i)
    return np.sort(true_minima)


def find_jump_indices(array):
    """
    Find the indices corresponding to the start and end of a jump in the data.
    Used to determine the photocurrent in I_sd vs t measurements.
    """
    arr_diffs = array[1:] - array[:-1]
    biggest_jump = max(np.abs(arr_diffs))
    jumps = np.abs(arr_diffs) > biggest_jump * 0.5
    jump_i = np.argwhere(jumps).flatten()
    jump_f = jump_i + 1
    # Some downward jumps take 2 datapoints - catch the actual finish point
    for j in range(len(jump_f)):
        if abs(array[jump_f[j]+1] - array[jump_i[j]]) > \
           abs(array[jump_f[j]] - array[jump_i[j]]):
            jump_f[j] += 1
    return jump_i, jump_f


"""
Objective 1:
Compare several Isd(Vg) measurements taken at different times, showing that
Device 1 is changes significantly with prior light exposure, and that Device 2
is significantly more stable.
"""

# Table recording file names, the label for the graph, and number of passes
file_table = (
    ("../cvd_func_15032024/CVD1F-Isd-Vg-Dark.dat", "First measurement", 8),
    ("../cvd_func_15032024/CVD1F-Isd-Vg-Light.dat", "First measurement", 8),
    ("../cvd_func_15032024/CVD2F-Isd-Vg-Dark.dat", "First measurement", 8),
    ("../cvd_func_15032024/CVD2F-Isd-Vg-Light.dat", "First measurement", 8),
    ("CVD1F-Isd-Vg-Dark.dat", "Second measurement", 6),
    ("CVD1F-Isd-Vg-Light.dat", "Second measurement", 8),
    ("CVD2F-Isd-Vg-Dark.dat", "Second measurement", 8),
    ("CVD2F-Isd-Vg-Light.dat", "Second measurement", 8),
    ("CVD1F-Isd-Vg-Dark-25max.dat", "Third measurement", 4),
    ("CVD1F-Isd-Vg-Light-25max.dat", "Third measurement", 4),
    ("CVD2F-Isd-Vg-Dark-35max.dat", "Third measurement", 4),
    ("CVD2F-Isd-Vg-Light-35max.dat", "Third measurement", 4),
    ("CVD1F-Isd-Vg-Dark-25max-2.dat", "Fourth measurement", 6),
    ("CVD1F-Isd-Vg-Light-25max-2.dat", "Fourth measurement", 6),
    ("CVD2F-Isd-Vg-Dark-35max-2.dat", "Fourth measurement", 6),
    ("CVD2F-Isd-Vg-Light-35max-2.dat", "Fourth measurement", 8),
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Device 1 - Functionalised with Quantum Dots")
ax1.set_title("Dark conditions")
ax2.set_title("Light conditions")
for ax in (ax1, ax2):
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
print("For Device 1 in dark conditions:")
for data in file_table[0::4]:
    Vg, Isd = np.loadtxt(data[0], delimiter='\t', usecols=(0, 3), unpack=True)
    # Use only the final forward & backward pass
    npass = data[2]
    Vg, Isd = Vg[(npass-2)*len(Vg)//npass:], Isd[(npass-2)*len(Isd)//npass:]
    ax1.plot(Vg, Isd, label=data[1])
    # Calculate the minimum points, i.e. the Dirac points for fwd & bwd passes
    if data[1] == "First measurement":  # but don't bother for measurement 1
        continue
    min_indices = find_min_point_indices(Vg, Isd)
    ax1.plot(Vg[min_indices], Isd[min_indices], 'kx')  # visual confirmation
    for i, direction in zip(min_indices, ("forward", "backward")):
        Vg_res = np.mean(np.abs(Vg[i:i+2] - Vg[i-1:i+1]))
        print(f"{data[1]}'s {direction} pass: V_Dirac = {Vg[i]:.1f} ± {Vg_res:.1f} V")
print("For Device 1 in illuminated conditions:")
for data in file_table[1::4]:
    Vg, Isd = np.loadtxt(data[0], delimiter='\t', usecols=(0, 3), unpack=True)
    # Use only the final forward & backward pass
    npass = data[2]
    Vg, Isd = Vg[(npass-2)*len(Vg)//npass:], Isd[(npass-2)*len(Isd)//npass:]
    ax2.plot(Vg, Isd, label=data[1])
    # Calculate the minimum points, i.e. the Dirac points for fwd & bwd passes
    if data[1] == "First measurement":  # but don't bother for measurement 1
        continue
    min_indices = find_min_point_indices(Vg, Isd)
    ax2.plot(Vg[min_indices], Isd[min_indices], 'kx')  # visual confirmation
    for i, direction in zip(min_indices, ("forward", "backward")):
        Vg_res = np.mean(np.abs(Vg[i:i+2] - Vg[i-1:i+1]))
        print(f"{data[1]}'s {direction} pass: V_Dirac = {Vg[i]:.1f} ± {Vg_res:.1f} V")
ax1.legend()
ax2.legend()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Device 2 - Functionalised with Perovskites")
ax1.set_title("Dark conditions")
ax2.set_title("Light conditions")
for ax in (ax1, ax2):
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
print("For Device 2 in dark conditions:")
for data in file_table[2::4]:
    Vg, Isd = np.loadtxt(data[0], delimiter='\t', usecols=(0, 3), unpack=True)
    # Use only the final forward & backward pass
    npass = data[2]
    Vg, Isd = Vg[(npass-2)*len(Vg)//npass:], Isd[(npass-2)*len(Isd)//npass:]
    ax1.plot(Vg, Isd, label=data[1])
    # Calculate the minimum points, i.e. the Dirac points for fwd & bwd passes
    # But don't bother for the first and second measurement
    if data[1] in ("First measurement", "Second measurement"):
        continue
    min_indices = find_min_point_indices(Vg, Isd)
    ax1.plot(Vg[min_indices], Isd[min_indices], 'kx')  # visual confirmation
    for i, direction in zip(min_indices, ("forward", "backward")):
        Vg_res = np.mean(np.abs(Vg[i:i+2] - Vg[i-1:i+1]))
        print(f"{data[1]}'s {direction} pass: V_Dirac = {Vg[i]:.1f} ± {Vg_res:.1f} V")
print("For Device 2 in illuminated conditions:")
for data in file_table[3::4]:
    Vg, Isd = np.loadtxt(data[0], delimiter='\t', usecols=(0, 3), unpack=True)
    # Use only the final forward & backward pass
    npass = data[2]
    Vg, Isd = Vg[(npass-2)*len(Vg)//npass:], Isd[(npass-2)*len(Isd)//npass:]
    ax2.plot(Vg, Isd, label=data[1])
    # Calculate the minimum points, i.e. the Dirac points for fwd & bwd passes
    # But don't bother for the first and second measurement
    if data[1] in ("First measurement", "Second measurement"):
        continue
    min_indices = find_min_point_indices(Vg, Isd)
    ax2.plot(Vg[min_indices], Isd[min_indices], 'kx')  # visual confirmation
    for i, direction in zip(min_indices, ("forward", "backward")):
        Vg_res = np.mean(np.abs(Vg[i:i+2] - Vg[i-1:i+1]))
        print(f"{data[1]}'s {direction} pass: V_Dirac = {Vg[i]:.1f} ± {Vg_res:.1f} V")
ax1.legend()
ax2.legend()

"""
Objective 2:
Compare the Isd(Vsd) measurements from 15032024 and 19032024 for both devices,
in light and in dark, to check for changes in device resistance.
"""

file_table = (
    ("../cvd_func_15032024/CVD1F-Isd-Vsd-Dark.dat", "First measurement"),
    ("../cvd_func_15032024/CVD1F-Isd-Vsd-Light.dat", "First measurement"),
    ("../cvd_func_15032024/CVD2F-Isd-Vsd-Dark.dat", "First measurement"),
    ("../cvd_func_15032024/CVD2F-Isd-Vsd-Light.dat", "First measurement"),
    ("CVD1F-Isd-Vsd-Dark.dat", "Second measurement"),
    ("CVD1F-Isd-Vsd-Light.dat", "Second measurement"),
    ("CVD2F-Isd-Vsd-Dark.dat", "Second measurement"),
    ("CVD2F-Isd-Vsd-Light.dat", "Second measurement"),
)

print()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Device 1 - Functionalised with Quantum Dots")
ax1.set_title("Dark conditions")
ax2.set_title("Light conditions")
print("For Device 1 in dark conditions:")
for data in file_table[0::4]:
    Vsd, Isd = np.loadtxt(data[0], delimiter='\t', unpack=True)
    ax1.plot(Vsd, Isd, label=data[1])
    (m, c), cv = curve_fit(linear_fit, Vsd, Isd)
    R, err = 1/m, 1/m * np.sqrt(cv[0, 0]) / m
    print(f"{data[1]}'s resistance: R_sd = {R:e} ± {err:e} Ohms")
print("For Device 1 in illuminated conditions:")
for data in file_table[1::4]:
    Vsd, Isd = np.loadtxt(data[0], delimiter='\t', unpack=True)
    ax2.plot(Vsd, Isd, label=data[1])
    (m, c), cv = curve_fit(linear_fit, Vsd, Isd)
    R, err = 1/m, 1/m * np.sqrt(cv[0, 0]) / m
    print(f"{data[1]}'s resistance: R_sd = {R:e} ± {err:e} Ohms")
ax1.legend()
ax2.legend()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Device 2 - Functionalised with Perovskites")
ax1.set_title("Dark conditions")
ax2.set_title("Light conditions")
print("For Device 2 in dark conditions:")
for data in file_table[2::4]:
    Vsd, Isd = np.loadtxt(data[0], delimiter='\t', unpack=True)
    ax1.plot(Vsd, Isd, label=data[1])
    (m, c), cv = curve_fit(linear_fit, Vsd, Isd)
    R, err = 1/m, 1/m * np.sqrt(cv[0, 0]) / m
    print(f"{data[1]}'s resistance: R_sd = {R:e} ± {err:e} Ohms")
print("For Device 2 in illuminated conditions:")
for data in file_table[3::4]:
    Vsd, Isd = np.loadtxt(data[0], delimiter='\t', unpack=True)
    ax2.plot(Vsd, Isd, label=data[1])
    (m, c), cv = curve_fit(linear_fit, Vsd, Isd)
    R, err = 1/m, 1/m * np.sqrt(cv[0, 0]) / m
    print(f"{data[1]}'s resistance: R_sd = {R:e} ± {err:e} Ohms")
ax1.legend()
ax2.legend()

"""
Objective 3:
Compare the Isd(t) measurements (at Vg = 10 mV) from 15032024 and 19032024 for
both devices, to check for changes in photocurrent (hence photoresponsivity).
"""

file_table = (
    ("../cvd_func_15032024/CVD1F-time.dat", "First measurment"),
    ("../cvd_func_15032024/CVD2F-time.dat", "First measurment"),
    ("CVD1F-time-10mV.dat", "Second measurment"),
    ("CVD2F-time-10mV.dat", "Second measurment"),
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("With constant $V_{sd} = 10$ mV and $V_g = 0$ V")
ax1.set_title("Device 1 - Quantum Dots")
ax2.set_title("Device 2 - Perovskites")
for data in file_table[0::2]:
    time, current = np.loadtxt(data[0], usecols=(1, 3), unpack=True)
    ax1.plot(time, current, label=data[1])
for data in file_table[1::2]:
    time, current = np.loadtxt(data[0], usecols=(1, 3), unpack=True)
    ax2.plot(time, current, label=data[1])
for ax in (ax1, ax2):
    ax.set_xlabel("Time, $t$, (s)")
    ax.set_ylabel("Source-drain current, $I_{sd}$, (A)")
ax1.legend()
ax2.legend()

# It looks like we've lost the slow times and their exponential curve shapes,
# so we'll just do a manual photocurrent calculation for measurement set 2.

device1_I_values = np.array([2.635, 2.721, 2.687, 2.614, 2.599, 2.691, 2.678,
                             2.592, 2.585, 2.675, 2.619, 2.536, 2.534, 2.614,
                             2.576, 2.492])*1e-6
device2_I_values = np.array([4.039, 4.108, 3.912, 3.840, 3.871, 3.947, 3.832,
                             3.753, 3.786, 3.862, 3.798, 3.718, 3.754, 3.832,
                             3.787, 3.710])*1e-6
magnitude_I_jumps1 = np.abs(device1_I_values[1::2] - device1_I_values[0::2])
device1_photocurrent = np.mean(magnitude_I_jumps1)
device1_photocurrent_err = np.std(magnitude_I_jumps1)
magnitude_I_jumps2 = np.abs(device2_I_values[1::2] - device2_I_values[0::2])
device2_photocurrent = np.mean(magnitude_I_jumps2)
device2_photocurrent_err = np.std(magnitude_I_jumps2)
print()
print("For Device 1, functionalised with quantum dots:")
print(f"The average photocurrent is {device1_photocurrent:e} ± {device1_photocurrent_err:e} A")
print("For Device 2, functionalised with perovskites:")
print(f"The average photocurrent is {device2_photocurrent:e} ± {device2_photocurrent_err:e} A")

"""
Objective 4:
Determine the photoresponsivity as a function of power from the Isd = 50 mV
photocurrent measurements with attenuating filters between the laser and sample
"""

# table of file names, graph labels and the filter's actual optical density
file_table = (
    ("CVD1F-time-50mV.dat", "OD0", 0),
    ("CVD1F-time-50mV-OD1.dat", "OD1", 1.1),
    ("CVD1F-time-50mV-OD2.dat", "OD2", 2.7),
    ("CVD1F-time-50mV-OD3.dat", "OD3", 3.6),
    ("CVD2F-time-50mV.dat", "OD0", 0),
    ("CVD2F-time-50mV-OD1.dat", "OD1", 1.1),
    ("CVD2F-time-50mV-OD2.dat", "OD2", 2.7),
    ("CVD2F-time-50mV-OD3.dat", "OD3", 3.6),
)

# Figure setup
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("With constant $V_{sd} = 50$ mV and $V_g = 0$ V")
fig2.suptitle("With constant $V_{sd} = 50$ mV and $V_g = 0$ V")
for ax in (ax1, ax3):
    ax.set_title("Device 1 - Quantum Dots")
for ax in (ax2, ax4):
    ax.set_title("Device 2 - Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Time, $t$, (s)")
    ax.set_ylabel("Source-drain current, $I_{sd}$, (A)")
for ax in (ax3, ax4):
    ax.set_xlabel("Power on device, $P$ (W)")
    ax.set_ylabel("Photoresponsivity, $R$ (A/W)")

P0 = 2.33e-6  # Power incident on device, without optical filters
dP0 = 0.07e-6  # The corresponding uncertainty
print()
print("For Device 1, functionalised with quantum dots:")
responsivity = []
power = []
for data in file_table[:4]:
    time, current = np.loadtxt(data[0], usecols=(1, 3), unpack=True)
    ax1.plot(time, current, label=data[1])
    jump_ind1, jump_ind2 = find_jump_indices(current)
    # ax1.plot(time[jump_ind1], current[jump_ind1], 'kx')  # visual check
    # ax1.plot(time[jump_ind2], current[jump_ind2], 'kx')
    photocurrent_values = np.abs(current[jump_ind2] - current[jump_ind1])
    Iph = np.mean(photocurrent_values)
    dIph = np.std(photocurrent_values)
    rel_dIph = dIph / Iph
    P = P0 * 10**(-data[2])  # Power after filtering
    Rph = Iph / P  # Photoresponsivity,
    dRph = Rph * np.sqrt((dIph/Iph)**2 + (dP0/P0)**2)  # and its uncertainty
    print(f"With {data[1]}, photoresponsivity R = {Rph:e} ± {dRph:e} A/W")
    responsivity.append(Rph)
    power.append(P)
ax1.legend(loc='lower left')
# Create log plot of responsivity versus power, and fit a straight line
(m, c), cv = curve_fit(linear_fit, np.log10(power), np.log10(responsivity))
print(f"Gradient of log(R_ph) vs log(P) plot: {m:e} ± {np.sqrt(cv[0, 0]):e}")
ax3.loglog(power, responsivity, 'bo', label='Data')
ax3.loglog(power, 10**c * np.array(power)**m, 'k-', label='Linear fit')
ax3.legend()

print("For Device 2, functionalised with perovskites:")
responsivity = []
power = []
for data in file_table[4:]:
    time, current = np.loadtxt(data[0], usecols=(1, 3), unpack=True)
    ax2.plot(time, current, label=data[1])
    jump_ind1, jump_ind2 = find_jump_indices(current)
    # ax2.plot(time[jump_ind1], current[jump_ind1], 'kx')  # visual check
    # ax2.plot(time[jump_ind2], current[jump_ind2], 'kx')
    photocurrent_values = np.abs(current[jump_ind2] - current[jump_ind1])
    Iph = np.mean(photocurrent_values)
    dIph = np.std(photocurrent_values)
    rel_dIph = dIph / Iph
    P = P0 * 10**(-data[2])  # Power after filtering
    Rph = Iph / P  # Photoresponsivity,
    dRph = Rph * np.sqrt((dIph/Iph)**2 + (dP0/P0)**2)  # and its uncertainty
    print(f"With {data[1]}, photoresponsivity R = {Rph:e} ± {dRph:e} A/W")
    responsivity.append(Rph)
    power.append(P)
ax2.legend(loc='lower left')
# Create log plot of responsivity versus power, and fit a straight line
(m, c), cv = curve_fit(linear_fit, np.log10(power), np.log10(responsivity))
print(f"Gradient of log(R_ph) vs log(P) plot: {m:e} ± {np.sqrt(cv[0, 0]):e}")
ax4.loglog(power, responsivity, 'bo', label='Data')
ax4.loglog(power, 10**c * np.array(power)**m, 'k-', label='Linear fit')
ax4.legend()

plt.show()
