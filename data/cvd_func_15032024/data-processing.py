import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


def on_exp(t, a, tau):
    return a * (1 - np.exp(- t / tau))


def off_exp(t, a, tau):
    return a * (np.exp(- t / tau) - 1)


"""
Objective 1: Graphing results, comparing the two graphene FETs
"""

# Graphing the source-drain current versus source-drain voltage measurements
# for CVD device 1 and 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Source-drain measurements with zero gate voltage")
axes = (ax1, ax2)
devices = ("CVD1", "CVD2")
conditions = ("Pre-functionalisation", "Dark conditions", "Light conditions")
for device, ax in zip(devices, axes):
    for condition in conditions:
        if condition == "Pre-functionalisation":
            filename = "../cvd_prefunc_12032024/" + device + "-Isd-Vsd.dat"
        else:
            filename = device + "F-Isd-Vsd-" + condition.split(" ")[0] + ".dat"
        Vsd, Isd = np.loadtxt(filename, delimiter='\t', usecols=(0, 1),
                              unpack=True)
        ax.plot(Vsd, Isd, label=condition)
    ax.set_xlabel("Source-drain voltage, $V_{sd}$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    ax.legend()
ax1.set_title("Device 1, Functionalised with Quantum Dots")
ax2.set_title("Device 2, Functionalised with Perovskites")
fig.tight_layout()

# Graphing the source-drain current versus gate voltage measurements for CVD
# device 1 and 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Gate voltage measurements with $V_{sd} = 10$ mV")
axes = (ax1, ax2)
devices = ("CVD1", "CVD2")
conditions = ("Pre-functionalisation", "Dark conditions", "Light conditions")
for device, ax in zip(devices, axes):
    for condition in conditions:
        if condition == "Pre-functionalisation":
            filename = "../cvd_prefunc_12032024/" + device + "-Isd-Vg.dat"
        else:
            filename = device + "F-Isd-Vg-" + condition.split(" ")[0] + ".dat"
        Vg, Isd = np.loadtxt(filename, delimiter='\t', usecols=(0, 3),
                             unpack=True)
        # Only plot from the last forward & backward pass
        Vg, Isd = Vg[6*len(Vg)//8:], Isd[6*len(Isd)//8:]
        ax.plot(Vg, Isd, label=condition)
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    ax.legend()
ax1.set_title("Device 1, Functionalised with Quantum Dots")
ax2.set_title("Device 2, Functionalised with Perovskites")
fig.tight_layout()

# Graphing the source-drain current versus time for alternating dark and light conditions

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Source-drain current measurements with $V_{sd} = 10$ mV")
axes = (ax1, ax2)
files = ("CVD1F-time.dat", "CVD2F-time.dat")
for f, ax in zip(files, axes):
    time, Isd = np.loadtxt(f, delimiter='\t', usecols=(1, 3), unpack=True)
    # ax.plot(time, Isd, '.-')
    ax.plot(Isd, '.-')
    ax.set_xlabel("Time, $t$ (s)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    print("Average time between I_{sd}(t) datapoints:",
          str(np.mean(time[1::2] - time[0:-1:2])), "seconds.")
ax1.set_title("Device 1, Functionalised with Quantum Dots")
ax2.set_title("Device 2, Functionalised with Perovskites")
fig.tight_layout()

"""
Objective 2: find the source-drain resistance of the two functionalised devices,
in light and in dark conditions.
"""

conditions = ("Dark", "Light")
devices = ("CVD1", "CVD2")
for device in devices:
    for condition in conditions:
        Vsd, Isd = np.loadtxt(device+"F-Isd-Vsd-"+condition+".dat",
                              delimiter='\t', unpack=True)
        (m, c), cv = curve_fit(linear_fit, Vsd, Isd)
        R, err = 1/m, 1/m * np.sqrt(cv[0, 0]) / m
        print(f"Resistance of {device} under {condition.lower()} conditions: R = {R:e} ± {err:e} Ohms")

"""
Objective 3: Calculate the field-effect mobility and carrier concentration in
the samples using the source-drain current vs gate voltage measurements
"""

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

# NOTE: I'm not liking the fits here. Need to talk to Oleg about linear region.
print()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.tight_layout()
devices = ("CVD1", "CVD2")
axes = (ax1, ax2)
conditions = ("Dark", "Light")
for device, ax in zip(devices, axes):
    for condition in conditions:
        Vg, Isd = np.loadtxt(device+"F-Isd-Vg-"+condition+".dat",
                             delimiter='\t', usecols=(0, 3), unpack=True)
        # Only use the last forward & backward pass
        Vg, Isd = Vg[6*len(Vg)//8:], Isd[6*len(Isd)//8:]
        gradient = np.gradient(Isd, Vg)
        gradient2 = np.gradient(gradient, Vg)
        approx_linear = np.abs(gradient2) <= 2e-9  # threshold is arbitrary
        # Linear fit
        (m, c), cv = curve_fit(linear_fit, Vg[approx_linear],
                               Isd[approx_linear])
        errs = np.sqrt(np.diag(cv))
        rel_dm = errs[0] / m
        rel_dc = errs[1] / c
        # Calculate mobility and carrier concentration
        mu = m * L * d / (W * V_ds * epsilon_0 * epsilon_r)
        rel_dmu = np.sqrt(rel_dm**2 + rel_dL**2 + rel_dW**2)
        dmu = rel_dmu * mu
        p = - c / m * (epsilon_0 * epsilon_r) / (e * d)
        rel_dp = np.sqrt(rel_dm**2 + rel_dc**2)
        dp = rel_dp * p
        print(f"Mobility of {device} under {condition.lower()} conditions: μ = {mu*1e4:e} ± {dmu*1e4:e} cm^2 V^-1 s^-1")
        print(f"Carrier concentration of {device} under {condition.lower()} conditions: p₀ = {p*1e-4:e} ± {dp*1e-4:e} cm^-2")
        # Show the extrapolation graphically
        ax.plot(Vg, Isd, label=condition+' conditions')
        ax.plot([min(Vg), - c / m], [m*min(Vg)+c, 0], linestyle='dotted', color='black')
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    ax.legend()
ax1.set_title("Device 1, Functionalised with Quantum Dots")
ax2.set_title("Device 2, Functionalised with Perovskites")

"""
Objective 4: Calculate the photocurrent from the Isd versus time measurements for both devices, and the on/off times for CVD1 only.
"""

# CVD1 calculations
time, Isd = np.loadtxt("CVD1F-time.dat", delimiter='\t', usecols=(1, 3),
                       unpack=True)
# This array defines the indices bouding each on/off region
region_indices = [29, 56, 84, 110, 138, 163, 187, 214, 238]
# Some other effect causes a breif current drop immediately before the current
# increase of the 'on' jumps. Remove these features, as they could confuse the
# photocurrent calculation.
for problem_index in region_indices[0:8:2]:
    Isd[problem_index] = Isd[problem_index - 1]
# Another effect causes the current to decrease at a roughly constant rate.
# Correct for this effect by subtracting the linearly decreasing current.
(m, c), cv = curve_fit(linear_fit, time, Isd)
Isd -= m * time
on_times, on_time_errs = [], []
off_times, off_time_errs = [], []
photocurrents, photocurrent_errs = [], []
plt.figure()
plt.plot(time, Isd)
# loop through regions and perform fitting calculations
for i in range(len(region_indices)-1):
    start, end = region_indices[i], region_indices[i+1]
    region_time = np.array(time[start: end])
    region_Isd = np.array(Isd[start: end])
    time_offset, current_offset = region_time[0], region_Isd[0]
    region_time -= time_offset
    region_Isd -= current_offset
    if i % 2 == 0:  # even i implies the region is an 'on' jump
        fit_func = on_exp
        times_array, time_errs_array = on_times, on_time_errs
    else:
        fit_func = off_exp
        times_array, time_errs_array = off_times, off_time_errs
    (I_jump, response_t), cv = curve_fit(fit_func, region_time, region_Isd)
    photocurrents.append(I_jump)
    photocurrent_errs.append(np.sqrt(cv[0, 0]))
    times_array.append(response_t)
    time_errs_array.append(np.sqrt(cv[1, 1]))
    smooth_times = np.linspace(0, region_time[-1], 100)
    plt.plot(smooth_times + time_offset,
             fit_func(smooth_times, I_jump, response_t) + current_offset,
             color='black', linestyle='dotted')
plt.legend(("Current", "Exponential fits"), loc='lower left')
# Testing reveals that the standard deviation of the various arrays is greater
# than the uncertainty propagated forwards from the covariance matrices.
print()
print("For CVD1, functionalised with quantum dots:")
print(f"The average photocurrent is {np.mean(photocurrents):e} ± {np.std(photocurrents):e} A")
print(f"The average 'on' response time is {np.mean(on_times):e} ± {np.std(on_times):e} seconds.")
print(f"The average 'off' response time is {np.mean(off_times):e} ± {np.std(off_times):e} seconds.")

# CVD2 calculations
device2_I_values = np.array([4.120, 4.160, 4.066, 4.026, 4.064, 4.102, 4.032,
                             3.985, 4.011, 4.054, 3.992, 3.948, 3.972, 4.013,
                             3.972, 3.932])*1e-6
magnitude_I_jumps2 = np.abs(device2_I_values[1::2] - device2_I_values[0::2])
device2_photocurrent = np.mean(magnitude_I_jumps2)
device2_photocurrent_err = np.std(magnitude_I_jumps2)
print()
print("For CVD2, functionalised with perovskites:")
print(f"The average photocurrent is {device2_photocurrent:e} ± {device2_photocurrent_err:e} A")

plt.show()
