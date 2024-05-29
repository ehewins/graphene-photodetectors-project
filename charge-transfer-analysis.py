import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 14})


def linear_fit(x, m, c):
    return m * x + c


def single_exp(t, a, tau_a):
    return a * (np.exp(- t / tau_a) - 1)


def double_exp(t, a, tau_a, b, tau_b):
    return a * (np.exp(- t / tau_a) - 1) + b * (np.exp(- t / tau_b) - 1)


def triple_exp(t, a, tau_a, b, tau_b, c, tau_c):
    return a * (np.exp(- t / tau_a) - 1) \
        + b * (np.exp(- t / tau_b) - 1) \
        + c * (np.exp(- t / tau_c) - 1)


def triple_fitting_process(times, currents, guesses=None):
    # Arrays must begin at zero for fitting process
    fit_t = times - times[0]
    fit_I = currents - currents[0]

    param_values = []
    param_errors = []
    fit_data = []
    for fit_func, end in zip((single_exp, double_exp, triple_exp), (2, 4, 6)):
        popt, pcov = curve_fit(fit_func, fit_t, fit_I,
                               p0=None if guesses is None else guesses[:end],
                               maxfev=10000 if guesses is None else 1000)
        param_values.append(popt)
        param_errors.append(np.sqrt(np.diag(pcov)))
        fit_data.append(fit_func(fit_t, *popt))

    return param_values, param_errors, np.array(fit_data) + currents[0]


"""
Objective 1:
Model the charge transfer seen in the long-time experiments of 22/03/24 in
terms of multiple simultaneous charge transfer processes, and show the
improvement in fit quality from doing so.
"""
print()
print("====================== Long Time Experiments [22/03/24] ======================")
print()

filename = "data/cvd_time_power_22032024/CVD1F-long-time.dat"
time, current = np.loadtxt(filename, usecols=(1, 3), unpack=True)
fit_region_indices = [[5265, 8114], [8114, len(current)-1]]

print("For the first region, equilibrating from light off to light on state:")
region_t = time[fit_region_indices[0][0]: fit_region_indices[0][1]]
region_I = current[fit_region_indices[0][0]: fit_region_indices[0][1]]
init_params = (8e-8, 9e0, 3e-7, 1e2, 6e-7, 4e2)
param_list, error_list, fit_points_list = triple_fitting_process(region_t,
                                                                 region_I,
                                                                 init_params)
fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
# ax1.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
ax1.set_xlabel("Time, $t$ (hours)")
ax1.set_ylabel("Source-drain current, $I_{sd}$ ($\\mu$A)")
ax1.plot(region_t/60/60, 1e6*region_I, color='k', linestyle='-', label='Data')
for n in range(3):
    ax1.plot(region_t/60/60, 1e6*fit_points_list[n], ':', color=f'C{n}',
             linewidth=3, label=f'Fit w/ {n+1} exponent{"s" if n > 0 else ""}')
    print("Fit params. for "+["one", "two", "three"][n]+" exponentials:")
    params, errors = param_list[n], error_list[n]
    for i in range(len(params)//2):
        a, a_err = params[2 * i], errors[2 * i]
        tau, tau_err = params[2 * i + 1], errors[2 * i + 1]
        result_string = f"a_{i+1:d} = {a:.2e} ± {a_err:.2e} A, "
        result_string += f"tau_{i+1:d} = {tau:.2f} ± {tau_err:.2f} s."
        print(result_string)
ax1.legend()
fig.tight_layout()

print()
print("For the second region, equilibrating from light on to light off state:")
region_t = time[fit_region_indices[1][0]: fit_region_indices[1][1]]
region_I = current[fit_region_indices[1][0]: fit_region_indices[1][1]]
init_params = (1e-7, 7e2, -4e-7, 1e4, -5e-7, 6e4)
param_list, error_list, fit_points_list = triple_fitting_process(region_t,
                                                                 region_I,
                                                                 init_params)
fig, ax2 = plt.subplots(1, 1, figsize=(7, 6))
# ax2.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
ax2.set_xlabel("Time, $t$ (hours)")
ax2.set_ylabel("Source-drain current, $I_{sd}$ ($\\mu$A)")
ax2.plot(region_t/60/60, 1e6*region_I, 'k-', label='Data')
inset = ax2.inset_axes([0.5, 0.07, 0.45, 0.48],
                       xlim=(2.2, 5), ylim=(2.32, 2.56))
ax2.indicate_inset_zoom(inset, edgecolor='black')
inset.plot(region_t[region_t < 18000]/60/60, 1e6*region_I[region_t < 18000], 'k-')
for n in range(3):
    ax2.plot(region_t/60/60, 1e6*fit_points_list[n], ':', color=f'C{n}',
             linewidth=3, label=f'Fit w/ {n+1} exponent{"s" if n > 0 else ""}')
    inset.plot(region_t[region_t < 18000]/60/60,
               1e6*fit_points_list[n][region_t < 18000], ':',
               color=f'C{n}', linewidth=3)
    print("Fit params. for "+["one", "two", "three"][n]+" exponentials:")
    params, errors = param_list[n], error_list[n]
    for i in range(len(params)//2):
        a, a_err = params[2 * i], errors[2 * i]
        tau, tau_err = params[2 * i + 1], errors[2 * i + 1]
        result_string = f"a_{i+1:d} = {a:.2e} ± {a_err:.2e} A, "
        result_string += f"tau_{i+1:d} = {tau:.2f} ± {tau_err:.2f} s."
        print(result_string)
ax2.legend(loc='upper left')
fig.tight_layout()

"""
Objective 2:
Create a figure for the first long-time experiment we ran - the one with
low resolution data and a final current different from the initial current.
"""

# print()
# print("=============== Low Resolution Long Time Experiments [22/03/24] ===============")
# print()

filename = "data/cvd_time_power_22032024/CVD1F-long-time-lowres.dat"
time, current = np.loadtxt(filename, usecols=(0, 1), unpack=True)
time /= 60*60  # Convert time to seconds for plotting
current *= 1e6  # Convert current to microamps for plotting
fig, ax = plt.subplots(1, 1, figsize=(5, 6))
ax.plot(time, current, label='Data')
ax.set_xlabel("Time, $t$ (hours)")
ax.set_ylabel("Source-drain current, $I_{sd}$ ($\\mu$A)")

# Select end region of data to extrapolate from
fit_t = time[10000:]
fit_I = current[10000:]
offset_t = fit_t[0]
offset_I = fit_I[0]
fit_t -= offset_t
fit_I -= offset_I
(a, tau), cv = curve_fit(single_exp, fit_t, fit_I, p0=(1e-6, 3e2))
extrap_t = 25  # Time in hours to extrapolate up until
smooth_t = np.linspace(0, extrap_t-offset_t, 200)
smooth_I = single_exp(smooth_t, a, tau)
# Plot the extrapolation, and a line showing the initial current
ax.plot([0, extrap_t], [current[0], current[0]], 'k--',
        label='Initial current')
ax.plot(smooth_t+offset_t, smooth_I+offset_I, ':', color='C3', linewidth='2',
        label='Extrapolation')
ax.legend()
fig.tight_layout()

"""
Objective 3:
Repeat this fitting process for the fast charge transfer processes observed in
Device 1 on 15/03/24, before the UV-induced changes made them unmeasurable.
"""

print()
print("========== First Post-Functionalisation Time Experiments [15/03/24] ==========")
print()
time, current = np.loadtxt("data/cvd_func_15032024/CVD1F-time.dat",
                           usecols=(1, 3), unpack=True)
# This array defines the indices bounding each on/off region
region_indices = [29, 56, 84, 110, 138, 163, 187, 214, 240]
# We want to remove some other, even faster effects, and the slow effects.
for problem_index in region_indices[0:8:2]:
    current[problem_index] = current[problem_index - 1]
(m, c), cv = curve_fit(linear_fit, time, current)
current -= m * time  # subtract (approximately) linear slow effect.

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# ax.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel("Source-drain current, $I_{sd}$ ($\\mu$A)")
ax.plot(time, 1e6*current, color='k', linestyle='-', label='Data')
singles, doubles = [], []
unlabelled = 2
init_params_1 = [-6e-8, 1e-2, -2e-8, 8e-1]
init_params_2 = [3e-8, 1, 5e-8, 1e-1]
for n, (region_start, region_end) in enumerate(zip(region_indices[:-1],
                                                   region_indices[1:])):
    region_t = np.array(time[region_start: region_end])
    region_I = np.array(current[region_start: region_end])
    fit_t = region_t - region_t[0]
    fit_I = region_I - region_I[0]
    init_params = init_params_1 if n % 2 == 0 else init_params_2
    for fit_func, storage_array, p0, N, colour in zip((single_exp, double_exp),
                                                      (singles, doubles),
                                                      (None, init_params),
                                                      (1, 2),
                                                      ('C3', 'C9')):
        params, cv = curve_fit(fit_func, fit_t, fit_I, p0=p0)
        smooth_t = np.linspace(fit_t[0], fit_t[-1], 50)
        smooth_I = fit_func(smooth_t, *params)
        label = f"Fits w/ {N} exponent{'s' if N > 1 else ''}"
        ax.plot(smooth_t + region_t[0], 1e6*(smooth_I + region_I[0]), ':',
                label=label if unlabelled else None,
                color=colour, linewidth=2.5)
        unlabelled -= 1 if unlabelled != 0 else 0
        storage_array.append(params)
ax.legend(loc="lower left")
fig.tight_layout()
on_mean_1, on_std_1 = np.mean(singles[::2], axis=0), np.std(singles[::2], axis=0)
off_mean_1, off_std_1 = np.mean(singles[1::2], axis=0), np.std(singles[1::2], axis=0)
on_mean_2, on_std_2 = np.mean(doubles[::2], axis=0), np.std(doubles[::2], axis=0)
off_mean_2, off_std_2 = np.mean(doubles[1:-1:2], axis=0), np.std(doubles[1:-1:2], axis=0)
print("For the light-on regions:")
print("Fit params. for one exponential:")
for i in range(1):
    a, a_err = on_mean_1[2 * i], on_std_1[2 * i]
    tau, tau_err = on_mean_1[2 * i + 1], on_std_1[2 * i + 1]
    result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
    result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
    print(result_string)
print("Fit params. for two exponentials:")
for i in range(2):
    a, a_err = on_mean_2[2 * i], on_std_2[2 * i]
    tau, tau_err = on_mean_2[2 * i + 1], on_std_2[2 * i + 1]
    result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
    result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
    print(result_string)
print("For the light-off regions:")
print("Fit params. for one exponential:")
for i in range(1):
    a, a_err = off_mean_1[2 * i], off_std_1[2 * i]
    tau, tau_err = off_mean_1[2 * i + 1], off_std_1[2 * i + 1]
    result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
    result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
    print(result_string)
print("Fit params. for two exponentials:")
for i in range(2):
    a, a_err = off_mean_2[2 * i], off_std_2[2 * i]
    tau, tau_err = off_mean_2[2 * i + 1], off_std_2[2 * i + 1]
    result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
    result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
    print(result_string)

plt.show()
