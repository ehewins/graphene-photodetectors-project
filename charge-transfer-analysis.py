import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
ax.plot(region_t, region_I, color='k', linestyle='-', label='Data')
for n in range(3):
    ax.plot(region_t, fit_points_list[n], color=f'C{n}', linestyle='dotted',
            linewidth=3, label=f'Fit w/ $N={n+1}$')
    print("Fit params. for "+["one", "two", "three"][n]+" exponentials:")
    params, errors = param_list[n], error_list[n]
    for i in range(len(params)//2):
        a, a_err = params[2 * i], errors[2 * i]
        tau, tau_err = params[2 * i + 1], errors[2 * i + 1]
        result_string = f"a_{i+1:d} = {a:.2e} ± {a_err:.2e} A, "
        result_string += f"tau_{i+1:d} = {tau:.2f} ± {tau_err:.2f} s."
        print(result_string)
ax.legend()

print()
print("For the second region, equilibrating from light on to light off state:")
region_t = time[fit_region_indices[1][0]: fit_region_indices[1][1]]
region_I = current[fit_region_indices[1][0]: fit_region_indices[1][1]]
init_params = (1e-7, 7e2, -4e-7, 1e4, -5e-7, 6e4)
param_list, error_list, fit_points_list = triple_fitting_process(region_t,
                                                                 region_I,
                                                                 init_params)
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
ax.plot(region_t, region_I, color='k', linestyle='-', label='Data')
for n in range(3):
    ax.plot(region_t, fit_points_list[n], color=f'C{n}', linestyle='dotted',
            linewidth=3, label=f'Fit w/ $N={n+1}$')
    print("Fit params. for "+["one", "two", "three"][n]+" exponentials:")
    params, errors = param_list[n], error_list[n]
    for i in range(len(params)//2):
        a, a_err = params[2 * i], errors[2 * i]
        tau, tau_err = params[2 * i + 1], errors[2 * i + 1]
        result_string = f"a_{i+1:d} = {a:.2e} ± {a_err:.2e} A, "
        result_string += f"tau_{i+1:d} = {tau:.2f} ± {tau_err:.2f} s."
        print(result_string)
ax.legend()

# """
# Objective 2:
# Repeat this fitting process for the first long-time experiment we ran - the one
# with low res. data and a final current different from the initial current.
# """

# print()
# print("=============== Low Resolution Long Time Experiments [22/03/24] ===============")
# print()

# filename = "data/cvd_time_power_22032024/CVD1F-long-time-lowres.dat"
# time, current = np.loadtxt(filename, usecols=(0, 1), unpack=True)
# fit_region_indices = [[3680, 8464], [8464, len(current)-1]]

# print("For the first region, equilibrating from light off to light on state:")
# region_t = time[fit_region_indices[0][0]: fit_region_indices[0][1]]
# region_I = current[fit_region_indices[0][0]: fit_region_indices[0][1]]
# init_params = np.tile((1e-6, 3e2), 3)
# param_list, error_list, fit_points_list = triple_fitting_process(region_t,
#                                                                  region_I,
#                                                                  init_params)
# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# ax.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
# ax.set_xlabel("Time, $t$ (s)")
# ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
# ax.plot(region_t, region_I, color='k', linestyle='-', label='Data')
# for n in range(3):
#     ax.plot(region_t, fit_points_list[n], color=f'C{n}', linestyle='dotted',
#             linewidth=3, label=f'Fit w/ $N={n+1}$')
#     print("Fit params. for "+["one", "two", "three"][n]+" exponentials:")
#     params, errors = param_list[n], error_list[n]
#     for i in range(len(params)//2):
#         a, a_err = params[2 * i], errors[2 * i]
#         tau, tau_err = params[2 * i + 1], errors[2 * i + 1]
#         result_string = f"a_{i+1:d} = {a:.2e} ± {a_err:.2e} A, "
#         result_string += f"tau_{i+1:d} = {tau:.2f} ± {tau_err:.2f} s."
#         print(result_string)
# ax.legend()

# print()
# print("For the second region, equilibrating from light on to light off state:")
# region_t = time[fit_region_indices[1][0]: fit_region_indices[1][1]]
# region_I = current[fit_region_indices[1][0]: fit_region_indices[1][1]]
# init_params = (2e-1, 2e3, -2e-1, 2e3, 1, 1)
# param_list, error_list, fit_points_list = triple_fitting_process(region_t,
#                                                                  region_I,
#                                                                  init_params)
# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# ax.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
# ax.set_xlabel("Time, $t$ (s)")
# ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
# ax.plot(region_t, region_I, color='k', linestyle='-', label='Data')
# for n in range(3):
#     ax.plot(region_t, fit_points_list[n], color=f'C{n}', linestyle='dotted',
#             linewidth=3, label=f'Fit w/ $N={n+1}$')
#     print("Fit params. for "+["one", "two", "three"][n]+" exponentials:")
#     params, errors = param_list[n], error_list[n]
#     for i in range(len(params)//2):
#         a, a_err = params[2 * i], errors[2 * i]
#         tau, tau_err = params[2 * i + 1], errors[2 * i + 1]
#         result_string = f"a_{i+1:d} = {a:.2e} ± {a_err:.2e} A, "
#         result_string += f"tau_{i+1:d} = {tau:.2f} ± {tau_err:.2f} s."
#         print(result_string)
# ax.legend()

# """
# Objective 3:
# Repeat this fitting process for the fast charge transfer processes observed in
# Device 1 on 15/03/24, before the UV-induced changes made them unmeasurable.
# """

# print()
# print("========== First Post-Functionalisation Time Experiments [15/03/24] ==========")
# print()
# time, current = np.loadtxt("data/cvd_func_15032024/CVD1F-time.dat",
#                            usecols=(1, 3), unpack=True)
# # This array defines the indices bouding each on/off region
# region_indices = [29, 56, 84, 110, 138, 163, 187, 214, 240]
# # We want to remove some other, even faster effects, and the slow effects.
# for problem_index in region_indices[0:8:2]:
#     current[problem_index] = current[problem_index - 1]
# (m, c), cv = curve_fit(linear_fit, time, current)
# current -= m * time  # subtract (approximately) linear slow effect.

# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# ax.set_title("Fitting with $I_{sd} = I_0 + \\sum_{i=1}^N a_i [\\exp(-t/\\tau_i) - 1]$")
# ax.set_xlabel("Time, $t$ (s)")
# ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
# ax.plot(time, current, color='k', linestyle='-', label='Data')
# singles, doubles, triples = [], [], []
# labelled = False
# # init_params = np.tile((2e-8, 2e-1), 3)
# for i in range(len(region_indices)-1):
#     region_start, region_end = region_indices[i], region_indices[i+1]
#     region_t = np.array(time[region_start: region_end])
#     region_I = np.array(current[region_start: region_end])
#     # init_params[::2] *= -1
#     param_list, error_list, fit_points_list = triple_fitting_process(region_t,
#                                                                      region_I)
#     singles.append(param_list[0])
#     doubles.append(param_list[1])
#     triples.append(param_list[2])
#     for n in range(3):
#         ax.plot(region_t, fit_points_list[n], color=f'C{n}',
#                 linestyle='dotted', linewidth=3,
#                 label=f'Fit w/ $N={n+1}$' if not labelled else None)
#     labelled = True
# ax.legend()
# on_mean_1, on_std_1 = np.mean(singles[::2], axis=0), np.std(singles[::2], axis=0)
# off_mean_1, off_std_1 = np.mean(singles[1::2], axis=0), np.std(singles[1::2], axis=0)
# on_mean_2, on_std_2 = np.mean(doubles[::2], axis=0), np.std(doubles[::2], axis=0)
# off_mean_2, off_std_2 = np.mean(doubles[1::2], axis=0), np.std(doubles[1::2], axis=0)
# on_mean_3, on_std_3 = np.mean(triples[::2], axis=0), np.std(triples[::2], axis=0)
# off_mean_3, off_std_3 = np.mean(triples[1::2], axis=0), np.std(triples[1::2], axis=0)
# print("For the light-on regions:")
# print("Fit params. for one exponential:")
# for i in range(1):
#     a, a_err = on_mean_1[2 * i], on_std_1[2 * i]
#     tau, tau_err = on_mean_1[2 * i + 1], on_std_1[2 * i + 1]
#     result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
#     result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
#     print(result_string)
# print("Fit params. for two exponentials:")
# for i in range(2):
#     a, a_err = on_mean_2[2 * i], on_std_2[2 * i]
#     tau, tau_err = on_mean_2[2 * i + 1], on_std_2[2 * i + 1]
#     result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
#     result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
#     print(result_string)
# print("Fit params. for three exponentials:")
# for i in range(3):
#     a, a_err = on_mean_3[2 * i], on_std_3[2 * i]
#     tau, tau_err = on_mean_3[2 * i + 1], on_std_3[2 * i + 1]
#     result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
#     result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
#     print(result_string)
# print("For the light-off regions:")
# print("Fit params. for one exponential:")
# for i in range(1):
#     a, a_err = off_mean_1[2 * i], off_std_1[2 * i]
#     tau, tau_err = off_mean_1[2 * i + 1], off_std_1[2 * i + 1]
#     result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
#     result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
#     print(result_string)
# print("Fit params. for two exponentials:")
# for i in range(2):
#     a, a_err = off_mean_2[2 * i], off_std_2[2 * i]
#     tau, tau_err = off_mean_2[2 * i + 1], off_std_2[2 * i + 1]
#     result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
#     result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
#     print(result_string)
# print("Fit params. for three exponentials:")
# for i in range(3):
#     a, a_err = off_mean_3[2 * i], off_std_3[2 * i]
#     tau, tau_err = off_mean_3[2 * i + 1], off_std_3[2 * i + 1]
#     result_string = f"a_{i+1} = {a:.2e} ± {a_err:.2e} A, "
#     result_string += f"tau_{i+1} = {tau:.2f} ± {tau_err:.2f} s."
#     print(result_string)

plt.show()
