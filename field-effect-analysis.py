import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


def top_hat(width):
    return np.ones(width) / width


"""
Objective 1:
Analyse how properties of the functionalised devices have changed between
measurements, including the location of the Dirac point (for the earlier
measurements, this must be estimated from the gradient), and mobility (").
"""

file_info = (
    ("data/cvd_func_15032024/CVD1F-Isd-Vg-Dark.dat", "15/03", 1),
    ("data/cvd_func_15032024/CVD1F-Isd-Vg-Light.dat", "15/03", 1),
    ("data/cvd_func_15032024/CVD2F-Isd-Vg-Dark.dat", "15/03", 1),
    # ("data/cvd_func_15032024/CVD2F-Isd-Vg-Light.dat", "15/03", 0),  # no linear region
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Dark.dat", "19/03 (1)", 2),
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Light.dat", "19/03 (1)", 2),
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max.dat", "19/03 (2)", 3),  # keep?
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max.dat", "19/03 (2)", 3),  # keep?
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max-2.dat", "19/03 (3)", 3),
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max-2.dat", "19/03 (3)", 3),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Dark.dat", "19/03 (1)", 1),
    # ("data/cvd_func_19032024/CVD2F-Isd-Vg-Light.dat", "19/03 (1)", 0),  # no linear region
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max.dat", "19/03 (2)", 3),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max.dat", "19/03 (2)", 2),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max-2.dat", "19/03 (3)", 3),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max-2.dat", "19/03 (3)", 2),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-post-long.dat", "22/03 (0)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-post-long.dat", "22/03 (0)", 3),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-final.dat", "22/03 (1)", 3),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Light-final.dat", "22/03 (1)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-final.dat", "22/03 (1)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Light-final.dat", "22/03 (1)", 3),
    ("data/cvd_high_current_26032024/CVD1F-Isd-Vg-Dark-final.dat", "26/03", 3),
    ("data/cvd_high_current_26032024/CVD1F-Isd-Vg-Light-final.dat", "26/03", 3),
    ("data/cvd_high_current_26032024/CVD2F-Isd-Vg-Dark-final.dat", "26/03", 3),
    ("data/cvd_high_current_26032024/CVD2F-Isd-Vg-Light-final.dat", "26/03", 3),
)

dVg = 0.2  # Gate voltage step resolution (V)
Vsd = 10e-3  # 10 mV source-drain voltage
d = 90e-9  # dielectric thickness (m)
epsilon_r = 3.9  # relative permittivity of dielectric
epsilon_0 = 8.85e-12  # permittivity of free space (F/m)
e = 1.6e-19  # charge magnitude of a single carrier (C)

width = 25  # width of top hat function for smoothing out the gradient function

# Creating empty data storage arrays
cvd1_dark_data = np.empty((0, 14, 4))
cvd1_light_data = np.empty((0, 14, 4))
cvd2_dark_data = np.empty((0, 10, 4))
cvd2_light_data = np.empty((0, 10, 4))
cvd1_dark_names = []
cvd1_light_names = []
cvd2_dark_names = []
cvd2_light_names = []

for filename, displayname, data_wealth in file_info:
    device = int(filename.split("/")[2][3])
    conditions = "Dark" if filename.split("/")[2][13] == "D" else "Light"
    results = np.nan * np.ones((14 if device == 1 else 10, 4))
    # if data_wealth < 1:
    #     continue
    Vg_data, Isd_data = np.loadtxt(filename, usecols=(0, 3), unpack=True)
    datapoints_per_pass = int((max(Vg_data) - min(Vg_data)) / dVg)
    num_passes = int(len(Vg_data) / datapoints_per_pass)
    dirac_voltages = []
    for p_num in range(0, num_passes, 2):  # only study the forward passes
        start_index = p_num * datapoints_per_pass
        stop_index = (p_num + 1) * datapoints_per_pass
        # sub-arrays for Vg_data and Isd_data in this pass
        Vg = Vg_data[start_index: stop_index]
        Isd = Isd_data[start_index: stop_index]
        # Resistivity equals resistance because the graphene channel is square
        sigma, rho = Isd / Vsd, Vsd / Isd  # conductivity and resistivity

        # Find a local maxima in the magnitude of sigma's gradient.
        # The gradient is very noisy, so locating the maxima requires smoothing
        gradient = np.gradient(sigma, Vg)
        gradient_pad = np.pad(gradient, (width-1)//2, mode='edge')
        gradient_smooth = np.convolve(gradient_pad, top_hat(width), mode='valid')
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # fig.suptitle(f"{filename}, pass {p_num}")
        # ax1.plot(Vg, sigma)
        # ax2.plot(Vg, gradient)
        # ax2.plot(Vg, gradient_smooth)
        gradient2 = np.gradient(np.abs(gradient_smooth), Vg)
        max_grad_index = np.nonzero((Vg < Vg[np.argmin(sigma)]) &
                                    (gradient2 > 0))[0][-1]
        # Calculate the max gradient from a small region around this point
        near_gradients = gradient[max_grad_index-5:
                                  min(max_grad_index+6, len(gradient))]
        near_sigmas = sigma[max_grad_index-5:
                            min(max_grad_index+6, len(gradient))]
        near_Vgs = Vg[max_grad_index-5: min(max_grad_index+6, len(gradient))]

        # Parameters for graphical plot of chosen point and their uncertainties
        m, a, b = np.mean((near_gradients, near_Vgs, near_sigmas), axis=1)
        dm, da, db = np.std((near_gradients, near_Vgs, near_sigmas), axis=1)
        c = b - m * a
        dc = np.sqrt(db**2 + (dm/a)**2 + (da/m)**2)
        # Calculate the mobility and extrapolate a Dirac voltage estimate
        mu_max = d/(epsilon_0 * epsilon_r) * abs(m)
        mu_max_err = d/(epsilon_0 * epsilon_r) * dm
        # p0 = (epsilon_0 * epsilon_r) / (e * d) * -c/m
        # p0_err = p0 * np.sqrt((dc/c)**2 + (dm/m)**2)
        V_dirac = -c/m
        V_dirac_err = V_dirac * np.sqrt((dc/c)**2 + (dm/m)**2)
        # Store the results of these calculations
        results[0:4, p_num//2] = mu_max, mu_max_err, V_dirac, V_dirac_err

        # Calculating additional results when the Dirac point is observed
        if data_wealth < 2:
            continue
        V_dirac = Vg[np.argmax(rho)]
        # p0 = (epsilon_0 * epsilon_r) / (e * d) * V_dirac
        # p0_err = (epsilon_0 * epsilon_r) / (e * d) * dVg
        rho_max = np.max(rho)
        # If the file under analysis has both sides of N.P. visable, we only
        # want to consider the left one for now.
        halfmax_index = np.argmin(np.abs(rho * [Vg < V_dirac] - rho_max/2))
        near_gradients = gradient[halfmax_index-5: halfmax_index+6]
        mu_halfmax = d/(epsilon_0 * epsilon_r) * abs(np.mean(near_gradients))
        mu_halfmax_err = d/(epsilon_0 * epsilon_r) * np.std(near_gradients)
        Vg_deviation = a / Vg[halfmax_index] - 1
        mu_deviation = mu_max / mu_halfmax - 1
        # Store the results of these calculations
        results[4:8, p_num//2] = V_dirac, rho_max, mu_halfmax, mu_halfmax_err

        # Calculating additional results when there's plenty of data to the
        # right of the Dirac point.
        if data_wealth < 3:
            continue
        try:
            max_grad_index = np.nonzero((Vg > V_dirac + 0.5) &
                                        (gradient2 < 0))[0][0]
        except:  # Some files need a different approach
            max_grad_index = np.nonzero((Vg > V_dirac + 0.5) &
                                        (gradient2 > 0))[0][-1]
        # Calculate the max gradient from a small region around this point
        near_gradients = gradient[max_grad_index-5:
                                  min(max_grad_index+6, len(gradient))]
        near_sigmas = sigma[max_grad_index-5:
                            min(max_grad_index+6, len(gradient))]
        near_Vgs = Vg[max_grad_index-5: min(max_grad_index+6, len(gradient))]

        # Parameters for graphical plot of chosen point and their uncertainties
        m, a, b = np.mean((near_gradients, near_Vgs, near_sigmas), axis=1)
        dm, da, db = np.std((near_gradients, near_Vgs, near_sigmas), axis=1)
        c = b - m * a
        dc = np.sqrt(db**2 + (dm/a)**2 + (da/m)**2)
        # Calculate the mobility
        mu_max = d/(epsilon_0 * epsilon_r) * abs(m)
        mu_max_err = d/(epsilon_0 * epsilon_r) * dm
        # Store the results of these calculations
        results[8:10, p_num//2] = mu_max, mu_max_err

        # Additional FWHM calculations, not possible for device 2
        if device == 2:
            continue
        halfmax_index_2 = np.argmin(np.abs(rho * [Vg > V_dirac] - rho_max/2))
        near_gradients = gradient[halfmax_index_2-5: halfmax_index_2+6]
        mu_halfmax = d/(epsilon_0 * epsilon_r) * np.mean(near_gradients)
        mu_halfmax_err = d/(epsilon_0 * epsilon_r) * np.std(near_gradients)
        Vg_deviation = a / Vg[halfmax_index_2] - 1
        mu_deviation = mu_max / mu_halfmax - 1
        # VWHM stuff and 3-point mobility calculation
        V_FWHM = Vg[halfmax_index_2] - Vg[halfmax_index]
        delta_n = V_FWHM * (epsilon_0 * epsilon_r) / (e * d)
        mu_3p = 4 / (e * delta_n * rho_max)
        # Store the results of these calculations
        results[10:14, p_num//2] = mu_halfmax, mu_halfmax_err, delta_n, mu_3p

    # Now we've looped through all the passes, store the results.
    match (device, conditions):
        case (1, "Dark"):
            cvd1_dark_data = np.append(cvd1_dark_data, [results], axis=0)
            cvd1_dark_names.append(displayname)
        case (1, "Light"):
            cvd1_light_data = np.append(cvd1_light_data, [results], axis=0)
            cvd1_light_names.append(displayname)
        case (2, "Dark"):
            cvd2_dark_data = np.append(cvd2_dark_data, [results], axis=0)
            cvd2_dark_names.append(displayname)
        case (2, "Light"):
            cvd2_light_data = np.append(cvd2_light_data, [results], axis=0)
            cvd2_light_names.append(displayname)

    # ax.plot(range(1, len(dirac_voltages)+1), dirac_voltages, '-o', label=date)
    # ax.set_xticks(range(1, 5))
    # ax.set_title(f"Device {device}, {conditions.lower()} conditions")
    # ax.set_xlabel("Gate voltage sweep no.")
    # ax.set_ylabel("Dirac voltage, $V_{Dirac}$ (V)")
    # ax.legend(loc='lower right')

# Graph how the Dirac has changed over time
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
for n, (label, data) in enumerate(zip(cvd1_dark_names, cvd1_dark_data)):
    estimates, estimate_errs = data[2, :], data[3, :]
    known_vals = data[4, :]
    ax1a.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
    ax1a.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
for n, (label, data) in enumerate(zip(cvd1_light_names, cvd1_light_data)):
    estimates, estimate_errs = data[2, :], data[3, :]
    known_vals = data[4, :]
    ax1b.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
    ax1b.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
for n, (label, data) in enumerate(zip(cvd2_dark_names, cvd2_dark_data)):
    estimates, estimate_errs = data[2, :], data[3, :]
    known_vals = data[4, :]
    ax2a.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
    ax2a.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
for n, (label, data) in enumerate(zip(cvd2_light_names, cvd2_light_data)):
    estimates, estimate_errs = data[2, :], data[3, :]
    known_vals = data[4, :]
    ax2b.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
    ax2b.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
fig1.suptitle("Device 1 (Quantum dots)")
fig2.suptitle("Device 2 (Perovskites)")
ax1a.set_title("Dark conditions")
ax1b.set_title("Light conditions")
ax2a.set_title("Dark conditions")
ax2b.set_title("Light conditions")
for ax in (ax1a, ax1b, ax2a, ax2b):
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Gate voltage sweep no.")
    ax.set_ylabel("Dirac voltage, $V_{Dirac}$ (V)")
    ax.legend(loc='lower right')

# Graph how the hole mobility has changed over time
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
for n, (label, data) in enumerate(zip(cvd1_dark_names, cvd1_dark_data)):
    mu_maxes, mu_max_errs = data[0, :], data[1, :]
    mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
    ax1a.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
    ax1a.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
for n, (label, data) in enumerate(zip(cvd1_light_names, cvd1_light_data)):
    mu_maxes, mu_max_errs = data[0, :], data[1, :]
    mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
    ax1b.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
    ax1b.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
for n, (label, data) in enumerate(zip(cvd2_dark_names, cvd2_dark_data)):
    mu_maxes, mu_max_errs = data[0, :], data[1, :]
    mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
    ax2a.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
    ax2a.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
for n, (label, data) in enumerate(zip(cvd2_light_names, cvd2_light_data)):
    mu_maxes, mu_max_errs = data[0, :], data[1, :]
    mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
    ax2b.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
    ax2b.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
fig1.suptitle("Device 1 (Quantum dots)")
fig2.suptitle("Device 2 (Perovskites)")
ax1a.set_title("Dark conditions")
ax1b.set_title("Light conditions")
ax2a.set_title("Dark conditions")
ax2b.set_title("Light conditions")
for ax in (ax1a, ax1b, ax2a, ax2b):
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Gate voltage sweep no.")
    ax.set_ylabel("Hole mobility, $\\mu_p$ (cm$^2$ V${^-1}$ s$^{-1}$)")
    ax.legend(loc='lower right')

# Graph how the electron mobility has changed over time
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
for n, (label, data) in enumerate(zip(cvd1_dark_names, cvd1_dark_data)):
    mu_maxes, mu_max_errs = data[8, :], data[9, :]
    if all(np.isnan(mu_maxes)):
        continue
    ax1a.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
    mu_FWHMs, mu_FWHM_errs = data[10, :], data[11, :]
    if all(np.isnan(mu_FWHMs)):
        continue
    ax1a.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
for n, (label, data) in enumerate(zip(cvd1_light_names, cvd1_light_data)):
    mu_maxes, mu_max_errs = data[8, :], data[9, :]
    if all(np.isnan(mu_maxes)):
        continue
    ax1b.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
    mu_FWHMs, mu_FWHM_errs = data[10, :], data[11, :]
    if all(np.isnan(mu_FWHMs)):
        continue
    ax1b.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
fig1.suptitle("Device 1 (Quantum dots)")
ax1a.set_title("Dark conditions")
ax1b.set_title("Light conditions")
for ax in (ax1a, ax1b):
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Gate voltage sweep no.")
    ax.set_ylabel("Electron mobility, $\\mu_n$ (cm$^2$ V${^-1}$ s$^{-1}$)")
    ax.legend(loc='lower right')

# Graph how the maximum resistivity has changed over time
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
for n, (label, rho) in enumerate(zip(cvd1_dark_names, cvd1_dark_data[:, 5, :])):
    if all(np.isnan(rho)):
        continue
    ax1a.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
for n, (label, rho) in enumerate(zip(cvd1_light_names, cvd1_light_data[:, 5, :])):
    if all(np.isnan(rho)):
        continue
    ax1b.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
for n, (label, rho) in enumerate(zip(cvd2_dark_names, cvd2_dark_data[:, 5, :])):
    if all(np.isnan(rho)):
        continue
    ax2a.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
for n, (label, rho) in enumerate(zip(cvd2_light_names, cvd2_light_data[:, 5, :])):
    if all(np.isnan(rho)):
        continue
    ax2b.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
fig1.suptitle("Device 1 (Quantum dots)")
fig2.suptitle("Device 2 (Perovskites)")
ax1a.set_title("Dark conditions")
ax1b.set_title("Light conditions")
ax2a.set_title("Dark conditions")
ax2b.set_title("Light conditions")
for ax in (ax1a, ax1b, ax2a, ax2b):
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Gate voltage sweep no.")
    ax.set_ylabel("Maximum resistivity, $\\rho_{max}$ ($\\Omega/sq$)")
    ax.legend(loc='lower right')

# Graph how δn and the 3-point mobility change over time (Device 1 only).
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
for n, (label, delta_n) in enumerate(zip(cvd1_dark_names, cvd1_dark_data[:, 12, :])):
    if all(np.isnan(delta_n)):
        continue
    ax1a.plot(range(1, len(delta_n)+1), delta_n*1e-4, '--o', label=label, color=f'C{n}')
for n, (label, delta_n) in enumerate(zip(cvd1_light_names, cvd1_light_data[:, 12, :])):
    if all(np.isnan(delta_n)):
        continue
    ax1b.plot(range(1, len(delta_n)+1), delta_n*1e-4, '--o', label=label, color=f'C{n}')
ax1a.set_title("Dark conditions")
ax1b.set_title("Light conditions")
for ax in (ax1a, ax1b):
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Gate voltage sweep no.")
    ax.set_ylabel("$\\delta n$ (cm$^{-2}$)")
    ax.legend(loc='lower right')
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
for n, (label, mu_3p) in enumerate(zip(cvd1_dark_names, cvd1_dark_data[:, 13, :])):
    if all(np.isnan(mu_3p)):
        continue
    ax1a.plot(range(1, len(mu_3p)+1), mu_3p*1e4, '--o', label=label, color=f'C{n}')
for n, (label, mu_3p) in enumerate(zip(cvd1_light_names, cvd1_light_data[:, 13, :])):
    if all(np.isnan(mu_3p)):
        continue
    ax1b.plot(range(1, len(mu_3p)+1), mu_3p*1e4, '--o', label=label, color=f'C{n}')
ax1a.set_title("Dark conditions")
ax1b.set_title("Light conditions")
for ax in (ax1a, ax1b):
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Gate voltage sweep no.")
    ax.set_ylabel("3-point mobility, $\\mu_{3p}$ (cm$^2$ V${^-1}$ s$^{-1}$)")
    ax.legend(loc='lower right')


plt.show()
