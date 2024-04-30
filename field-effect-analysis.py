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
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-post-long.dat", "22/03 (1)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-post-long.dat", "22/03 (1)", 3),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-final.dat", "22/03 (2)", 3),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Light-final.dat", "22/03 (2)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-final.dat", "22/03 (2)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Light-final.dat", "22/03 (2)", 3),
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
dark1_data = np.empty((0, 11, 4))
light1_data = np.empty((0, 11, 4))
dark2_data = np.empty((0, 9, 4))
light2_data = np.empty((0, 9, 4))
dark1_names = []
light1_names = []
dark2_names = []
light2_names = []

for filename, displayname, data_wealth in file_info:
    device = int(filename.split("/")[2][3])
    conditions = "Dark" if filename.split("/")[2][13] == "D" else "Light"
    results = np.nan * np.ones((11 if device == 1 else 9, 4))
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
        # ax1.plot([a], [b], 'bx')
        # ax1.plot([min(Vg), -c/m], [m*min(Vg)+c, 0], 'k--')
        # Calculate the mobility and extrapolate a Dirac voltage estimate
        mu_max = d/(epsilon_0 * epsilon_r) * abs(m)
        mu_max_err = d/(epsilon_0 * epsilon_r) * dm
        # p0 = (epsilon_0 * epsilon_r) / (e * d) * -c/m
        # p0_err = p0 * np.sqrt((dc/c)**2 + (dm/m)**2)
        V_dirac = -c/m
        V_dirac_err = V_dirac * np.sqrt((dc/c)**2 + (dm/m)**2)
        # Store the results of these calculations
        results[0:2, p_num//2] = mu_max, V_dirac

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
        # ax1.plot([Vg[halfmax_index]], [sigma[halfmax_index]], 'rx')
        near_gradients = gradient[halfmax_index-5: halfmax_index+6]
        mu_halfmax = d/(epsilon_0 * epsilon_r) * abs(np.mean(near_gradients))
        mu_halfmax_err = d/(epsilon_0 * epsilon_r) * np.std(near_gradients)
        V_diff = Vg[halfmax_index] - a
        if device == 1 and displayname != "19/03 (1)":
            halfmax_index_2 = np.argmin(np.abs(rho * [Vg > V_dirac] - rho_max/2))
            # ax1.plot([Vg[halfmax_index_2]], [sigma[halfmax_index_2]], 'rx')
            V_FWHM = Vg[halfmax_index_2] - Vg[halfmax_index]
        else:
            V_FWHM = 2 * (V_dirac - Vg[halfmax_index])
        # VWHM stuff and 3-point mobility calculation
        delta_n = V_FWHM * (epsilon_0 * epsilon_r) / (e * d)
        mu_3p = 4 / (e * delta_n * rho_max)
        # Store the results of these calculations
        results[2:8, p_num//2] = V_dirac, rho_max, delta_n, mu_3p, mu_halfmax, V_diff

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
        # ax1.plot([a], [b], 'bx')
        # ax1.plot([max(Vg), -c/m], [m*max(Vg)+c, 0], 'k--')
        # Calculate the mobility
        mu_max = d/(epsilon_0 * epsilon_r) * abs(m)
        mu_max_err = d/(epsilon_0 * epsilon_r) * dm
        # Store the results of these calculations
        results[8, p_num//2] = mu_max
        if device == 1:
            near_gradients = gradient[halfmax_index_2-5: halfmax_index_2+6]
            mu_halfmax_2 = d/(epsilon_0 * epsilon_r) * np.mean(near_gradients)
            mu_halfmax_2_err = d/(epsilon_0 * epsilon_r) * np.std(near_gradients)
            V_diff = Vg[halfmax_index_2] - a
            results[9:11, p_num//2] = mu_halfmax_2, V_diff

    # Now we've looped through all the passes, store the results.
    match (device, conditions):
        case (1, "Dark"):
            dark1_data = np.append(dark1_data, [results], axis=0)
            dark1_names.append(displayname)
        case (1, "Light"):
            light1_data = np.append(light1_data, [results], axis=0)
            light1_names.append(displayname)
        case (2, "Dark"):
            dark2_data = np.append(dark2_data, [results], axis=0)
            dark2_names.append(displayname)
        case (2, "Light"):
            light2_data = np.append(light2_data, [results], axis=0)
            light2_names.append(displayname)

# Create plot showing how the Dirac voltage has varied between measurements
dark1_dirac_Vs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in dark1_data[:, 2, :]])
dark1_dirac_V_errs = np.nanstd(dark1_data[:, 2, :], axis=1)
light1_dirac_Vs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light1_data[:, 2, :]])
light1_dirac_V_errs = np.nanstd(light1_data[:, 2, :], axis=1)
dark2_dirac_Vs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in dark2_data[:, 2, :]])
dark2_dirac_V_errs = np.nanstd(dark2_data[:, 2, :], axis=1)
light2_dirac_Vs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light2_data[:, 2, :]])
light2_dirac_V_errs = np.nanstd(light2_data[:, 2, :], axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Measurement set label \n(date and number)")
    ax.set_ylabel("Dirac voltage, $V_{Dirac}$ (V)")
ax1.errorbar(dark1_names, dark1_dirac_Vs, yerr=dark1_dirac_V_errs, color='C0', ecolor='C3', fmt='o', capsize=4)
ax1.errorbar(light1_names, light1_dirac_Vs, yerr=light1_dirac_V_errs, color='C2', ecolor='C3', fmt='o', capsize=4)
ax2.errorbar(dark2_names, dark2_dirac_Vs, yerr=dark2_dirac_V_errs, label='Dark conditions', color='C0', ecolor='C3', fmt='o', capsize=4)
ax2.errorbar(light2_names, light2_dirac_Vs, yerr=light2_dirac_V_errs, label='Light conditions', color='C2', ecolor='C3', fmt='o', capsize=4)
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 2, :].flatten(), '.', color='C0')
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 2, :].flatten(), '.', color='C2')
ax2.plot(np.tile(np.array([dark2_names]).T, 4).flatten(),
         dark2_data[:, 2, :].flatten(), '.', color='C0')
ax2.plot(np.tile(np.array([light2_names]).T, 4).flatten(),
         light2_data[:, 2, :].flatten(), '.', color='C2')
fig.legend()

# Create plot showing how the hole mobilities have varied between measurements
# (mobilities determined from maximum gradient)
dark1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 0, :]])
dark1_mu_max_errs = np.nanstd(dark1_data[:, 0, :], axis=1)
light1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 0, :]])
light1_mu_max_errs = np.nanstd(light1_data[:, 0, :], axis=1)
dark2_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark2_data[:, 0, :]])
dark2_mu_max_errs = np.nanstd(dark2_data[:, 0, :], axis=1)
light2_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light2_data[:, 0, :]])
light2_mu_max_errs = np.nanstd(light2_data[:, 0, :], axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Measurement set label \n(date and number)")
    ax.set_ylabel("Hole mobility, $\\mu_p$ (cm$^2$ V${^-1}$ s$^{-1}$)")
ax1.errorbar(dark1_names, dark1_mu_maxes*1e4, yerr=dark1_mu_max_errs*1e4, color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_maxes*1e4, yerr=light1_mu_max_errs*1e4, color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(dark2_names, dark2_mu_maxes*1e4, yerr=dark2_mu_max_errs*1e4, label='$\\mu_{max}$, Dark conditions', color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(light2_names, light2_mu_maxes*1e4, yerr=light2_mu_max_errs*1e4, label='$\\mu_{max}$, Light conditions', color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 0, :].flatten()*1e4, 'o', color='C0', markersize=3)
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 0, :].flatten()*1e4, 'o', color='C2', markersize=3)
ax2.plot(np.tile(np.array([dark2_names]).T, 4).flatten(),
         dark2_data[:, 0, :].flatten()*1e4, 'o', color='C0', markersize=3)
ax2.plot(np.tile(np.array([light2_names]).T, 4).flatten(),
         light2_data[:, 0, :].flatten()*1e4, 'o', color='C2', markersize=3)
# On the same plot, show the mobilities at the FWHM
dark1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 6, :]])
dark1_mu_FWHM_errs = np.nanstd(dark1_data[:, 6, :], axis=1)
light1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 6, :]])
light1_mu_FWHM_errs = np.nanstd(light1_data[:, 6, :], axis=1)
dark2_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark2_data[:, 6, :]])
dark2_mu_FWHM_errs = np.nanstd(dark2_data[:, 6, :], axis=1)
light2_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light2_data[:, 6, :]])
light2_mu_FWHM_errs = np.nanstd(light2_data[:, 6, :], axis=1)
ax1.errorbar(dark1_names, dark1_mu_FWHMs*1e4, yerr=dark1_mu_FWHM_errs*1e4, color='C0', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_FWHMs*1e4, yerr=light1_mu_FWHM_errs*1e4, color='C2', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax2.errorbar(dark2_names, dark2_mu_FWHMs*1e4, yerr=dark2_mu_FWHM_errs*1e4, label='$\\mu_{FWHM}$, Dark conditions', color='C0', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax2.errorbar(light2_names, light2_mu_FWHMs*1e4, yerr=light2_mu_FWHM_errs*1e4, label='$\\mu_{FWHM}$, Light conditions', color='C2', ecolor='C3', fmt='^', markersize=6, capsize=4)
fig.legend()
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 6, :].flatten()*1e4, '^', color='C0', markersize=3)
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 6, :].flatten()*1e4, '^', color='C2', markersize=3)
ax2.plot(np.tile(np.array([dark2_names]).T, 4).flatten(),
         dark2_data[:, 6, :].flatten()*1e4, '^', color='C0', markersize=3)
ax2.plot(np.tile(np.array([light2_names]).T, 4).flatten(),
         light2_data[:, 6, :].flatten()*1e4, '^', color='C2', markersize=3)

# Create plot showing how the electron mobilities have varied between measurements
dark1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 8, :]])
dark1_mu_max_errs = np.nanstd(dark1_data[:, 8, :], axis=1)
light1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 8, :]])
light1_mu_max_errs = np.nanstd(light1_data[:, 8, :], axis=1)
dark2_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark2_data[:, 8, :]])
dark2_mu_max_errs = np.nanstd(dark2_data[:, 8, :], axis=1)
light2_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light2_data[:, 8, :]])
light2_mu_max_errs = np.nanstd(light2_data[:, 8, :], axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Measurement set label \n(date and number)")
    ax.set_ylabel("Electron mobility, $\\mu_n$ (cm$^2$ V${^-1}$ s$^{-1}$)")
ax1.errorbar(dark1_names, dark1_mu_maxes*1e4, yerr=dark1_mu_max_errs*1e4, color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_maxes*1e4, yerr=light1_mu_max_errs*1e4, color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(dark2_names, dark2_mu_maxes*1e4, yerr=dark2_mu_max_errs*1e4, label='$\\mu_{max}$, Dark conditions', color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(light2_names, light2_mu_maxes*1e4, yerr=light2_mu_max_errs*1e4, label='$\\mu_{max}$, Light conditions', color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 8, :].flatten()*1e4, 'o', color='C0', markersize=3)
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 8, :].flatten()*1e4, 'o', color='C2', markersize=3)
ax2.plot(np.tile(np.array([dark2_names]).T, 4).flatten(),
         dark2_data[:, 8, :].flatten()*1e4, 'o', color='C0', markersize=3)
ax2.plot(np.tile(np.array([light2_names]).T, 4).flatten(),
         light2_data[:, 8, :].flatten()*1e4, 'o', color='C2', markersize=3)
# On the same plot, show the mobilities at the FWHM
dark1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 9, :]])
dark1_mu_FWHM_errs = np.nanstd(dark1_data[:, 9, :], axis=1)
light1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 9, :]])
light1_mu_FWHM_errs = np.nanstd(light1_data[:, 9, :], axis=1)
ax1.errorbar(dark1_names, dark1_mu_FWHMs*1e4, yerr=dark1_mu_FWHM_errs*1e4, label='$\\mu_{FWHM}$, Dark conditions', color='C0', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_FWHMs*1e4, yerr=light1_mu_FWHM_errs*1e4, label='$\\mu_{FWHM}$, Light conditions', color='C2', ecolor='C3', fmt='^', markersize=6, capsize=4)
fig.legend()
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 9, :].flatten()*1e4, '^', color='C0', markersize=3)
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 9, :].flatten()*1e4, '^', color='C2', markersize=3)
fig.legend()

# Print results of how other properties were effected by the long experiments
# First define which experiments took place before the long experiment:
dark1_prelong = np.array([int(date[:2]) < 22 for date in dark1_names])
light1_prelong = np.array([int(date[:2]) < 22 for date in light1_names])
dark2_prelong = np.array([int(date[:2]) < 22 for date in dark2_names])
light2_prelong = np.array([int(date[:2]) < 22 for date in light2_names])
# Then extract values from the storage arrays
dark1_rhos = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                       else np.nan for row in dark1_data[:, 3, :]])
light1_rhos = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                        else np.nan for row in light1_data[:, 3, :]])
dark2_rhos = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                       else np.nan for row in dark2_data[:, 3, :]])
light2_rhos = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                        else np.nan for row in light2_data[:, 3, :]])
dark1_delta_ns = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in dark1_data[:, 4, :]])
light1_delta_ns = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light1_data[:, 4, :]])
dark2_delta_ns = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in dark2_data[:, 4, :]])
light2_delta_ns = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light2_data[:, 4, :]])
dark1_mu_3ps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark1_data[:, 5, :]])
light1_mu_3ps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in light1_data[:, 5, :]])
dark2_mu_3ps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark2_data[:, 5, :]])
light2_mu_3ps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in light2_data[:, 5, :]])
dark1_V_diffs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark1_data[:, 7, :]])
light1_V_diffs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in light1_data[:, 7, :]])
dark2_V_diffs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark2_data[:, 7, :]])
light2_V_diffs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in light2_data[:, 7, :]])
print()
print("Parameter value averages before the long experiments:")
print("Max resistivity for Device 1 (dark conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(dark1_rhos[dark1_prelong]),
                                      np.nanstd(dark1_rhos[dark1_prelong])))
print("Max resistivity for Device 1 (light conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(light1_rhos[light1_prelong]),
                                      np.nanstd(light1_rhos[light1_prelong])))
print("Max resistivity for Device 2 (dark conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(dark2_rhos[dark2_prelong]),
                                      np.nanstd(dark2_rhos[dark2_prelong])))
print("Max resistivity for Device 2 (light conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(light2_rhos[light2_prelong]),
                                      np.nanstd(light2_rhos[light2_prelong])))
print("δn for Device 1 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(dark1_delta_ns[dark1_prelong]),
                                       1e-4*np.nanstd(dark1_delta_ns[dark1_prelong])))
print("δn for Device 1 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(light1_delta_ns[light1_prelong]),
                                       1e-4*np.nanstd(light1_delta_ns[light1_prelong])))
print("δn for Device 2 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(dark2_delta_ns[dark2_prelong]),
                                       1e-4*np.nanstd(dark2_delta_ns[dark2_prelong])))
print("δn for Device 2 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(light2_delta_ns[light2_prelong]),
                                       1e-4*np.nanstd(light2_delta_ns[light2_prelong])))
print("3-point mobility for Device 1 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(dark1_mu_3ps[dark1_prelong]),
                                                1e4*np.nanstd(dark1_mu_3ps[dark1_prelong])))
print("3-point mobility for Device 1 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(light1_mu_3ps[light1_prelong]),
                                                1e4*np.nanstd(light1_mu_3ps[light1_prelong])))
print("3-point mobility for Device 2 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(dark2_mu_3ps[dark2_prelong]),
                                                1e4*np.nanstd(dark2_mu_3ps[dark2_prelong])))
print("3-point mobility for Device 2 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(light2_mu_3ps[light2_prelong]),
                                                1e4*np.nanstd(light2_mu_3ps[light2_prelong])))
print("max-mobility/FWHM voltage difference for Device 1 (dark conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(dark1_V_diffs[dark1_prelong]),
                                   np.nanstd(dark1_V_diffs[dark1_prelong])))
print("max-mobility/FWHM voltage difference for Device 1 (light conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(light1_V_diffs[light1_prelong]),
                                   np.nanstd(light1_V_diffs[light1_prelong])))
print("max-mobility/FWHM voltage difference for Device 2 (dark conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(dark2_V_diffs[dark2_prelong]),
                                   np.nanstd(dark2_V_diffs[dark2_prelong])))
print("max-mobility/FWHM voltage difference for Device 2 (light conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(light2_V_diffs[light2_prelong]),
                                   np.nanstd(light2_V_diffs[light2_prelong])))
print()
print("Parameter value averages after the long experiments:")
print("Max resistivity for Device 1 (dark conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(dark1_rhos[~dark1_prelong]),
                                      np.nanstd(dark1_rhos[~dark1_prelong])))
print("Max resistivity for Device 1 (light conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(light1_rhos[~light1_prelong]),
                                      np.nanstd(light1_rhos[~light1_prelong])))
print("Max resistivity for Device 2 (dark conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(dark2_rhos[~dark2_prelong]),
                                      np.nanstd(dark2_rhos[~dark2_prelong])))
print("Max resistivity for Device 2 (light conditions): " +
      "{0:.2e} ± {1:.2e} Ω/sq".format(np.nanmean(light2_rhos[~light2_prelong]),
                                      np.nanstd(light2_rhos[~light2_prelong])))
print("δn for Device 1 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(dark1_delta_ns[~dark1_prelong]),
                                       1e-4*np.nanstd(dark1_delta_ns[~dark1_prelong])))
print("δn for Device 1 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(light1_delta_ns[~light1_prelong]),
                                       1e-4*np.nanstd(light1_delta_ns[~light1_prelong])))
print("δn for Device 2 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(dark2_delta_ns[~dark2_prelong]),
                                       1e-4*np.nanstd(dark2_delta_ns[~dark2_prelong])))
print("δn for Device 2 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^-2".format(1e-4*np.nanmean(light2_delta_ns[~light2_prelong]),
                                       1e-4*np.nanstd(light2_delta_ns[~light2_prelong])))
print("3-point mobility for Device 1 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(dark1_mu_3ps[~dark1_prelong]),
                                                1e4*np.nanstd(dark1_mu_3ps[~dark1_prelong])))
print("3-point mobility for Device 1 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(light1_mu_3ps[~light1_prelong]),
                                                1e4*np.nanstd(light1_mu_3ps[~light1_prelong])))
print("3-point mobility for Device 2 (dark conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(dark2_mu_3ps[~dark2_prelong]),
                                                1e4*np.nanstd(dark2_mu_3ps[~dark2_prelong])))
print("3-point mobility for Device 2 (light conditions): " +
      "{0:.2e} ± {1:.2e} cm^2 V^-1 s^-1".format(1e4*np.nanmean(light2_mu_3ps[~light2_prelong]),
                                                1e4*np.nanstd(light2_mu_3ps[~light2_prelong])))
print("max-mobility/FWHM voltage difference for Device 1 (dark conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(dark1_V_diffs[~dark1_prelong]),
                                   np.nanstd(dark1_V_diffs[~dark1_prelong])))
print("max-mobility/FWHM voltage difference for Device 1 (light conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(light1_V_diffs[~light1_prelong]),
                                   np.nanstd(light1_V_diffs[~light1_prelong])))
print("max-mobility/FWHM voltage difference for Device 2 (dark conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(dark2_V_diffs[~dark2_prelong]),
                                   np.nanstd(dark2_V_diffs[~dark2_prelong])))
print("max-mobility/FWHM voltage difference for Device 2 (light conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(light2_V_diffs[~light2_prelong]),
                                   np.nanstd(light2_V_diffs[~light2_prelong])))

# Overshoot in V_dirac estimate vs. actual V_dirac (when both are measured)
dark1_dirac_V_extraps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                  else np.nan for row in dark1_data[:, 1, :]])
light1_dirac_V_extraps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                   else np.nan for row in light1_data[:, 1, :]])
dark2_dirac_V_extraps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                  else np.nan for row in dark2_data[:, 1, :]])
light2_dirac_V_extraps = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                   else np.nan for row in light2_data[:, 1, :]])
dark1_dirac_V_overshoots = np.abs(dark1_dirac_V_extraps - dark1_dirac_Vs)
light1_dirac_V_overshoots = np.abs(light1_dirac_V_extraps - light1_dirac_Vs)
dark2_dirac_V_overshoots = np.abs(dark2_dirac_V_extraps - dark2_dirac_Vs)
light2_dirac_V_overshoots = np.abs(light2_dirac_V_extraps - light2_dirac_Vs)
print()
print("Average voltage difference between the actual Dirac voltage, and \
linear extrapolations from the point with maximum mobility:")
print("For Device 1 (dark conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(dark1_dirac_V_overshoots),
                                   np.nanstd(dark1_dirac_V_overshoots)))
print("For Device 1 (light conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(light1_dirac_V_overshoots),
                                   np.nanstd(light1_dirac_V_overshoots)))
print("For Device 2 (dark conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(dark2_dirac_V_overshoots),
                                   np.nanstd(dark2_dirac_V_overshoots)))
print("For Device 2 (light conditions): " +
      "{0:.1f} ± {1:.1f} V".format(np.nanmean(light2_dirac_V_overshoots),
                                   np.nanstd(light2_dirac_V_overshoots)))

# Comparing maximum (hole) mobility to the 3-point mobility
# I've already redefined these, so I've got to set them back the right values
dark1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 0, :]])
light1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 0, :]])
dark2_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark2_data[:, 0, :]])
light2_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light2_data[:, 0, :]])
dark1_mu_3p_diffs = np.abs(dark1_mu_3ps / dark1_mu_maxes - 1)
light1_mu_3p_diffs = np.abs(light1_mu_3ps / light1_mu_maxes - 1)
dark2_mu_3p_diffs = np.abs(dark2_mu_3ps / dark2_mu_maxes - 1)
light2_mu_3p_diffs = np.abs(light2_mu_3ps / light2_mu_maxes - 1)
print()
print("Mean percentage difference between mu_3p and mu_max (p)")
print("For Device 1 (dark conditions): " +
      "{0:.1f} ± {1:.1f} %".format(100*np.nanmean(dark1_mu_3p_diffs),
                                   100*np.nanstd(dark1_mu_3p_diffs)))
print("For Device 1 (light conditions): " +
      "{0:.1f} ± {1:.1f} %".format(100*np.nanmean(light1_mu_3p_diffs),
                                   100*np.nanstd(light1_mu_3p_diffs)))
print("For Device 2 (dark conditions): " +
      "{0:.2f} ± {1:.2f} %".format(100*np.nanmean(dark2_mu_3p_diffs),
                                   100*np.nanstd(dark2_mu_3p_diffs)))
print("For Device 2 (light conditions): " +
      "{0:.2f} ± {1:.2f} %".format(100*np.nanmean(light2_mu_3p_diffs),
                                   100*np.nanstd(light2_mu_3p_diffs)))

# Comparing (numerically) the agreement between mu_max and mu_FWHM
# First for hole mobility
# I've already redefined these, so I've got to set them back the right values
dark1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 6, :]])
light1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 6, :]])
dark1_mu_FWHM_diffs = np.abs(dark1_mu_FWHMs / dark1_mu_maxes - 1)
light1_mu_FWHM_diffs = np.abs(light1_mu_FWHMs / light1_mu_maxes - 1)
dark2_mu_FWHM_diffs = np.abs(dark2_mu_FWHMs / dark2_mu_maxes - 1)
light2_mu_FWHM_diffs = np.abs(light2_mu_FWHMs / light2_mu_maxes - 1)
print()
print("Mean percentage difference between mu_FWHM and mu_max (p)")
print("For Device 1 (dark conditions): " +
      "{0:.1f} ± {1:.1f} %".format(100*np.nanmean(dark1_mu_FWHM_diffs),
                                   100*np.nanstd(dark1_mu_FWHM_diffs)))
print("For Device 1 (light conditions): " +
      "{0:.1f} ± {1:.1f} %".format(100*np.nanmean(light1_mu_FWHM_diffs),
                                   100*np.nanstd(light1_mu_FWHM_diffs)))
print("For Device 2 (dark conditions): " +
      "{0:.2f} ± {1:.2f} %".format(100*np.nanmean(dark2_mu_FWHM_diffs),
                                   100*np.nanstd(dark2_mu_FWHM_diffs)))
print("For Device 2 (light conditions): " +
      "{0:.2f} ± {1:.2f} %".format(100*np.nanmean(light2_mu_FWHM_diffs),
                                   100*np.nanstd(light2_mu_FWHM_diffs)))
# Second for electron mobility (device 1 only)
dark1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 8, :]])
light1_mu_maxes = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 8, :]])
dark1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 9, :]])
light1_mu_FWHMs = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 9, :]])
dark1_mu_FWHM_diffs = np.abs(dark1_mu_FWHMs / dark1_mu_maxes - 1)
light1_mu_FWHM_diffs = np.abs(light1_mu_FWHMs / light1_mu_maxes - 1)
print()
print("Mean percentage difference between mu_FWHM and mu_max (n)")
print("For Device 1 (dark conditions): " +
      "{0:.1f} ± {1:.1f} %".format(100*np.nanmean(dark1_mu_FWHM_diffs),
                                   100*np.nanstd(dark1_mu_FWHM_diffs)))
print("For Device 1 (light conditions): " +
      "{0:.1f} ± {1:.1f} %".format(100*np.nanmean(light1_mu_FWHM_diffs),
                                   100*np.nanstd(light1_mu_FWHM_diffs)))

###############################################################################
# Crap below vvvvvvvvvvvv

# # Graph how the Dirac has changed over time
# fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
# fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
# for n, (label, data) in enumerate(zip(dark1_names, dark1_data)):
#     estimates, estimate_errs = data[2, :], data[3, :]
#     known_vals = data[4, :]
#     ax1a.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
#     ax1a.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
# for n, (label, data) in enumerate(zip(light1_names, light1_data)):
#     estimates, estimate_errs = data[2, :], data[3, :]
#     known_vals = data[4, :]
#     ax1b.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
#     ax1b.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
# for n, (label, data) in enumerate(zip(dark2_names, dark2_data)):
#     estimates, estimate_errs = data[2, :], data[3, :]
#     known_vals = data[4, :]
#     ax2a.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
#     ax2a.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
# for n, (label, data) in enumerate(zip(light2_names, light2_data)):
#     estimates, estimate_errs = data[2, :], data[3, :]
#     known_vals = data[4, :]
#     ax2b.plot(range(1, len(estimates)+1), estimates, '--o', label=label, color=f'C{n}')
#     ax2b.plot(range(1, len(known_vals)+1), known_vals, '-o', color=f'C{n}')
# fig1.suptitle("Device 1 (Quantum dots)")
# fig2.suptitle("Device 2 (Perovskites)")
# ax1a.set_title("Dark conditions")
# ax1b.set_title("Light conditions")
# ax2a.set_title("Dark conditions")
# ax2b.set_title("Light conditions")
# for ax in (ax1a, ax1b, ax2a, ax2b):
#     ax.set_xticks(range(1, 5))
#     ax.set_xlabel("Gate voltage sweep no.")
#     ax.set_ylabel("Dirac voltage, $V_{Dirac}$ (V)")
#     ax.legend(loc='lower right')

# # Graph how the hole mobility has changed over time
# fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
# fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
# for n, (label, data) in enumerate(zip(dark1_names, dark1_data)):
#     mu_maxes, mu_max_errs = data[0, :], data[1, :]
#     mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
#     ax1a.errorbar(range(1, len(mu_maxes)+1), mu_maxes*1e4, yerr=mu_max_errs*1e4, fmt='--o', label=label, color=f'C{n}')
#     ax1a.errorbar(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, yerr=mu_FWHM_errs*1e4, fmt='-o', color=f'C{n}')
# for n, (label, data) in enumerate(zip(light1_names, light1_data)):
#     mu_maxes, mu_max_errs = data[0, :], data[1, :]
#     mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
#     ax1b.errorbar(range(1, len(mu_maxes)+1), mu_maxes*1e4, yerr=mu_max_errs*1e4, fmt='--o', label=label, color=f'C{n}')
#     ax1b.errorbar(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, yerr=mu_FWHM_errs*1e4, fmt='-o', color=f'C{n}')
# for n, (label, data) in enumerate(zip(dark2_names, dark2_data)):
#     mu_maxes, mu_max_errs = data[0, :], data[1, :]
#     mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
#     ax2a.errorbar(range(1, len(mu_maxes)+1), mu_maxes*1e4, yerr=mu_max_errs*1e4, fmt='--o', label=label, color=f'C{n}')
#     ax2a.errorbar(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, yerr=mu_FWHM_errs*1e4, fmt='-o', color=f'C{n}')
# for n, (label, data) in enumerate(zip(light2_names, light2_data)):
#     mu_maxes, mu_max_errs = data[0, :], data[1, :]
#     mu_FWHMs, mu_FWHM_errs = data[6, :], data[7, :]
#     ax2b.errorbar(range(1, len(mu_maxes)+1), mu_maxes*1e4, yerr=mu_max_errs*1e4, fmt='--o', label=label, color=f'C{n}')
#     ax2b.errorbar(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, yerr=mu_FWHM_errs*1e4, fmt='-o', color=f'C{n}')
# fig1.suptitle("Device 1 (Quantum dots)")
# fig2.suptitle("Device 2 (Perovskites)")
# ax1a.set_title("Dark conditions")
# ax1b.set_title("Light conditions")
# ax2a.set_title("Dark conditions")
# ax2b.set_title("Light conditions")
# for ax in (ax1a, ax1b, ax2a, ax2b):
#     ax.set_xticks(range(1, 5))
#     ax.set_xlabel("Gate voltage sweep no.")
#     ax.set_ylabel("Hole mobility, $\\mu_p$ (cm$^2$ V${^-1}$ s$^{-1}$)")
#     ax.legend(loc='lower right')

# # Graph how the electron mobility has changed over time
# fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
# for n, (label, data) in enumerate(zip(dark1_names, dark1_data)):
#     mu_maxes, mu_max_errs = data[8, :], data[9, :]
#     if all(np.isnan(mu_maxes)):
#         continue
#     ax1a.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
#     mu_FWHMs, mu_FWHM_errs = data[10, :], data[11, :]
#     if all(np.isnan(mu_FWHMs)):
#         continue
#     ax1a.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
# for n, (label, data) in enumerate(zip(light1_names, light1_data)):
#     mu_maxes, mu_max_errs = data[8, :], data[9, :]
#     if all(np.isnan(mu_maxes)):
#         continue
#     ax1b.plot(range(1, len(mu_maxes)+1), mu_maxes*1e4, '--o', label=label, color=f'C{n}')
#     mu_FWHMs, mu_FWHM_errs = data[10, :], data[11, :]
#     if all(np.isnan(mu_FWHMs)):
#         continue
#     ax1b.plot(range(1, len(mu_FWHMs)+1), mu_FWHMs*1e4, '-o', color=f'C{n}')
# fig1.suptitle("Device 1 (Quantum dots)")
# ax1a.set_title("Dark conditions")
# ax1b.set_title("Light conditions")
# for ax in (ax1a, ax1b):
#     ax.set_xticks(range(1, 5))
#     ax.set_xlabel("Gate voltage sweep no.")
#     ax.set_ylabel("Electron mobility, $\\mu_n$ (cm$^2$ V${^-1}$ s$^{-1}$)")
#     ax.legend(loc='lower right')

# # Graph how the maximum resistivity has changed over time
# fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
# fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
# for n, (label, rho) in enumerate(zip(dark1_names, dark1_data[:, 5, :])):
#     if all(np.isnan(rho)):
#         continue
#     ax1a.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
# for n, (label, rho) in enumerate(zip(light1_names, light1_data[:, 5, :])):
#     if all(np.isnan(rho)):
#         continue
#     ax1b.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
# for n, (label, rho) in enumerate(zip(dark2_names, dark2_data[:, 5, :])):
#     if all(np.isnan(rho)):
#         continue
#     ax2a.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
# for n, (label, rho) in enumerate(zip(light2_names, light2_data[:, 5, :])):
#     if all(np.isnan(rho)):
#         continue
#     ax2b.plot(range(1, len(rho)+1), rho, '--o', label=label, color=f'C{n}')
# fig1.suptitle("Device 1 (Quantum dots)")
# fig2.suptitle("Device 2 (Perovskites)")
# ax1a.set_title("Dark conditions")
# ax1b.set_title("Light conditions")
# ax2a.set_title("Dark conditions")
# ax2b.set_title("Light conditions")
# for ax in (ax1a, ax1b, ax2a, ax2b):
#     ax.set_xticks(range(1, 5))
#     ax.set_xlabel("Gate voltage sweep no.")
#     ax.set_ylabel("Maximum resistivity, $\\rho_{max}$ ($\\Omega/sq$)")
#     ax.legend(loc='lower right')

# # Graph how δn and the 3-point mobility change over time (Device 1 only).
# fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
# for n, (label, delta_n) in enumerate(zip(dark1_names, dark1_data[:, 12, :])):
#     if all(np.isnan(delta_n)):
#         continue
#     ax1a.plot(range(1, len(delta_n)+1), delta_n*1e-4, '--o', label=label, color=f'C{n}')
# for n, (label, delta_n) in enumerate(zip(light1_names, light1_data[:, 12, :])):
#     if all(np.isnan(delta_n)):
#         continue
#     ax1b.plot(range(1, len(delta_n)+1), delta_n*1e-4, '--o', label=label, color=f'C{n}')
# ax1a.set_title("Dark conditions")
# ax1b.set_title("Light conditions")
# for ax in (ax1a, ax1b):
#     ax.set_xticks(range(1, 5))
#     ax.set_xlabel("Gate voltage sweep no.")
#     ax.set_ylabel("$\\delta n$ (cm$^{-2}$)")
#     ax.legend(loc='lower right')
# fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
# for n, (label, mu_3p) in enumerate(zip(dark1_names, dark1_data[:, 13, :])):
#     if all(np.isnan(mu_3p)):
#         continue
#     ax1a.plot(range(1, len(mu_3p)+1), mu_3p*1e4, '--o', label=label, color=f'C{n}')
# for n, (label, mu_3p) in enumerate(zip(light1_names, light1_data[:, 13, :])):
#     if all(np.isnan(mu_3p)):
#         continue
#     ax1b.plot(range(1, len(mu_3p)+1), mu_3p*1e4, '--o', label=label, color=f'C{n}')
# ax1a.set_title("Dark conditions")
# ax1b.set_title("Light conditions")
# for ax in (ax1a, ax1b):
#     ax.set_xticks(range(1, 5))
#     ax.set_xlabel("Gate voltage sweep no.")
#     ax.set_ylabel("3-point mobility, $\\mu_{3p}$ (cm$^2$ V${^-1}$ s$^{-1}$)")
#     ax.legend(loc='lower right')

plt.show()
