import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 14})


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
    ("data/cvd_func_15032024/CVD1F-Isd-Vg-Dark.dat", "Day 1", 1),
    ("data/cvd_func_15032024/CVD1F-Isd-Vg-Light.dat", "Day 1", 1),
    ("data/cvd_func_15032024/CVD2F-Isd-Vg-Dark.dat", "Day 1", 1),
    # ("data/cvd_func_15032024/CVD2F-Isd-Vg-Light.dat", "Day 1", 0),  # no linear region
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Dark.dat", "Day 2 (1)", 2),
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Light.dat", "Day 2 (1)", 2),
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max.dat", "Day 2 (2)", 3),  # keep?
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max.dat", "Day 2 (2)", 3),  # keep?
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max-2.dat", "Day 2 (3)", 3),
    ("data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max-2.dat", "Day 2 (3)", 3),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Dark.dat", "Day 2 (1)", 1),
    # ("data/cvd_func_19032024/CVD2F-Isd-Vg-Light.dat", "Day 2 (1)", 0),  # no linear region
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max.dat", "Day 2 (2)", 3),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max.dat", "Day 2 (2)", 2),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max-2.dat", "Day 2 (3)", 3),
    ("data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max-2.dat", "Day 2 (3)", 2),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-post-long.dat", "Day 3 (1)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-post-long.dat", "Day 3 (1)", 3),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-final.dat", "Day 3 (2)", 3),
    ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Light-final.dat", "Day 3 (2)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-final.dat", "Day 3 (2)", 3),
    ("data/cvd_time_power_22032024/CVD2F-Isd-Vg-Light-final.dat", "Day 3 (2)", 3),
    ("data/cvd_high_current_26032024/CVD1F-Isd-Vg-Dark-final.dat", "Day 4", 3),
    ("data/cvd_high_current_26032024/CVD1F-Isd-Vg-Light-final.dat", "Day 4", 3),
    ("data/cvd_high_current_26032024/CVD2F-Isd-Vg-Dark-final.dat", "Day 4", 3),
    ("data/cvd_high_current_26032024/CVD2F-Isd-Vg-Light-final.dat", "Day 4", 3),
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
        elif device == 2:
            V_FWHM = 2 * (V_dirac - Vg[halfmax_index])
        else:
            V_FWHM = np.nan  # for Device 1, 19/03 (1)
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

# Create arrays for the final value of each parameter in each condition
# the standard deviation of the measurements leading up to that final value.
dark1_mu_max_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 0, :]])
dark1_dirac_V_extrap = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                else np.nan for row in dark1_data[:, 1, :]])
dark1_dirac_V = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark1_data[:, 2, :]])
dark1_rho_max = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark1_data[:, 3, :]])
dark1_delta_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark1_data[:, 4, :]])
dark1_mu_3p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                       else np.nan for row in dark1_data[:, 5, :]])
dark1_mu_FWHM_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in dark1_data[:, 6, :]])
dark1_V_diff_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 7, :]])
dark1_mu_max_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 8, :]])
dark1_mu_FWHM_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in dark1_data[:, 9, :]])
dark1_V_diff_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark1_data[:, 10, :]])
dark1_mu_max_p_err = np.nanstd(dark1_data[:, 0, :], axis=1)
dark1_dirac_V_extrap_err = np.nanstd(dark1_data[:, 1, :], axis=1)
dark1_dirac_V_err = np.nanstd(dark1_data[:, 2, :], axis=1)
dark1_rho_max_err = np.nanstd(dark1_data[:, 3, :], axis=1)
dark1_delta_n_err = np.nanstd(dark1_data[:, 4, :], axis=1)
dark1_mu_3p_err = np.nanstd(dark1_data[:, 5, :], axis=1)
dark1_mu_FWHM_p_err = np.nanstd(dark1_data[:, 6, :], axis=1)
dark1_V_diff_p_err = np.nanstd(dark1_data[:, 7, :], axis=1)
dark1_mu_max_n_err = np.nanstd(dark1_data[:, 8, :], axis=1)
dark1_mu_FWHM_n_err = np.nanstd(dark1_data[:, 9, :], axis=1)
dark1_V_diff_n_err = np.nanstd(dark1_data[:, 10, :], axis=1)
light1_mu_max_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light1_data[:, 0, :]])
light1_dirac_V_extrap = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                  else np.nan for row in light1_data[:, 1, :]])
light1_dirac_V = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 2, :]])
light1_rho_max = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 3, :]])
light1_delta_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light1_data[:, 4, :]])
light1_mu_3p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in light1_data[:, 5, :]])
light1_mu_FWHM_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                             else np.nan for row in light1_data[:, 6, :]])
light1_V_diff_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light1_data[:, 7, :]])
light1_mu_max_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light1_data[:, 8, :]])
light1_mu_FWHM_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                             else np.nan for row in light1_data[:, 9, :]])
light1_V_diff_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light1_data[:, 10, :]])
light1_mu_max_p_err = np.nanstd(light1_data[:, 0, :], axis=1)
light1_dirac_V_extrap_err = np.nanstd(light1_data[:, 1, :], axis=1)
light1_dirac_V_err = np.nanstd(light1_data[:, 2, :], axis=1)
light1_rho_max_err = np.nanstd(light1_data[:, 3, :], axis=1)
light1_delta_n_err = np.nanstd(light1_data[:, 4, :], axis=1)
light1_mu_3p_err = np.nanstd(light1_data[:, 5, :], axis=1)
light1_mu_FWHM_p_err = np.nanstd(light1_data[:, 6, :], axis=1)
light1_V_diff_p_err = np.nanstd(light1_data[:, 7, :], axis=1)
light1_mu_max_n_err = np.nanstd(light1_data[:, 8, :], axis=1)
light1_mu_FWHM_n_err = np.nanstd(light1_data[:, 9, :], axis=1)
light1_V_diff_n_err = np.nanstd(light1_data[:, 10, :], axis=1)
dark2_mu_max_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark2_data[:, 0, :]])
dark2_dirac_V_extrap = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                else np.nan for row in dark2_data[:, 1, :]])
dark2_dirac_V = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark2_data[:, 2, :]])
dark2_rho_max = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark2_data[:, 3, :]])
dark2_delta_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in dark2_data[:, 4, :]])
dark2_mu_3p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                       else np.nan for row in dark2_data[:, 5, :]])
dark2_mu_FWHM_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in dark2_data[:, 6, :]])
dark2_V_diff_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark2_data[:, 7, :]])
dark2_mu_max_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                          else np.nan for row in dark2_data[:, 8, :]])
dark2_mu_max_p_err = np.nanstd(dark2_data[:, 0, :], axis=1)
dark2_dirac_V_extrap_err = np.nanstd(dark2_data[:, 1, :], axis=1)
dark2_dirac_V_err = np.nanstd(dark2_data[:, 2, :], axis=1)
dark2_rho_max_err = np.nanstd(dark2_data[:, 3, :], axis=1)
dark2_delta_n_err = np.nanstd(dark2_data[:, 4, :], axis=1)
dark2_mu_3p_err = np.nanstd(dark2_data[:, 5, :], axis=1)
dark2_mu_FWHM_p_err = np.nanstd(dark2_data[:, 6, :], axis=1)
dark2_V_diff_p_err = np.nanstd(dark2_data[:, 7, :], axis=1)
dark2_mu_max_n_err = np.nanstd(dark2_data[:, 8, :], axis=1)
light2_mu_max_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light2_data[:, 0, :]])
light2_dirac_V_extrap = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                                  else np.nan for row in light2_data[:, 1, :]])
light2_dirac_V = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light2_data[:, 2, :]])
light2_rho_max = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light2_data[:, 3, :]])
light2_delta_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                           else np.nan for row in light2_data[:, 4, :]])
light2_mu_3p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                         else np.nan for row in light2_data[:, 5, :]])
light2_mu_FWHM_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                             else np.nan for row in light2_data[:, 6, :]])
light2_V_diff_p = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light2_data[:, 7, :]])
light2_mu_max_n = np.array([row[~np.isnan(row)][-1] if any(~np.isnan(row))
                            else np.nan for row in light2_data[:, 8, :]])
light2_mu_max_p_err = np.nanstd(light2_data[:, 0, :], axis=1)
light2_dirac_V_extrap_err = np.nanstd(light2_data[:, 1, :], axis=1)
light2_dirac_V_err = np.nanstd(light2_data[:, 2, :], axis=1)
light2_rho_max_err = np.nanstd(light2_data[:, 3, :], axis=1)
light2_delta_n_err = np.nanstd(light2_data[:, 4, :], axis=1)
light2_mu_3p_err = np.nanstd(light2_data[:, 5, :], axis=1)
light2_mu_FWHM_p_err = np.nanstd(light2_data[:, 6, :], axis=1)
light2_V_diff_p_err = np.nanstd(light2_data[:, 7, :], axis=1)
light2_mu_max_n_err = np.nanstd(light2_data[:, 8, :], axis=1)

# Create plot showing how the Dirac voltage has varied between measurements
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Day and measurement numbers")
    ax.set_ylabel("Dirac voltage, $V_{Dirac}$ (V)")
ax1.errorbar(dark1_names, dark1_dirac_V, yerr=dark1_dirac_V_err, label='Dark conditions', color='C0', ecolor='C3', fmt='o', capsize=4)
ax1.errorbar(light1_names, light1_dirac_V, yerr=light1_dirac_V_err, label='Light conditions', color='C2', ecolor='C3', fmt='o', capsize=4)
ax2.errorbar(dark2_names, dark2_dirac_V, yerr=dark2_dirac_V_err, label='Dark conditions', color='C0', ecolor='C3', fmt='o', capsize=4)
ax2.errorbar(light2_names, light2_dirac_V, yerr=light2_dirac_V_err, label='Light conditions', color='C2', ecolor='C3', fmt='o', capsize=4)
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 2, :].flatten(), '.', color='C0')
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 2, :].flatten(), '.', color='C2')
ax2.plot(np.tile(np.array([dark2_names]).T, 4).flatten(),
         dark2_data[:, 2, :].flatten(), '.', color='C0')
ax2.plot(np.tile(np.array([light2_names]).T, 4).flatten(),
         light2_data[:, 2, :].flatten(), '.', color='C2')
ax1.legend()
ax2.legend()
# ax1.set_xticklabels(dark1_names, fontsize=12)
# ax2.set_xticklabels(dark2_names, fontsize=12)
fig.tight_layout()

# Create plot showing how the hole mobilities have varied between measurements
# (mobilities determined from maximum gradient)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Day and measurement numbers")
    ax.set_ylabel("Hole mobility, $\\mu_p$ (cm$^2$ V${^-1}$ s$^{-1}$)")
ax1.errorbar(dark1_names, dark1_mu_max_p*1e4, yerr=dark1_mu_max_p_err*1e4, label='$\\mu_{max}$', color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_max_p*1e4, yerr=light1_mu_max_p_err*1e4, label='$\\mu_{max}$', color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(dark2_names, dark2_mu_max_p*1e4, yerr=dark2_mu_max_p_err*1e4, label='$\\mu_{max}$', color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(light2_names, light2_mu_max_p*1e4, yerr=light2_mu_max_p_err*1e4, label='$\\mu_{max}$', color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
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
ax1.errorbar(dark1_names, dark1_mu_FWHM_p*1e4, yerr=dark1_mu_FWHM_p_err*1e4, label='$\\mu_{FWHM}$ (Dark)', color='C0', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_FWHM_p*1e4, yerr=light1_mu_FWHM_p_err*1e4, label='$\\mu_{FWHM}$ (Light)', color='C2', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax2.errorbar(dark2_names, dark2_mu_FWHM_p*1e4, yerr=dark2_mu_FWHM_p_err*1e4, label='$\\mu_{FWHM}$ (Dark)', color='C0', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax2.errorbar(light2_names, light2_mu_FWHM_p*1e4, yerr=light2_mu_FWHM_p_err*1e4, label='$\\mu_{FWHM}$ (Light)', color='C2', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax1.legend(ncols=2)
ax2.legend(ncols=2)
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 6, :].flatten()*1e4, '^', color='C0', markersize=3)
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 6, :].flatten()*1e4, '^', color='C2', markersize=3)
ax2.plot(np.tile(np.array([dark2_names]).T, 4).flatten(),
         dark2_data[:, 6, :].flatten()*1e4, '^', color='C0', markersize=3)
ax2.plot(np.tile(np.array([light2_names]).T, 4).flatten(),
         light2_data[:, 6, :].flatten()*1e4, '^', color='C2', markersize=3)
ax1.set_xticklabels(dark1_names, fontsize=12)
ax2.set_xticklabels(dark2_names, fontsize=12)
fig.tight_layout()

# Create plot showing how the electron mobilities have varied between measurements
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Day and measurement numbers")
    ax.set_ylabel("Electron mobility, $\\mu_n$ (cm$^2$ V${^-1}$ s$^{-1}$)")
ax1.errorbar(dark1_names, dark1_mu_max_n*1e4, yerr=dark1_mu_max_n_err*1e4, label='$\\mu_{max}$', color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_max_n*1e4, yerr=light1_mu_max_n_err*1e4, label='$\\mu_{max}$', color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(dark2_names, dark2_mu_max_n*1e4, yerr=dark2_mu_max_n_err*1e4, label='$\\mu_{max}$ (Dark)', color='C0', ecolor='C3', fmt='o', markersize=6, capsize=4)
ax2.errorbar(light2_names, light2_mu_max_n*1e4, yerr=light2_mu_max_n_err*1e4, label='$\\mu_{max}$ (Light)', color='C2', ecolor='C3', fmt='o', markersize=6, capsize=4)
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
ax1.errorbar(dark1_names, dark1_mu_FWHM_n*1e4, yerr=dark1_mu_FWHM_n_err*1e4, label='$\\mu_{FWHM}$ (Dark)', color='C0', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax1.errorbar(light1_names, light1_mu_FWHM_n*1e4, yerr=light1_mu_FWHM_n_err*1e4, label='$\\mu_{FWHM}$ (Light)', color='C2', ecolor='C3', fmt='^', markersize=6, capsize=4)
ax1.legend(ncols=2)
ax2.legend(ncols=2, loc='lower left')
# Also plot the individual datapoints which go into calculating the mean
ax1.plot(np.tile(np.array([dark1_names]).T, 4).flatten(),
         dark1_data[:, 9, :].flatten()*1e4, '^', color='C0', markersize=3)
ax1.plot(np.tile(np.array([light1_names]).T, 4).flatten(),
         light1_data[:, 9, :].flatten()*1e4, '^', color='C2', markersize=3)
# ax1.set_xticklabels(dark1_names, fontsize=12)
# ax2.set_xticklabels(dark2_names, fontsize=12)
fig.tight_layout()

# Calculating some extra parameters:
# Overshoot in V_dirac estimate vs. actual V_dirac (when both are measured)
dark1_dirac_V_overshoots = np.abs(dark1_dirac_V_extrap - dark1_dirac_V)
light1_dirac_V_overshoots = np.abs(light1_dirac_V_extrap - light1_dirac_V)
dark2_dirac_V_overshoots = np.abs(dark2_dirac_V_extrap - dark2_dirac_V)
light2_dirac_V_overshoots = np.abs(light2_dirac_V_extrap - light2_dirac_V)
# Comparing maximum (hole) mobility to the 3-point mobility
dark1_mu_3p_diffs = np.abs(dark1_mu_3p / dark1_mu_max_p - 1)
light1_mu_3p_diffs = np.abs(light1_mu_3p / light1_mu_max_p - 1)
dark2_mu_3p_diffs = np.abs(dark2_mu_3p / dark2_mu_max_p - 1)
light2_mu_3p_diffs = np.abs(light2_mu_3p / light2_mu_max_p - 1)
# Comparing (numerically) the agreement between mu_max and mu_FWHM
# First for hole mobility
dark1_mu_FWHM_p_diffs = np.abs(dark1_mu_FWHM_p / dark1_mu_max_p - 1)
light1_mu_FWHM_p_diffs = np.abs(light1_mu_FWHM_p / light1_mu_max_p - 1)
dark2_mu_FWHM_p_diffs = np.abs(dark2_mu_FWHM_p / dark2_mu_max_p - 1)
light2_mu_FWHM_p_diffs = np.abs(light2_mu_FWHM_p / light2_mu_max_p - 1)
# Second for electron mobility (device 1 only)
dark1_mu_FWHM_n_diffs = np.abs(dark1_mu_FWHM_n / dark1_mu_max_n - 1)
light1_mu_FWHM_n_diffs = np.abs(light1_mu_FWHM_n / light1_mu_max_n - 1)

# Print results of how other properties were effected by the long experiments
# First define which experiments took place before the long experiment:
dark1_prelong = np.array([int(day[4]) < 3 for day in dark1_names])
light1_prelong = np.array([int(day[4]) < 3 for day in light1_names])
dark2_prelong = np.array([int(day[4]) < 3 for day in dark2_names])
light2_prelong = np.array([int(day[4]) < 3 for day in light2_names])

# Names to print when analysing various parameters
parameter_titles = ("Dirac voltage", "V_Dirac extrapolation overshoot",
                    "Max resistivity", "Carrier density FMHM",
                    "Max mobility (p)", "Max mobility (n)", "3-point mobility",
                    "%-difference mu_3p vs. mu_max (p)",
                    "FWHM mobility (p)", "FWHM mobility (n)",
                    "Max-mobility/FWHM voltage difference (p)",
                    "Max-mobility/FWHM voltage difference (n)",
                    "%-difference mu_max vs mu_FWHM (p)",
                    "%-difference mu_max vs mu_FWHM (n)")
# The units of these parameters
parameter_units = ("V", "V", "Ω/□", "cm^-2", "cm^2 V^-1 s^-1",
                   "cm^2 V^-1 s^-1", "cm^2 V^-1 s^-1", "%", "cm^2 V^-1 s^-1",
                   "cm^2 V^-1 s^-1", "V", "V", "%", "%")
# The parameters themselves, scaled where necessary to match units
parameters = ((dark1_dirac_V, light1_dirac_V, dark2_dirac_V, light2_dirac_V),
              (dark1_dirac_V_overshoots, light1_dirac_V_overshoots,
               dark2_dirac_V_overshoots, light2_dirac_V_overshoots),
              (dark1_rho_max, light1_rho_max, dark2_rho_max, light2_rho_max),
              (1e-4*dark1_delta_n, 1e-4*light1_delta_n, 1e-4*dark2_delta_n,
               1e-4*light2_delta_n),
              (1e4*dark1_mu_max_p, 1e4*light1_mu_max_p, 1e4*dark2_mu_max_p,
               1e4*light2_mu_max_p),
              (1e4*dark1_mu_max_n, 1e4*light1_mu_max_n, 1e4*dark2_mu_max_n,
               1e4*light2_mu_max_n),
              (1e4*dark1_mu_3p, 1e4*light1_mu_3p, 1e4*dark2_mu_3p,
               1e4*light2_mu_3p),
              (100*dark1_mu_3p_diffs, 100*light1_mu_3p_diffs,
               100*dark2_mu_3p_diffs, 100*light2_mu_3p_diffs),
              (1e4*dark1_mu_FWHM_p, 1e4*light1_mu_FWHM_p, 1e4*dark2_mu_FWHM_p,
               1e4*light2_mu_FWHM_p),
              (1e4*dark1_mu_FWHM_n, 1e4*light1_mu_FWHM_n),
              (dark1_V_diff_p, light1_V_diff_p, dark2_V_diff_p, light2_V_diff_p),
              (dark1_V_diff_n, light1_V_diff_n),
              (100*dark1_mu_FWHM_p_diffs, 100*light1_mu_FWHM_p_diffs,
               100*dark2_mu_FWHM_p_diffs, 100*light2_mu_FWHM_p_diffs),
              (100*dark1_mu_FWHM_n_diffs, 100*light1_mu_FWHM_n_diffs))
setups = ("Device 1 (dark conditions)", "Device 1 (light conditions)",
          "Device 2 (dark conditions)", "Device 2 (light conditions)")
selectors = (dark1_prelong, light1_prelong, dark2_prelong, light2_prelong)
for title, unit, param_set in zip(parameter_titles, parameter_units, parameters):
    print()
    print(f"Changes in average measured {title} due to the long experiments:")
    for param, setup, selector in zip(param_set, setups, selectors):
        before = np.nanmean(param[selector])
        before_err = np.nanstd(param[selector])
        after = np.nanmean(param[~selector])
        after_err = np.nanstd(param[~selector])
        total = np.nanmean(param)
        total_err = np.nanstd(param)
        print(f"{setup}: {before:.2e} ± {before_err:.2e} -> " +
              f"{after:.2e} ± {after_err:.2e} {unit} ")
        print(f"(Overall: {total:.2e} ± {total_err:.2e} {unit})")

"""
Objective 2:
Create graphs showing conductivity and resistivity traces to provide
illustration of some of the physical processes.
"""

# Plots showing the shift in the conductivity trace due to light exposure
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6))
fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
files = ("data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max.dat",
         "data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max.dat",
         "data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max-2.dat",
         "data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max-2.dat",
         "data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max.dat",
         "data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max.dat",
         "data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max-2.dat",
         "data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max-2.dat")

for filename in files:
    device = filename[26]  # strings '1' or '2'
    conditions = "Dark" if filename[36] == "D" else "Light"
    Vg_data, Isd_data = np.loadtxt(filename, usecols=(0, 3), unpack=True)
    datapoints_per_pass = int((max(Vg_data) - min(Vg_data)) / dVg)
    num_passes = int(len(Vg_data) / datapoints_per_pass)
    # Extract data from the final forward Vg sweep
    start_index = (num_passes - 2) * datapoints_per_pass
    stop_index = (num_passes - 1) * datapoints_per_pass
    Vg = Vg_data[start_index: stop_index]
    Isd = Isd_data[start_index: stop_index]
    sigma = Isd / Vsd
    ax = ax1 if device == '1' else ax2
    ax.plot(Vg, sigma, color='C0' if conditions == 'Dark' else 'C3',
            linestyle=':' if filename[-5] == '2' else '-',
            label=f'{conditions} conditions' if filename[-5] != '2' else None)
# ax1.set_title("Device 1 - Functionalised with Quantum Dots")
# ax2.set_title("Device 2 - Functionalised with Perovskites")
for ax in (ax1, ax2):
    ax.legend()
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Conductivity, $\\sigma$ ($\\Omega^{-1}$□)")
    ax.ticklabel_format(scilimits=[-4, 6], useMathText=True)
fig1.tight_layout()
fig2.tight_layout()

# Plots showing linear extraplotion from the point with maximum mobility,
# and methods for determining mobilities.
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
files = ("data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-post-long.dat",
         "data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-post-long.dat")
for filename in files:
    device = filename[32]  # strings '1' or '2'
    Vg_data, Isd_data = np.loadtxt(filename, usecols=(0, 3), unpack=True)
    datapoints_per_pass = int((max(Vg_data) - min(Vg_data)) / dVg)
    num_passes = int(len(Vg_data) / datapoints_per_pass)
    # Extract data from the final forward Vg sweep
    start_index = (num_passes - 2) * datapoints_per_pass
    stop_index = (num_passes - 1) * datapoints_per_pass
    Vg = Vg_data[start_index: stop_index]
    Isd = Isd_data[start_index: stop_index]
    sigma = Isd / Vsd
    axes = (ax1a, ax1b) if device == '1' else (ax2a, ax2b)
    axes[0].plot(Vg, sigma)
    V_dirac = Vg[np.argmin(sigma)]
    gradient = np.gradient(sigma, Vg)
    gradient_pad = np.pad(gradient, (width-1)//2, mode='edge')
    gradient_smooth = np.convolve(gradient_pad, top_hat(width), mode='valid')
    axes[1].plot(Vg, gradient, 'k:', label='$d\\sigma/dV_g$')
    axes[1].plot(Vg, gradient_smooth, label='$d\\sigma/dV_g$, smoothed')
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
    axes[0].plot([a1, a2], [b1, b2], 'X', color='C3',
                 label='Max gradient points')
    axes[1].plot([a1, a2], [m1, m2], 'X', color='C3',
                 label='Max gradient points')
    grad_line_p1 = a1 - 5
    grad_line_p2 = min(a2 + 5, Vg[-1])
    axes[0].plot([grad_line_p1, -c1/m1], [m1*grad_line_p1+c1, 0], 'k:')
    axes[0].plot([grad_line_p2, -c2/m2], [m2*grad_line_p2+c2, 0], 'k:',
                 label='Tangent lines')
# fig1.suptitle("Device 1 - Functionalised with Quantum Dots")
# fig2.suptitle("Device 2 - Functionalised with Perovskites")
for ax in (ax1a, ax2a):
    ax.legend()
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Conductivity, $\\sigma$ ($\\Omega^{-1}$□)")
    ax.ticklabel_format(scilimits=[-4, 6], useMathText=True)
for ax in (ax1b, ax2b):
    ax.legend()
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Gradient of conductivity, $d\\sigma/dV_g$ ($\\Omega^{-1}$□V$^{-1}$)")
    ax.ticklabel_format(useMathText=True)
fig1.tight_layout()
fig2.tight_layout()


"""
Objective 3:
Create a graph displaying the results for the pristine graphene samples,
and demonstrate that the linear region of the graph was not reached.
Also display results for the samples immediately after functionalisation.
Estimate the field effect mobility and Dirac voltage for each data set.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
files = ("data/cvd_prefunc_12032024/CVD1-Isd-Vg.dat",
         "data/cvd_prefunc_12032024/CVD2-Isd-Vg.dat",
         "data/cvd_func_15032024/CVD1F-Isd-Vg-Dark.dat",
         "data/cvd_func_15032024/CVD2F-Isd-Vg-Dark.dat")
labels = ("Pristine Device 1", "Pristine Device 2", "Device 1 w/ QDs", "Device 2 w/ NCs")
colours = ("C0", "C3", "C9", "C6")
print()
print("Early measurment estimates of μ_FE and V_Dirac for the graphene samples:")
for filename, label, colour, in zip(files, labels, colours):
    Vg_data, Isd_data = np.loadtxt(filename, usecols=(0, 3), unpack=True)
    datapoints_per_pass = int((max(Vg_data) - min(Vg_data)) / dVg)
    num_passes = int(len(Vg_data) / datapoints_per_pass)
    # Extract data from the final forward Vg sweep
    start_index = (num_passes - 2) * datapoints_per_pass
    stop_index = (num_passes - 1) * datapoints_per_pass
    Vg = Vg_data[start_index: stop_index]
    Isd = Isd_data[start_index: stop_index]
    sigma = Isd / Vsd
    ax1.plot(Vg, sigma, label=label, color=colour)
    gradient = np.gradient(sigma, Vg)
    ax2.plot(Vg, gradient, label=label, color=colour)
    min_grad = min(gradient)
    min_grad_index = np.argmin(gradient)
    mu = d/(epsilon_0 * epsilon_r) * abs(min_grad)
    V_dirac_extrap = Vg[min_grad_index] - sigma[min_grad_index] / min_grad
    print(f"For {label}, μ_FE >= {mu*1e4:.0f} cm^2 V^-1 s^-1, " +
          f"and V_Dirac <= {V_dirac_extrap:.0f} V")
    print("This V_Dirac corresponds to a carrier concentration of " +
          f"{1e-4*V_dirac_extrap * (epsilon_0 * epsilon_r) / (e * d):.2e} cm^-2")
    ax1.plot([Vg[min_grad_index], V_dirac_extrap],
             [sigma[min_grad_index], 0], ':', color=colour)
for ax in (ax1, ax2):
    ax.legend()
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.ticklabel_format(scilimits=[-3, 6], useMathText=True)
ax1.set_ylabel("Conductivity, $\\sigma$ ($\\Omega^{-1}$□)")
ax2.set_ylabel("Gradient of conductivity, $d\\sigma/dV_g$ ($\\Omega^{-1}$□V$^{-1}$)")
fig.tight_layout()

"""
Objective 4:
Create a graph displaying the results for the functionalised OFET samples.
Estimate the field effect mobility and carrier concentration.
"""

# Physical dimensions of the OFET chip are different than the GFET
d = 230e-9  # dielectric thickness (m)
L = 10e-6  # Length of the channel (parallel to current flow) (m)
dL = 0.5e-6  # Length uncertainty
W = 9730e-6  # Width of the channel (perpendicular to current flow) (m)
dW = 10e-6  # Width uncertainty
Vsd = 5  # Source-drain voltage (V)

fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6))
fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
files = ("data/ofet3_16022024/OFET3F-Isd-Vg-Dark.dat",
         "data/ofet3_16022024/OFET3F-Isd-Vg-Light.dat",
         "data/ofet4_23022024/OFET4F-Isd-Vg-Dark.dat",
         "data/ofet4_23022024/OFET4F-Isd-Vg-Light.dat")
print()
print("Estimates of μ_FE and V_Dirac for the functionalised OFET samples:")
for filename in files:
    conditions = filename[34:-4]  # string 'Dark' or 'Light'
    device = filename[24]  # string '3' or '4'
    ax = ax2 if device == '3' else ax1
    Vg, Isd = np.loadtxt(filename, usecols=(0, 3), unpack=True)
    sigma = L / W * Isd / Vsd
    ax.plot(Vg, sigma, label=f'{conditions} conditions')
    fitting_Vg = Vg[len(Vg)//4: len(Vg)//2]
    fitting_sigma = sigma[len(sigma)//4: len(sigma)//2]
    (m, c), cv = curve_fit(linear_fit, fitting_Vg, fitting_sigma)
    mu = d/(epsilon_0 * epsilon_r) * abs(m)
    mu_err = mu * np.sqrt((dL/L)**2 + (dW/W)**2 + (cv[0, 0]/m)**2)
    p0 = -c/m * (epsilon_0 * epsilon_r) / (e * d)
    p0_err = p0 * np.sqrt((dL/L)**2 + (dW/W)**2 + (cv[0, 0]/m)**2 + (cv[1, 1]/c)**2)
    print(f"For OFET{device} in {conditions.lower()} conditions, " +
          f"μ_FE = {mu*1e4:.2e} ± {mu_err*1e4:.2e} cm^2 V^-1 s^-1, " +
          f"and p_0 = {p0*1e-4:.2e} ± {p0_err*1e-4:.2e} cm^-2")
for ax in (ax1, ax2):
    ax.legend()
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Conductivity, $\\sigma$ ($\\Omega^{-1}$□)")
    ax.ticklabel_format(useMathText=True)
# ax1.set_title("OFET functionalised with Quantum Dots")
# ax2.set_title("OFET functionalised with Perovskites")
fig1.tight_layout()
fig2.tight_layout()

plt.show()
