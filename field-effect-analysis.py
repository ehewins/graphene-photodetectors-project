import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


"""
Objective 1:
Analyse how properties of the functionalised devices have changed between
measurements, including the location of the Dirac point (for the earlier
measurements, this must be estimated from the gradient), and mobility (").
"""

filenames = (
    "data/cvd_func_15032024/CVD1F-Isd-Vg-Dark.dat",
    "data/cvd_func_15032024/CVD1F-Isd-Vg-Light.dat",
    "data/cvd_func_15032024/CVD2F-Isd-Vg-Dark.dat",
    "data/cvd_func_15032024/CVD2F-Isd-Vg-Light.dat",
    "data/cvd_func_19032024/CVD1F-Isd-Vg-Dark.dat",
    "data/cvd_func_19032024/CVD1F-Isd-Vg-Light.dat",
    # "data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max.dat",
    # "data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max.dat",
    "data/cvd_func_19032024/CVD1F-Isd-Vg-Dark-25max-2.dat",
    "data/cvd_func_19032024/CVD1F-Isd-Vg-Light-25max-2.dat",
    "data/cvd_func_19032024/CVD2F-Isd-Vg-Dark.dat",
    "data/cvd_func_19032024/CVD2F-Isd-Vg-Light.dat",
    "data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max.dat",
    "data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max.dat",
    "data/cvd_func_19032024/CVD2F-Isd-Vg-Dark-35max-2.dat",
    "data/cvd_func_19032024/CVD2F-Isd-Vg-Light-35max-2.dat",
    "data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-post-long.dat",
    "data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-post-long.dat",
    "data/cvd_time_power_22032024/CVD1F-Isd-Vg-Dark-final.dat",
    "data/cvd_time_power_22032024/CVD1F-Isd-Vg-Light-final.dat",
    "data/cvd_time_power_22032024/CVD2F-Isd-Vg-Dark-final.dat",
    "data/cvd_time_power_22032024/CVD2F-Isd-Vg-Light-final.dat",
    "data/cvd_high_current_26032024/CVD1F-Isd-Vg-Dark-final.dat",
    "data/cvd_high_current_26032024/CVD1F-Isd-Vg-Light-final.dat",
    "data/cvd_high_current_26032024/CVD2F-Isd-Vg-Dark-final.dat",
    "data/cvd_high_current_26032024/CVD2F-Isd-Vg-Light-final.dat",
)

dVg = 0.2  # Gate voltage step resolution (V)

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
for filename in filenames:
    device = int(filename.split("/")[2][3])
    conditions = "Dark" if filename.split("/")[2][13] == "D" else "Light"
    date = filename.split("/")[1][-8:-6] + "/" + filename.split("/")[1][-6:-4]
    Vg, Isd = np.loadtxt(filename, usecols=(0, 3), unpack=True)
    datapoints_per_pass = int((max(Vg) - min(Vg)) / dVg)
    num_passes = int(len(Vg) / datapoints_per_pass)
    dirac_voltages = []
    for p_num in range(0, num_passes, 2):  # only study the forward passes
        start_index = p_num * datapoints_per_pass
        stop_index = (p_num + 1) * datapoints_per_pass
        # sub-arrays for Vg and Isd in this pass
        p_Vg = Vg[start_index: stop_index]
        p_Isd = Isd[start_index: stop_index]
        V_dirac_visible = p_Isd[-1] != min(p_Isd)
        if V_dirac_visible:
            dirac_voltages.append(p_Vg[np.argmin(p_Isd)])
    # Test convergence ratios on Device 2, which is much more predictable.
    dirac_voltages = np.asarray(dirac_voltages)
    if len(dirac_voltages) == 0:
        continue
    match (device, conditions):
        case (1, "Dark"): ax = ax1a
        case (1, "Light"): ax = ax1b
        case (2, "Dark"): ax = ax2a
        case (2, "Light"): ax = ax2b
    ax.plot(range(1, len(dirac_voltages)+1), dirac_voltages, '-o', label=date)
    ax.set_xticks(range(1, 5))
    ax.set_title(f"Device {device}, {conditions.lower()} conditions")
    ax.set_xlabel("Gate voltage sweep no.")
    ax.set_ylabel("Dirac voltage, $V_{Dirac}$ (V)")
    ax.legend(loc='lower right')

plt.show()
