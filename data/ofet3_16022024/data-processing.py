import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


# Plot the time series data where the device as suddenly illuminated
current, time = np.loadtxt("OFET3F-TimeExperiment-Vg=0,Vsd=5V.dat",
                           delimiter='\t', usecols=(1, 4), unpack=True)
fig, ax = plt.subplots()
ax.plot(time[:163], current[:163])
ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
ax.set_title("$I_{sd}(t)$ with $V_g = 0$ V and $V_{sd} = 5$ V.")
ax.annotate("Dark Conditions", (4, 5.415e-10))
ax.annotate("405 nm Illumination", (22.5, 5.415e-10))

# Creating comparitive plots for with/without perovskites & with/without light

# If the third column is 1, we only use the data from the second pass
# (an anomaly in one of these datasets messes up the axis scaling).
comparisons = (("OFET3-Isd-Vg.dat", "OFET3F-Isd-Vg-Dark.dat", 0),
               ("OFET3-Isd-Vsd.dat", "OFET3F-Isd-Vsd-Dark.dat", 0),
               ("OFET3F-Isd-Vg-Dark.dat", "OFET3F-Isd-Vg-Light.dat", 0),
               ("OFET3F-Isd-Vsd-Dark.dat", "OFET3F-Isd-Vsd-Light.dat", 1))

for files in comparisons:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    data1 = np.loadtxt(files[0], delimiter='\t')
    data2 = np.loadtxt(files[1], delimiter='\t')
    if files[2] == 1:
        data1 = data1[len(data1)//2:, :]
        data2 = data2[len(data2)//2:, :]
    base = files[0].split(".", maxsplit=1)[0]
    depend_var = base.split("-")[2]
    data1_x = data1[:, 0]
    data2_x = data2[:, 0]
    if depend_var == "Vg":
        data1_ya = data1[:, 3]
        data2_ya = data2[:, 3]
        data1_yb = data1[:, 1]
        data2_yb = data2[:, 1]
        ax_a.set_xlabel("Gate voltage, $V_g$ (V)")
        ax_b.set_xlabel("Gate voltage, $V_g$ (V)")
        title_part_2 = " with $V_{sd} = 5$ V"
    else:
        data1_ya = data1[:, 1]
        data2_ya = data2[:, 1]
        data1_yb = data1[:, 3]
        data2_yb = data2[:, 3]
        ax_a.set_xlabel("Source-drain voltage, $V_{sd}$ (V)")
        ax_b.set_xlabel("Source-drain voltage, $V_{sd}$ (V)")
        title_part_2 = " with $V_g = 5$ V"
    ax_a.set_ylabel("Source-Drain Current, $I_{sd}$ (A)")
    ax_b.set_ylabel("Gate current, $I_g$ (A)")
    ax_a.plot(data1_x, data1_ya)
    ax_a.plot(data2_x, data2_ya)
    ax_b.plot(data1_x, data1_yb)
    ax_b.plot(data2_x, data2_yb)
    if base.split("-")[-1] != "Dark":
        ax_a.legend(["Pre-functionalisation", "Post-functionalisation"])
        ax_b.legend(["Pre-functionalisation", "Post-functionalisation"])
        title_part_1 = "Before and after functionalisation"
    else:
        ax_a.legend(["Dark", "Light"])
        ax_b.legend(["Dark", "Light"])
        title_part_1 = "Before and after illumination (405 nm)"
    plt.suptitle(title_part_1 + title_part_2)
    plt.tight_layout()

"""
Find the source-drain resistance from the gradient of the linear regions of the
I_{sd}(V_{sd}) graphs, for (a) - pre-functionalisation, (b), functionalised in
dark conditions, & (c) functionalised in light conditions.
"""

# Resistance pre-functionalisation
Vsd, Isd = np.loadtxt("OFET3-Isd-Vsd.dat", delimiter='\t', usecols=(0, 1),
                      unpack=True)
Vsd_pass1 = Vsd[:len(Vsd)//2]
Vsd_pass2 = Vsd[len(Vsd)//2:]
Isd_pass1 = Isd[:len(Isd)//2]
Isd_pass2 = Isd[len(Isd)//2:]
linear_start_V1 = 4
linear_start_V2 = 16
linear_region1 = Vsd_pass1 >= linear_start_V1
linear_region2 = Vsd_pass2 <= linear_start_V2

(m1, c1), cv1 = curve_fit(linear_fit, Vsd_pass1[linear_region1],
                          Isd_pass1[linear_region1])
(m2, c2), cv2 = curve_fit(linear_fit, Vsd_pass2[linear_region2],
                          Isd_pass2[linear_region2])
R1, err1 = 1/m1, 1/m1 * np.sqrt(cv1[0, 0]) / m1
R2, err2 = 2/m2, 2/m2 * np.sqrt(cv2[0, 0]) / m2
R_avg = (R1 + R2) / 2
err_avg = np.sqrt(err1**2 + err2**2) / 2

print("Pre functionalisation, for source-drain measurements:")
print(f"The resistance during the forward pass is {R1:e} ± {err1:e} Ohms")
print(f"The resistance during the backard pass is {R2:e} ± {err2:e} Ohms")
print(f"The average resistance is {R_avg:e} ± {err_avg:e} Ohms")

# Resistance post-functionalisation, dark conditions
Vsd, Isd = np.loadtxt("OFET3F-Isd-Vsd-Dark.dat", delimiter='\t',
                      usecols=(0, 1), unpack=True)
Vsd_pass1 = Vsd[:len(Vsd)//4]
Vsd_pass2 = Vsd[len(Vsd)//4:len(Vsd)//2]
Vsd_pass3 = Vsd[len(Vsd)//2:3*len(Vsd)//4]
Vsd_pass4 = Vsd[3*len(Vsd)//4:]
Isd_pass1 = Isd[:len(Isd)//4]
Isd_pass2 = Isd[len(Isd)//4:len(Isd)//2]
Isd_pass3 = Isd[len(Isd)//2:3*len(Isd)//4]
Isd_pass4 = Isd[3*len(Isd)//4:]

# This code block was used to inform the decision of where should be defined as
# the linear region.

# gradient1 = np.gradient(Isd_pass1, Vsd_pass1)
# gradient2 = np.gradient(Isd_pass2, Vsd_pass2)
# gradient3 = np.gradient(Isd_pass3, Vsd_pass3)
# gradient4 = np.gradient(Isd_pass4, Vsd_pass4)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
# ax1.plot(Vsd_pass1, Isd_pass1)
# ax1.plot(Vsd_pass2, Isd_pass2)
# ax1.plot(Vsd_pass3, Isd_pass3)
# ax1.plot(Vsd_pass4, Isd_pass4)
# ax2.plot(Vsd_pass1, gradient1)
# ax2.plot(Vsd_pass2, gradient2)
# ax2.plot(Vsd_pass3, gradient3)
# ax2.plot(Vsd_pass4, gradient4)
# ax3.plot(Vsd_pass1, np.gradient(gradient1, Vsd_pass1))
# ax3.plot(Vsd_pass2, np.gradient(gradient2, Vsd_pass2))
# ax3.plot(Vsd_pass3, np.gradient(gradient3, Vsd_pass3))
# ax3.plot(Vsd_pass4, np.gradient(gradient4, Vsd_pass4))
# for ax in (ax1, ax2, ax3):
#     ax.set_xlabel("$V_{sd}$ (V)")
# ax1.set_ylabel("$I_{sd}$ (A)")
# ax2.set_ylabel("$dI_{sd} \\ / \\ dV_{sd}$ (AV$^{-1}$)")
# ax3.set_ylabel("$d^2I_{sd} \\ / \\ dV_{sd}^2$ (AV$^{-2}$)")
# fig.tight_layout()
# fig.suptitle("Dark conditions")

linear_region_start_V = -8.8
linear_region_stop_V = -3.5
linear_region1 = (Vsd_pass1 >= linear_region_start_V) & \
    (Vsd_pass1 <= linear_region_stop_V)
linear_region2 = (Vsd_pass2 >= linear_region_start_V) & \
    (Vsd_pass2 <= linear_region_stop_V)
linear_region3 = (Vsd_pass3 >= linear_region_start_V) & \
    (Vsd_pass3 <= linear_region_stop_V)
linear_region4 = (Vsd_pass4 >= linear_region_start_V) & \
    (Vsd_pass4 <= linear_region_stop_V)
(m1, c1), cv1 = curve_fit(linear_fit, Vsd_pass1[linear_region1],
                          Isd_pass1[linear_region1])
(m2, c2), cv2 = curve_fit(linear_fit, Vsd_pass2[linear_region2],
                          Isd_pass2[linear_region2])
(m3, c3), cv3 = curve_fit(linear_fit, Vsd_pass3[linear_region3],
                          Isd_pass3[linear_region3])
(m4, c4), cv4 = curve_fit(linear_fit, Vsd_pass4[linear_region4],
                          Isd_pass4[linear_region4])
R1, err1 = 1/m1, 1/m1 * np.sqrt(cv1[0, 0]) / m1
R2, err2 = 2/m2, 2/m2 * np.sqrt(cv2[0, 0]) / m2
R3, err3 = 3/m3, 3/m3 * np.sqrt(cv3[0, 0]) / m3
R4, err4 = 4/m4, 4/m4 * np.sqrt(cv4[0, 0]) / m4
R_fwd_avg = (R1 + R3) / 2
R_bwd_avg = (R2 + R4) / 2
err_fwd_avg = np.sqrt(err1**2 + err3**2) / 2
err_bwd_avg = np.sqrt(err2**2 + err4**2) / 2
R_avg = (R1 + R2 + R3 + R4) / 4
err_avg = np.sqrt(err1**2 + err2**2 + err3**2 + err4**2) / 4
print()
print("Post functionalisation (in dark conditions), for source-drain measurements (V_g = 5 V):")
print(f"The resistance during the first forward pass is {R1:e} ± {err1:e} Ohms")
print(f"The resistance during the first backard pass is {R2:e} ± {err2:e} Ohms")
print(f"The resistance during the second forward pass is {R3:e} ± {err3:e} Ohms")
print(f"The resistance during the second backard pass is {R4:e} ± {err4:e} Ohms")
print(f"The average resistance during a forward pass is {R_fwd_avg:e} ± {err_fwd_avg:e} Ohms")
print(f"The average resistance during a backward pass is {R_bwd_avg:e} ± {err_bwd_avg:e} Ohms")
print(f"The overall average resistance is {R_avg:e} ± {err_avg:e} Ohms")

# Resistance post-functionalisation, light conditions
Vsd, Isd = np.loadtxt("OFET3F-Isd-Vsd-Light.dat", delimiter='\t',
                      usecols=(0, 1), unpack=True)
Vsd_pass1 = Vsd[:len(Vsd)//4]
Vsd_pass2 = Vsd[len(Vsd)//4:len(Vsd)//2]
Vsd_pass3 = Vsd[len(Vsd)//2:3*len(Vsd)//4]
Vsd_pass4 = Vsd[3*len(Vsd)//4:]
Isd_pass1 = Isd[:len(Isd)//4]
Isd_pass2 = Isd[len(Isd)//4:len(Isd)//2]
Isd_pass3 = Isd[len(Isd)//2:3*len(Isd)//4]
Isd_pass4 = Isd[3*len(Isd)//4:]

# This code block was used to inform the decision of where should be defined as
# the linear region.

# gradient1 = np.gradient(Isd_pass1, Vsd_pass1)
# gradient2 = np.gradient(Isd_pass2, Vsd_pass2)
# gradient3 = np.gradient(Isd_pass3, Vsd_pass3)
# gradient4 = np.gradient(Isd_pass4, Vsd_pass4)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
# ax1.plot(Vsd_pass1, Isd_pass1)
# ax1.plot(Vsd_pass2, Isd_pass2)
# ax1.plot(Vsd_pass3, Isd_pass3)
# ax1.plot(Vsd_pass4, Isd_pass4)
# ax2.plot(Vsd_pass1, gradient1)
# ax2.plot(Vsd_pass2, gradient2)
# ax2.plot(Vsd_pass3, gradient3)
# ax2.plot(Vsd_pass4, gradient4)
# ax3.plot(Vsd_pass1, np.gradient(gradient1, Vsd_pass1))
# ax3.plot(Vsd_pass2, np.gradient(gradient2, Vsd_pass2))
# ax3.plot(Vsd_pass3, np.gradient(gradient3, Vsd_pass3))
# ax3.plot(Vsd_pass4, np.gradient(gradient4, Vsd_pass4))
# for ax in (ax1, ax2, ax3):
#     ax.set_xlabel("$V_{sd}$ (V)")
# ax1.set_ylabel("$I_{sd}$ (A)")
# ax1.set_ylim(-1.3e-8, 1.3e-8)
# ax2.set_ylabel("$dI_{sd} \\ / \\ dV_{sd}$ (AV$^{-1}$)")
# ax2.set_ylim(1e-10, 2.8e-9)
# ax3.set_ylabel("$d^2I_{sd} \\ / \\ dV_{sd}^2$ (AV$^{-2}$)")
# ax3.set_ylim(-3.7e-9, 3.8e-9)
# fig.tight_layout()
# fig.suptitle("Illuminated conditions")

linear_region_start_V = -7.8
linear_region_stop_V = -3.6
linear_region1 = (Vsd_pass1 >= linear_region_start_V) & \
    (Vsd_pass1 <= linear_region_stop_V)
linear_region4 = (Vsd_pass4 >= linear_region_start_V) & \
    (Vsd_pass4 <= linear_region_stop_V)
(m1, c1), cv1 = curve_fit(linear_fit, Vsd_pass1[linear_region1],
                          Isd_pass1[linear_region1])
(m4, c4), cv4 = curve_fit(linear_fit, Vsd_pass4[linear_region4],
                          Isd_pass4[linear_region4])
R1, err1 = 1/m1, 1/m1 * np.sqrt(cv1[0, 0]) / m1
R4, err4 = 4/m4, 4/m4 * np.sqrt(cv4[0, 0]) / m4
R_avg = (R1 + R4) / 2
err_avg = np.sqrt(err1**2 + err4**2) / 2
print()
print("Post functionalisation (in illuminated conditions), for source-drain measurements:")
print(f"The resistance during the first forward pass is {R1:e} ± {err1:e} Ohms")
print(f"The resistance during the second backard pass is {R4:e} ± {err4:e} Ohms")
print(f"The average resistance is {R_avg:e} ± {err_avg:e} Ohms")

"""
Calculate the photocurrent using the measurements with V_g as the dependent
variable. Also calculate the field effect mobility and doping levels from the
gradient of the linear regions.
"""

Vg, Isd_dark = np.loadtxt("OFET3F-Isd-Vg-Dark.dat", delimiter='\t',
                          usecols=(0, 3), unpack=True)
Isd_light = np.loadtxt("OFET3F-Isd-Vg-Light.dat", delimiter='\t',
                       usecols=(3), unpack=True)
Vg1 = Vg[:len(Vg)//2]
Vg2 = Vg[len(Vg)//2:]
Isd_dark1 = Isd_dark[:len(Isd_dark)//2]
Isd_dark2 = Isd_dark[len(Isd_dark)//2:]
Isd_light1 = Isd_light[:len(Isd_light)//2]
Isd_light2 = Isd_light[len(Isd_light)//2:]
Iph1 = Isd_light1 - Isd_dark1
Iph2 = Isd_light2 - Isd_dark2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(Vg1, Isd_light1, '--', color='C0', label='Illuminated conditions, forward pass')
ax1.plot(Vg2, Isd_light2, '--', color='C1', label='Illuminated conditions, backward pass')
ax1.plot(Vg1, Isd_dark1, '-', color='C0', label='Illuminated conditions, forward pass')
ax1.plot(Vg2, Isd_dark2, '-', color='C1', label='Dark conditions, backward pass')
ax2.plot(Vg1, Iph1, label='Forward pass')
ax2.plot(Vg2, Iph2, label='Backward pass')
ax1.set_xlabel("Gate voltage, $V_g$ (V)")
ax1.set_ylabel("Source-drain current, $I_{sd}$ (A)")
ax2.set_xlabel("Gate voltage, $V_g$ (V)")
ax2.set_ylabel("Photocurrent, $I_{ph}$ (A)")
ax1.legend()
ax2.legend()
plt.tight_layout()

print()
print("From the measurements of source-drain current against gate voltage, with a souce-drain voltage of 5 V:")
print(f"During the backward pass, the average photocurrent was {np.mean(Iph2):e} ± {np.std(Iph2):e} A")

# Determine linear region based on the latter half of the fwd. and bwd. passes
(m_dark1, c_dark1), cv_dark1 = curve_fit(linear_fit, Vg1[len(Vg1)//2:],
                                         Isd_dark1[len(Isd_dark1)//2:])
(m_dark2, c_dark2), cv_dark2 = curve_fit(linear_fit, Vg2[len(Vg2)//2:],
                                         Isd_dark2[len(Isd_dark2)//2:])
(m_light1, c_light1), cv_light1 = curve_fit(linear_fit, Vg1[len(Vg1)//2:],
                                            Isd_light1[len(Isd_light1)//2:])
(m_light2, c_light2), cv_light2 = curve_fit(linear_fit, Vg2[len(Vg2)//2:],
                                            Isd_light2[len(Isd_light2)//2:])
# Calculate averages from forward & backward passes
m_dark = (m_dark1 + m_dark2) / 2
c_dark = (c_dark1 + c_dark2) / 2
errs_dark = np.sqrt(np.diag(cv_dark1 + cv_dark2))
m_light = (m_light1 + m_light2) / 2
c_light = (c_light1 + c_light2) / 2
errs_light = np.sqrt(np.diag(cv_light1 + cv_light2))
# relative uncertainties in the gradients and intercepts
rel_dm_dark = errs_dark[0] / m_dark
rel_dc_dark = errs_dark[1] / c_dark
rel_dm_light = errs_light[0] / m_light
rel_dc_light = errs_light[1] / c_light

# Calculate field effect mobility and doping
L = 1e-5  # channel length (m)
dL = 0.05e-5  # uncertainty in channel length (m)
rel_dL = dL / L
W = 9.73e-3  # channel width (m)
dW = 0.01e-3  # uncertainty in channel width (m)
rel_dW = dW / W
d = 230e-9  # dielectric thickness (m)
epsilon_r = 3.9  # relative permittivity of dielectric
epsilon_0 = 8.85e-12  # permittivity of free space (F/m)
V_ds = 5  # source-drain voltage (V)
e = 1.6e-19  # charge magnitude of a single carrier (C)
# calculations of the properties themselves
mu_dark = m_dark * L * d / (W * V_ds * epsilon_0 * epsilon_r)
mu_light = m_light * L * d / (W * V_ds * epsilon_0 * epsilon_r)
p0_dark = - c_dark / m_dark * (epsilon_0 * epsilon_r) / (e * d)
p0_light = - c_light / m_light * (epsilon_0 * epsilon_r) / (e * d)
# calculation of the uncertainty in these properties
rel_dmu_dark = np.sqrt(rel_dm_dark**2 + rel_dL**2 + rel_dW**2)
dmu_dark = rel_dmu_dark * mu_dark
rel_dmu_light = np.sqrt(rel_dm_light**2 + rel_dL**2 + rel_dW**2)
dmu_light = rel_dmu_light * mu_light
rel_dp0_dark = np.sqrt(rel_dc_dark**2 + rel_dm_dark**2)
dp0_dark = rel_dp0_dark * p0_dark
rel_dp0_light = np.sqrt(rel_dc_light**2 + rel_dm_light**2)
dp0_light = rel_dp0_light * p0_light
print("The field effect mobility in dark conditions is:")
print(f"({mu_dark*1e4:e} ± {dmu_dark*1e4:e}) cm^2 V^-1 s^-1")
print("The carrier density in dark conditions is:")
print(f"({p0_dark*1e-4:e} ± {dp0_dark*1e-4:e}) cm^-2")
print("The field effect mobility in light conditions is:")
print(f"({mu_light*1e4:e} ± {dmu_light*1e4:e}) cm^2 V^-1 s^-1")
print("The carrier density in light conditions is:")
print(f"({p0_light*1e-4:e} ± {dp0_light*1e-4:e}) cm^-2")

# Graphing the same results as above, but scaling the axes so the plot is of
# conductivity against carrier concentration.
# TODO: Come back and do something about the negative value of p
sigma_dark = Isd_dark / V_ds * L / W
sigma_light = Isd_light / V_ds * L / W
p = (epsilon_0 * epsilon_r) / (e * d) * Vg
plt.figure()
plt.plot(p*1e-4, sigma_dark*1e4, label='Dark conditions')
plt.plot(p*1e-4, sigma_light*1e4, label='Illuminated conditions')
plt.xlabel("Carrier concentration, $p$, (cm$^{-2}$)")
plt.ylabel("Conductivity, $\\sigma$ ($\\Omega^{-1}$cm$^2$)")
# plt.plot(p*1e-4, e*(p-p0_dark)*mu_dark*1e4)
# plt.plot(p*1e-4, e*(p-p0_light)*mu_light*1e4)
plt.legend()

"""
Calculate the photocurrent using the measurements with time as the dependent
variable.
"""

current, time = np.loadtxt("OFET3F-TimeExperiment-Vg=0,Vsd=5V.dat",
                           delimiter='\t', usecols=(1, 4), unpack=True)
# time = time[:163]
# current = current[:163]
dark_end = 18  # time in seconds
light_start = 20.5  # time in seconds
time_dark = time[time <= dark_end]
current_dark = current[time <= dark_end]
time_light = time[time >= light_start]
current_light = current[time >= light_start]

(m_dark, c_dark), cv_dark = curve_fit(linear_fit, time_dark, current_dark)
(m_light, c_light), cv_light = curve_fit(linear_fit, time_light, current_light)
dark_errs = np.sqrt(np.diag(cv_dark))
light_errs = np.sqrt(np.diag(cv_light))
print()
print("From the I_{sd} vs time measurements (V_g = 0 V, V_{sd}=5 V)")
print(f"Dark region gradient = {m_dark:e} ± {dark_errs[0]:e} A/s")
print(f"Light region gradient = {m_light:e} ± {light_errs[0]:e} A/s")
print(f"Photocurrent = {c_light - c_dark:e} ± {np.sqrt(np.sum(np.diag(cv_dark + cv_light)))} A")

plt.show()
