import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


# Plot the time series data where the device as suddenly illuminated
time, current = np.loadtxt("OFET4F-time.dat", delimiter='\t',
                           usecols=(1, 5), unpack=True)  # Might be col 3 not 5
fig, ax = plt.subplots()
ax.plot(time, current)
ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
ax.set_title("$I_{sd}(t)$ with $V_g = 0$ V and $V_{sd} = 5$ V.")
ax.annotate("Dark Conditions", (4, 5.415e-10))
ax.annotate("405 nm Illumination", (22.5, 5.415e-10))
# plt.savefig("OFET4F-time.png", dpi=500)

# Creating comparitive plots for with/without QDs & with/without light

comparisons = (("OFET4-Isd-Vg.dat", "OFET4F-Isd-Vg-Dark.dat"),
               ("OFET4-Isd-Vsd.dat", "OFET4F-Isd-Vsd-Dark.dat"),
               ("OFET4F-Isd-Vg-Dark.dat", "OFET4F-Isd-Vg-Light.dat"),
               ("OFET4F-Isd-Vsd-Dark.dat", "OFET4F-Isd-Vsd-Light.dat"))

for files in comparisons:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    data1 = np.loadtxt(files[0], delimiter='\t')
    data2 = np.loadtxt(files[1], delimiter='\t')
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
Vsd, Isd = np.loadtxt("OFET4-Isd-Vsd.dat", delimiter='\t', usecols=(0, 1),
                      unpack=True)
Vsd_pass1 = Vsd[:len(Vsd)//4]
Vsd_pass2 = Vsd[len(Vsd)//4:len(Vsd)//2]
Vsd_pass3 = Vsd[len(Vsd)//2:3*len(Vsd)//4]
Vsd_pass4 = Vsd[3*len(Vsd)//4:]
Isd_pass1 = Isd[:len(Isd)//4]
Isd_pass2 = Isd[len(Isd)//4:len(Isd)//2]
Isd_pass3 = Isd[len(Isd)//2:3*len(Isd)//4]
Isd_pass4 = Isd[3*len(Isd)//4:]

# Consider the linear region as the latter half of each fwd/bwd pass
(m1, c1), cv1 = curve_fit(linear_fit, Vsd_pass1[len(Vsd_pass1)//2:],
                          Isd_pass1[len(Isd_pass1)//2:])
(m2, c2), cv2 = curve_fit(linear_fit, Vsd_pass2[len(Vsd_pass2)//2:],
                          Isd_pass2[len(Isd_pass2)//2:])
(m3, c3), cv3 = curve_fit(linear_fit, Vsd_pass3[len(Vsd_pass3)//2:],
                          Isd_pass3[len(Isd_pass3)//2:])
(m4, c4), cv4 = curve_fit(linear_fit, Vsd_pass4[len(Vsd_pass4)//2:],
                          Isd_pass4[len(Isd_pass4)//2:])
R1, err1 = 1/m1, 1/m1 * np.sqrt(cv1[0, 0]) / m1
R2, err2 = 1/m2, 1/m2 * np.sqrt(cv2[0, 0]) / m2
R3, err3 = 1/m3, 1/m3 * np.sqrt(cv3[0, 0]) / m3
R4, err4 = 1/m4, 1/m4 * np.sqrt(cv4[0, 0]) / m4
R_fwd_avg = (R1 + R3) / 2
err_fwd_avg = np.sqrt(err1**2 + err3**2) / 2
R_bwd_avg = (R2 + R4) / 2
err_bwd_avg = np.sqrt(err2**2 + err4**2) / 2
R_avg = (R1 + R2 + R3 + R4) / 4
err_avg = np.sqrt(err1**2 + err2**2 + err3**2 + err4**2) / 4

print("Pre functionalisation, for source-drain measurements:")
print(f"The resistance during the first forward pass is {R1:e} ± {err1:e} Ohms")
print(f"The resistance during the first backward pass is {R2:e} ± {err2:e} Ohms")
print(f"The resistance during the second forward pass is {R3:e} ± {err3:e} Ohms")
print(f"The resistance during the second backward pass is {R4:e} ± {err4:e} Ohms")
print(f"The average resistance for the forward passes is {R_fwd_avg:e} ± {err_fwd_avg:e} Ohms")
print(f"The average resistance for the backward passes is {R_bwd_avg:e} ± {err_bwd_avg:e} Ohms")
print(f"The overall average resistance is {R_avg:e} ± {err_avg:e} Ohms")

# Resistance post-functionalisation, dark conditions

Vsd, Isd = np.loadtxt("OFET4F-Isd-Vsd-Dark.dat", delimiter='\t',
                      usecols=(0, 1), unpack=True)
Vsd_pass1 = Vsd[:len(Vsd)//4]
Vsd_pass2 = Vsd[len(Vsd)//4:len(Vsd)//2]
Vsd_pass3 = Vsd[len(Vsd)//2:3*len(Vsd)//4]
Vsd_pass4 = Vsd[3*len(Vsd)//4:]
Isd_pass1 = Isd[:len(Isd)//4]
Isd_pass2 = Isd[len(Isd)//4:len(Isd)//2]
Isd_pass3 = Isd[len(Isd)//2:3*len(Isd)//4]
Isd_pass4 = Isd[3*len(Isd)//4:]

# Consider the linear region as the latter half of each fwd/bwd pass
(m1, c1), cv1 = curve_fit(linear_fit, Vsd_pass1[len(Vsd_pass1)//2:],
                          Isd_pass1[len(Isd_pass1)//2:])
(m2, c2), cv2 = curve_fit(linear_fit, Vsd_pass2[len(Vsd_pass2)//2:],
                          Isd_pass2[len(Isd_pass2)//2:])
(m3, c3), cv3 = curve_fit(linear_fit, Vsd_pass3[len(Vsd_pass3)//2:],
                          Isd_pass3[len(Isd_pass3)//2:])
(m4, c4), cv4 = curve_fit(linear_fit, Vsd_pass4[len(Vsd_pass4)//2:],
                          Isd_pass4[len(Isd_pass4)//2:])
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
print(f"The resistance during the first backward pass is {R2:e} ± {err2:e} Ohms")
print(f"The resistance during the second forward pass is {R3:e} ± {err3:e} Ohms")
print(f"The resistance during the second backward pass is {R4:e} ± {err4:e} Ohms")
print(f"The average resistance during a forward pass is {R_fwd_avg:e} ± {err_fwd_avg:e} Ohms")
print(f"The average resistance during a backward pass is {R_bwd_avg:e} ± {err_bwd_avg:e} Ohms")
print(f"The overall average resistance is {R_avg:e} ± {err_avg:e} Ohms")

# Resistance post-functionalisation, illuminated conditions

Vsd, Isd = np.loadtxt("OFET4F-Isd-Vsd-Light.dat", delimiter='\t',
                      usecols=(0, 1), unpack=True)
Vsd_pass1 = Vsd[:len(Vsd)//4]
Vsd_pass2 = Vsd[len(Vsd)//4:len(Vsd)//2]
Vsd_pass3 = Vsd[len(Vsd)//2:3*len(Vsd)//4]
Vsd_pass4 = Vsd[3*len(Vsd)//4:]
Isd_pass1 = Isd[:len(Isd)//4]
Isd_pass2 = Isd[len(Isd)//4:len(Isd)//2]
Isd_pass3 = Isd[len(Isd)//2:3*len(Isd)//4]
Isd_pass4 = Isd[3*len(Isd)//4:]

# Consider the linear region as the latter half of each fwd/bwd pass
(m1, c1), cv1 = curve_fit(linear_fit, Vsd_pass1[len(Vsd_pass1)//2:],
                          Isd_pass1[len(Isd_pass1)//2:])
(m2, c2), cv2 = curve_fit(linear_fit, Vsd_pass2[len(Vsd_pass2)//2:],
                          Isd_pass2[len(Isd_pass2)//2:])
(m3, c3), cv3 = curve_fit(linear_fit, Vsd_pass3[len(Vsd_pass3)//2:],
                          Isd_pass3[len(Isd_pass3)//2:])
(m4, c4), cv4 = curve_fit(linear_fit, Vsd_pass4[len(Vsd_pass4)//2:],
                          Isd_pass4[len(Isd_pass4)//2:])
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
print("Post functionalisation (in illuminated conditions), for source-drain measurements (V_g = 5 V):")
print(f"The resistance during the first forward pass is {R1:e} ± {err1:e} Ohms")
print(f"The resistance during the first backward pass is {R2:e} ± {err2:e} Ohms")
print(f"The resistance during the second forward pass is {R3:e} ± {err3:e} Ohms")
print(f"The resistance during the second backward pass is {R4:e} ± {err4:e} Ohms")
print(f"The average resistance during a forward pass is {R_fwd_avg:e} ± {err_fwd_avg:e} Ohms")
print(f"The average resistance during a backward pass is {R_bwd_avg:e} ± {err_bwd_avg:e} Ohms")
print(f"The overall average resistance is {R_avg:e} ± {err_avg:e} Ohms")

"""
Calculate the field effect mobility and carrier concentration using the
gradient of the linear region in the I_sd versus V_g graphs.
"""

Vg, Isd_dark = np.loadtxt("OFET4F-Isd-Vg-Dark.dat", delimiter='\t',
                          usecols=(0, 3), unpack=True)
Isd_light = np.loadtxt("OFET4F-Isd-Vg-Light.dat", delimiter='\t',
                       usecols=(3), unpack=True)
Vg1 = Vg[:len(Vg)//2]
Vg2 = Vg[len(Vg)//2:]
Isd_dark1 = Isd_dark[:len(Isd_dark)//2]
Isd_dark2 = Isd_dark[len(Isd_dark)//2:]
Isd_light1 = Isd_light[:len(Isd_light)//2]
Isd_light2 = Isd_light[len(Isd_light)//2:]

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
print()
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
# plt.savefig("OFET4F-sigma-n-dark.png", dpi=500)

plt.show()
