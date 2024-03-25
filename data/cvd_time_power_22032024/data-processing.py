import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


def top_hat(width):
    return np.ones(width) / width


"""
Objective 1:
Analyse the long running experiments, determining the characteristic timescale
of dark -> light and light -> dark equilibriation processes (not sure about the
terminology there).
"""

# time, current = np.loadtxt("CVD1F-long-time.dat", delimiter='\t',
#                            usecols=(1, 3), unpack=True)
# regions = ((0, 5264), (5265, 8113), (8114, 9170), (9170, len(current)-1))
# region_names = ("Initial Dark Period", "Illuminated Period",
#                 "Final Dark Period (decreasing current)",
#                 "Final Dark Period (increasing current)")

# current *= 1e6  # convert to microamps for plotting
# time /= 3600  # convert to hours for plotting
# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# ax.plot(time, current)
# ax.set_xlabel("Time, $t$ (hours)")
# ax.set_ylabel("Source-drain current, $I_{sd}$, ($\\mu$A)")
# on_region = (5220, 5310)  # indices defining subregions of interest
# on_region_time = time[on_region[0]:on_region[1]]
# on_region_current = current[on_region[0]:on_region[1]]
# off_region = (5700, 14600)
# off_region_time = time[off_region[0]:off_region[1]]
# off_region_current = current[off_region[0]:off_region[1]]
# inset_1 = ax.inset_axes([0.2, 0.55, 0.2, 0.4],
#                         xlim=(min(on_region_time), max(on_region_time)),
#                         ylim=(min(on_region_current), max(on_region_current)))
# inset_1.plot(on_region_time, on_region_current)
# inset_2 = ax.inset_axes([0.5, 0.07, 0.4, 0.45],
#                         xlim=(min(off_region_time), max(off_region_time)),
#                         ylim=(min(off_region_current), max(off_region_current)))
# inset_2.plot(off_region_time, off_region_current)
# ax.indicate_inset_zoom(inset_1, edgecolor='black')
# ax.indicate_inset_zoom(inset_2, edgecolor='black')

"""
Objective 2:
Determine the Dirac voltage for this set of measurements, as well as the FWHM
and hence the three point mobility. Compare to the mobility found using the
gradient of various parts of the curve.
"""

Vsd = 10e-3  # 10 mV source-drain voltage
d = 90e-9  # dielectric thickness (m)
epsilon_r = 3.9  # relative permittivity of dielectric
epsilon_0 = 8.85e-12  # permittivity of free space (F/m)

# Table recording file names, the label for the graph, and number of passes
file_table = (
    ("CVD1F-Isd-Vg-Dark-final.dat", "Dark conditions", 4),
    ("CVD1F-Isd-Vg-Light-final.dat", "Light conditions", 6),
    ("CVD2F-Isd-Vg-Dark-final.dat", "Dark conditions", 6),
    ("CVD2F-Isd-Vg-Light-final.dat", "Light conditions", 4),
)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
for R_ax, I_ax, data in zip((ax1, ax1, ax2, ax2), (ax3, ax3, ax4, ax4), file_table):
    print()
    Vg, Isd = np.loadtxt(data[0], delimiter='\t', usecols=(0, 3), unpack=True)
    npass = data[2]
    points_per_pass = len(Vg)//npass
    # Only use the final forward pass.
    Vg = Vg[(npass-2)*points_per_pass:(npass-1)*points_per_pass]
    Isd = Isd[(npass-2)*points_per_pass:(npass-1)*points_per_pass]
    # Only use the first forward pass.
    # Vg = Vg[:points_per_pass]
    # Isd = Isd[:points_per_pass]
    V_dirac = Vg[np.argmin(Isd)]
    print(f"Dirac point for {data[0][:4]} in {data[1].lower()} at {V_dirac:.2f} V")
    Rsd = Vsd / Isd  # Device is square, so resistance = resistivity
    R_ax.plot(Vg, Rsd, label=data[1])
    I_ax.plot(Vg, Isd, label=data[1], linewidth=2)

    # MOBILITY CALCULATIONS:
    # Obtain the two volgate values at which resistivity is closest to its FWHM
    FWHM_indices = np.sort(np.argsort(np.abs(Rsd - max(Rsd)/2))[:2])
    FWHM_voltages = Vg[FWHM_indices]
    if FWHM_voltages[1] - FWHM_voltages[0] < 0.5:  # i.e. only one side visible
        print("Assuming equal electron and hole mobilities,", end=' ')
        FWHM_indices[1] = FWHM_indices[0]
        FWHM_voltages[1] = V_dirac + (V_dirac - FWHM_voltages[0])
    I_ax.plot(FWHM_voltages, Isd[FWHM_indices], 'kx')
    V_FWHM = FWHM_voltages[1] - FWHM_voltages[0]
    print(f"FWHM for {data[0][:4]} in {data[1].lower()} is {V_FWHM:.2f} V")
    # Use the FWHM to calculate the three-point mobility
    mu_3p = 4 * d / (epsilon_0 * epsilon_r * V_FWHM * max(Rsd))
    print(f"-> implying a 3-point mobility of {mu_3p*1e4:e} cm^2 V^-1 s^-1")
    # Calculate the gradient, for alternative mobility calculations
    noisy_gradient = np.gradient(Isd, Vg)
    # Smooth out the noise in this gradient
    gradient = np.convolve(noisy_gradient, top_hat(20), mode='same')
    # Mobility from gradient at FWHM voltages:
    print("The gradient at FWHM voltages implies mobility:", end=' ')
    for i in np.unique(FWHM_indices):
        # Plot extrapolations to Isd = 0
        m, a, b = gradient[i], Vg[i], Isd[i]
        c = b - m * a
        boundary_V = min(Vg) if m < 0 else max(Vg)
        I_ax.plot([boundary_V, -c/m], [m*boundary_V+c, 0],
                  linestyle='dotted', color='black', linewidth=1.2)
        # Calculate mobility and print result
        mu = d / (epsilon_0 * epsilon_r * Vsd) * abs(m)
        print(f"{mu:e} cm^2 V^-1 s^-1,", end=' ')
    print()
    # Mobility from maximum gradient of Isd(Vg) curve (local to V_dirac):
    # find the two turning points of the gradient either side of V_dirac
    gradient2 = np.gradient(np.abs(gradient), Vg)
    # the +/-0.5 below is for insurance.
    linear_index_1 = np.nonzero((Vg < V_dirac-0.5) & (gradient2 > 0))[0][-1]
    linear_index_2 = np.nonzero((Vg > V_dirac+0.5) & (gradient2 < 0))[0][0]
    print("The maxiumum gradient implies mobility:", end=' ')
    for i in (linear_index_1, linear_index_2):
        # Plot extrapolations to Isd = 0
        m, a, b = gradient[i], Vg[i], Isd[i]
        c = b - m * a
        boundary_V = min(Vg) if m < 0 else max(Vg)
        I_ax.plot([boundary_V, -c/m], [m*boundary_V+c, 0],
                  linestyle='dotted', color='blue', linewidth=1.2)
        # Calculate mobility and print result
        mu = d / (epsilon_0 * epsilon_r * Vsd) * abs(m)
        print(f"{mu:e} cm^2 V^-1 s^-1,", end=' ')
    print()
for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.legend()
for ax in (ax1, ax2):
    ax.set_ylabel("Resistivity, $\\rho$ ($\\Omega$/sq)")
for ax in (ax3, ax4):
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
for ax in (ax1, ax3):
    ax.set_title("Device 1 - Functionalised with Quantum Dots")
for ax in (ax2, ax4):
    ax.set_title("Device 2 - Functionalised with Perovskites")

"""
Obtains plots of the assumed photocurrent as a function of gate voltage
"""

CVD1_Vg, CVD1_Isd_dark = np.loadtxt(file_table[0][0], delimiter='\t',
                                    usecols=(0, 3), unpack=True)
npass = file_table[0][2]
start, end = np.array([npass-2, npass-1]) * len(CVD1_Isd_dark) // npass
CVD1_Vg, CVD1_Isd_dark = CVD1_Vg[start:end], CVD1_Isd_dark[start:end]
CVD1_Isd_light = np.loadtxt(file_table[1][0], delimiter='\t', usecols=(3))
npass = file_table[1][2]
start, end = np.array([npass-2, npass-1]) * len(CVD1_Isd_light) // npass
CVD1_Isd_light = CVD1_Isd_light[start:end]

CVD2_Vg, CVD2_Isd_dark = np.loadtxt(file_table[2][0], delimiter='\t',
                                    usecols=(0, 3), unpack=True)
npass = file_table[2][2]
start, end = np.array([npass-2, npass-1]) * len(CVD2_Isd_dark) // npass
CVD2_Vg, CVD2_Isd_dark = CVD2_Vg[start:end], CVD2_Isd_dark[start:end]
CVD2_Isd_light = np.loadtxt(file_table[3][0], delimiter='\t', usecols=(3))
npass = file_table[3][2]
start, end = np.array([npass-2, npass-1]) * len(CVD2_Isd_light) // npass
CVD2_Isd_light = CVD2_Isd_light[start:end]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(CVD1_Vg, np.abs(CVD1_Isd_dark - CVD1_Isd_light))
ax2.plot(CVD2_Vg, np.abs(CVD2_Isd_dark - CVD2_Isd_light))
for ax in (ax1, ax2):
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Theorised photocurrent, $I_{ph}$ (A)")
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")

plt.show()
