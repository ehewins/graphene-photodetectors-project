import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


def on_exp(t, a, tau):
    return a * (1 - np.exp(- t / tau))


def off_exp(t, a, tau):
    return a * (np.exp(- t / tau) - 1)


def top_hat(width):
    return np.ones(width) / width


"""
Objective 1:
Analyse the long running experiments, determining the characteristic timescale
of dark -> light and light -> dark equilibriation processes (not sure about the
terminology there).
"""

time, current = np.loadtxt("CVD1F-long-time.dat", delimiter='\t',
                           usecols=(1, 3), unpack=True)
fit_region_indices = ((0, 5265), (5265, 8114), (8114, 9170), (9170, len(current)-1))
fit_region_names = ("Initial dark period", "Illuminated period",
                    "Final dark period (decreasing current)",
                    "Final dark period (increasing current)")

current *= 1e6  # convert to microamps for plotting
time /= 3600  # convert to hours for plotting
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(time, current)
ax.set_xlabel("Time, $t$ (hours)")
ax.set_ylabel("Source-drain current, $I_{sd}$, ($\\mu$A)")
on_region = (5220, 5310)  # indices defining subregions of interest
on_region_time = time[on_region[0]:on_region[1]]
on_region_current = current[on_region[0]:on_region[1]]
off_region = (5700, 14400)
off_region_time = time[off_region[0]:off_region[1]]
off_region_current = current[off_region[0]:off_region[1]]
inset_1 = ax.inset_axes([0.18, 0.55, 0.18, 0.35],
                        xlim=(min(on_region_time), max(on_region_time)),
                        ylim=(min(on_region_current), max(on_region_current)))
ax.indicate_inset_zoom(inset_1, edgecolor='black')
inset_1.plot(on_region_time, on_region_current)
inset_1.plot(np.ones(2)*time[fit_region_indices[0][1]],
             [min(on_region_current), max(on_region_current)],
             color='black', linestyle='dotted')
inset_1.annotate("Light off", (1.461, 3.1), rotation=90)
inset_1.annotate("Light on", (1.465, 3.1), rotation=90)
inset_2 = ax.inset_axes([0.5, 0.07, 0.42, 0.48],
                        xlim=(min(off_region_time), max(off_region_time)),
                        ylim=(min(off_region_current), max(off_region_current)))
ax.indicate_inset_zoom(inset_2, edgecolor='black')
inset_2.plot(off_region_time, off_region_current)
inset_2.plot(np.ones(2)*time[fit_region_indices[1][1]],
             [min(off_region_current), max(off_region_current)],
             color='black', linestyle='dotted')
inset_2.annotate("Light on", (2.14, 2.39), rotation=90)
inset_2.annotate("Light off", (2.30, 2.39), rotation=90)

current /= 1e6  # convert to back to amps
time *= 3600  # convert back to seconds
# Performing fits to determine parameters for various regions of the graph.
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 6))
for i, ax in zip(range(4), (ax1, ax2, ax3, ax4)):
    start, stop = fit_region_indices[i]
    region_I, region_t = current[start:stop], time[start:stop]
    ax.plot(region_t, region_I)
    ax.set_xlabel("Time, $t$ (seconds)")
    ax.set_ylabel("Source-drain current, $I_{sd}$, (A)")
    ax.set_title(fit_region_names[i])
    if i == 0:
        continue
    elif region_I[-1] > region_I[0]:
        fit_function = on_exp
    else:
        fit_function = off_exp
    fitting_t = region_t - region_t[0]
    fitting_I = region_I - region_I[0]
    (I_change, response_t), cv = curve_fit(fit_function, fitting_t, fitting_I)
    I_change_err, response_t_err = np.sqrt(np.diag(cv))
    print(f"During the {fit_region_names[i]}, the current changed by {I_change:e} ± {I_change_err:e} A, with a response time of {response_t:e} ± {response_t_err:e} seconds.")
    ax.plot(region_t, fit_function(fitting_t, I_change, response_t) +
            region_I[0], 'k--')

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
e = 1.6e-19  # charge magnitude of a single carrier (C)

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
    V_dirac = Vg[np.argmin(Isd)]
    print(f"Dirac point for {data[0][:4]} in {data[1].lower()} at {V_dirac:.2f} V")
    p0 = V_dirac * (epsilon_0 * epsilon_r) / (e * d)
    print(f"-> implying an initial carrier concentration of {p0*1e-4:e} cm^-2")
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
    I_ax.plot(FWHM_voltages, Isd[FWHM_indices], 'rx')
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
                  linestyle='dotted', color='red')
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
    print("The maximum gradient implies mobility:", end=' ')
    for i in (linear_index_1, linear_index_2):
        # Plot extrapolations to Isd = 0
        m, a, b = gradient[i], Vg[i], Isd[i]
        c = b - m * a
        boundary_V = min(Vg) if m < 0 else max(Vg)
        I_ax.plot([Vg[i]], [Isd[i]], 'bx')
        I_ax.plot([boundary_V, -c/m], [m*boundary_V+c, 0],
                  linestyle='dotted', color='blue')
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

# Setup for Device 1 data
CVD1_Vg, CVD1_Isd_dark = np.loadtxt(file_table[0][0], delimiter='\t',
                                    usecols=(0, 3), unpack=True)
npass = file_table[0][2]
start, end = np.array([npass-2, npass-1]) * len(CVD1_Isd_dark) // npass
CVD1_Vg, CVD1_Isd_dark = CVD1_Vg[start:end], CVD1_Isd_dark[start:end]
CVD1_Isd_light = np.loadtxt(file_table[1][0], delimiter='\t', usecols=(3))
npass = file_table[1][2]
start, end = np.array([npass-2, npass-1]) * len(CVD1_Isd_light) // npass
CVD1_Isd_light = CVD1_Isd_light[start:end]

# Setup for Device 2 data
CVD2_Vg, CVD2_Isd_dark = np.loadtxt(file_table[2][0], delimiter='\t',
                                    usecols=(0, 3), unpack=True)
npass = file_table[2][2]
start, end = np.array([npass-2, npass-1]) * len(CVD2_Isd_dark) // npass
CVD2_Vg, CVD2_Isd_dark = CVD2_Vg[start:end], CVD2_Isd_dark[start:end]
CVD2_Isd_light = np.loadtxt(file_table[3][0], delimiter='\t', usecols=(3))
npass = file_table[3][2]
start, end = np.array([npass-2, npass-1]) * len(CVD2_Isd_light) // npass
CVD2_Isd_light = CVD2_Isd_light[start:end]

# Calculate predicted photocurrent (magnitude), and plot.
Iph1 = np.abs(CVD1_Isd_dark - CVD1_Isd_light)
Iph2 = np.abs(CVD2_Isd_dark - CVD2_Isd_light)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(CVD1_Vg, Iph1)
ax1.plot(np.ones(2)*CVD1_Vg[np.argmax(Iph1)], [0, max(Iph1)], 'k--')
ax2.plot(CVD2_Vg, Iph2)
ax2.plot(np.ones(2)*CVD2_Vg[np.argmax(Iph2)], [0, max(Iph2)], 'k--')
for ax in (ax1, ax2):
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Theorised photocurrent, $I_{ph}$ (A)")
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")

"""
Objective 3:
Graph the current versus time measurements, with variable power light pulses.
Use these to determine the relationship between power and photoresponsivity.
"""

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("With constant $V_{sd} = 50$ mV and $V_g = 0$ V")
fig2.suptitle("With constant $V_{sd} = 50$ mV and $V_g = 0$ V")
for ax in (ax1, ax3):
    ax.set_title("Device 1 - Quantum Dots")
for ax in (ax2, ax4):
    ax.set_title("Device 2 - Perovskites")
for ax in (ax1, ax2):
    ax.set_xlabel("Time, $t$, (s)")
    ax.set_ylabel("Source-drain current, $I_{sd}$, (A)")
for ax in (ax3, ax4):
    ax.set_xlabel("Power on device, $P$ (W)")
    ax.set_ylabel("Photoresponsivity, $R$ (A/W)")
time1, current1 = np.loadtxt("CVD1F-time-power.dat", delimiter='\t',
                             usecols=(1, 3), unpack=True)
time2, current2 = np.loadtxt("CVD2F-time-power.dat", delimiter='\t',
                             usecols=(1, 3), unpack=True)
ax1.plot(time1, current1, label='Current')
ax2.plot(time2, current2, label='Current')
current1_jump_indices = [422, 423, 481, 482, 546, 547, 602, 603, 641, 642, 698,
                         699, 753, 754, 810, 811, 861, 862, 905, 906, 962, 963,
                         1030, 1031, 1065, 1066, 1120, 1121]
current2_jump_indices = [66, 67, 105, 106, 141, 142, 178, 179, 214, 215, 250,
                         251, 286, 287, 326, 327, 362, 363, 398, 399, 437, 438,
                         470, 471, 509, 510, 541, 542]
# For both sets of measurements, the first three pairs of jumps have filter
# OD3, the next three pairs have OD2, the next pair has OD1, and the final
# pair has OD0.
P0 = 2.33e-6  # Power incident on device, without optical filters
dP0 = 0.07e-6  # The corresponding uncertainty
ODs = [3.7, 2.6, 1.1, 0]
jumps_per_OD = [6, 4, 2, 2]
index_range_boundaries = np.append([0], 2*np.cumsum(jumps_per_OD))
print()
print("For Device 1, functionalised with quantum dots:")
responsivity = []
power = []
for i in range(4):
    OD = ODs[i]
    filter_name = "OD" + str(OD)[0]
    # Calculate photoresponsivity for this filter
    start, stop = index_range_boundaries[i], index_range_boundaries[i+1]
    initial_indices = current1_jump_indices[start:stop][0::2]
    final_indices = current1_jump_indices[start:stop][1::2]
    photocurrents = np.abs(current1[final_indices] - current1[initial_indices])
    Iph = np.mean(photocurrents)
    dIph = np.std(photocurrents)
    rel_dIph = dIph / Iph
    P = P0 * 10**(-OD)  # Power after filtering
    Rph = Iph / P  # Photoresponsivity,
    dRph = Rph * np.sqrt((dIph/Iph)**2 + (dP0/P0)**2)  # & its uncertainty
    print(f"With filter {filter_name}, photoresponsivity R = {Rph:e} ± {dRph:e} A/W")
    responsivity.append(Rph)
    power.append(P)
    # Mark jumps corresponding to this filter on the graph
    jumps_labelled = 0
    for initial, final in zip(initial_indices, final_indices):
        ax1.plot([time1[initial], time1[final]],
                 [current1[initial], current1[final]],
                 color=f'C{i+1}', linewidth=1.8,
                 label='Filter '+filter_name if jumps_labelled == 0 else None)
        jumps_labelled = 1
ax1.legend()
# Create log plot of responsivity versus power, and fit a straight line
(m, c), cv = curve_fit(linear_fit, np.log10(power), np.log10(responsivity))
print(f"Gradient of log(R_ph) vs log(P) plot: {m:e} ± {np.sqrt(cv[0, 0]):e}")
ax3.loglog(power, responsivity, 'bo', label='Data')
ax3.loglog(power, 10**c * np.array(power)**m, 'k-', label='Linear fit')
ax3.legend()

print("For Device 2, functionalised with perovskites:")
responsivity = []
power = []
for i in range(4):
    OD = ODs[i]
    filter_name = "OD" + str(OD)[0]
    start, stop = index_range_boundaries[i], index_range_boundaries[i+1]
    initial_indices = current2_jump_indices[start:stop][0::2]
    final_indices = current2_jump_indices[start:stop][1::2]
    photocurrents = np.abs(current2[final_indices] - current2[initial_indices])
    Iph = np.mean(photocurrents)
    dIph = np.std(photocurrents)
    rel_dIph = dIph / Iph
    P = P0 * 10**(-OD)  # Power after filtering
    Rph = Iph / P  # Photoresponsivity,
    dRph = Rph * np.sqrt((dIph/Iph)**2 + (dP0/P0)**2)  # & its uncertainty
    print(f"With filter {filter_name}, photoresponsivity R = {Rph:e} ± {dRph:e} A/W")
    responsivity.append(Rph)
    power.append(P)
    # Mark jumps corresponding to this filter on the graph
    labelled = 0
    for initial, final in zip(initial_indices, final_indices):
        ax2.plot([time2[initial], time2[final]],
                 [current2[initial], current2[final]],
                 color=f'C{i+1}', linewidth=1.8,
                 label=f'Filter {filter_name}' if labelled == 0 else None)
        labelled = 1
ax2.legend()
# Create log plot of responsivity versus power, and fit a straight line
(m, c), cv = curve_fit(linear_fit, np.log10(power), np.log10(responsivity))
print(f"Gradient of log(R_ph) vs log(P) plot: {m:e} ± {np.sqrt(cv[0, 0]):e}")
ax4.loglog(power, responsivity, 'bo', label='Data')
ax4.loglog(power, 10**c * np.array(power)**m, 'k-', label='Linear fit')
ax4.legend()

plt.show()
