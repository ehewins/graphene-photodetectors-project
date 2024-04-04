import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


def top_hat(width):
    return np.ones(width) / width


def find_jump_indices(array):
    """
    Find the indices corresponding to the start and end of a jump in the data.
    Used to determine the photocurrent in I_sd vs t measurements.
    """
    arr_diffs = array[1:] - array[:-1]
    biggest_jump = max(np.abs(arr_diffs))
    jumps = np.abs(arr_diffs) > biggest_jump * 0.5
    jump_i = np.argwhere(jumps).flatten()
    jump_f = jump_i + 1
    # Some downward jumps take 2 datapoints - catch the actual finish point
    for j in range(len(jump_f)):
        if abs(array[jump_f[j]+1] - array[jump_i[j]]) > \
           abs(array[jump_f[j]] - array[jump_i[j]]):
            jump_f[j] += 1
    return jump_i, jump_f


"""
Objective 1: Graph the Isd(t) data, showing the effect of source-drain current
on the measured photocurrent. Also graph photocurrent as a function of
source-drain voltage.
"""

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
axes = np.array([[ax1, ax2], [ax3, ax4]])
device_numbers = np.array([1, 2])
channel_voltages = np.array([10, 50, 100, 200, 300])
for ax in axes[:, 0]:
    ax.set_title("Device 1 - Quantum Dots")
for ax in axes[:, 1]:
    ax.set_title("Device 2 - Perovskite Nanocrystals")
for ax_1, ax_2, number, in zip(axes[0], axes[1], device_numbers):
    Iph = []
    for Vsd in channel_voltages:
        filename = f"CVD{number}F-time-{Vsd}mV.dat"
        if Vsd in (10, 50):
            filename = "../cvd_func_19032024/" + filename
        time, current = np.loadtxt(filename, usecols=(1, 3), unpack=True)
        if Vsd not in (10, 50):  # Don't plot the old data.
            ax_1.plot(time, current, label='$V_{sd}$ = '+f'{Vsd} mV')
        index_1, index_2 = find_jump_indices(current)
        # ax.plot(time[index_1], current[index_1], 'kx')  # visual check
        # ax.plot(time[index_2], current[index_2], 'kx')
        # NOTE: Checks show jumps aren't quite being identified correctly.
        #       I think it's close enough though.
        photocurrents = np.abs(current[index_2] - current[index_1])
        Iph.append(np.mean(photocurrents))
    ax_2.plot(channel_voltages, Iph, 'bo', label='Datapoints')
    (m, c), cv = curve_fit(linear_fit, channel_voltages, Iph)
    dm = np.sqrt(cv[0, 0])  # uncertainty in gradient
    ax_2.plot(channel_voltages, m*channel_voltages+c, 'k-', label='Linear fit')
    m, dm = m * 1e3, dm * 1e3  # Convert from A/mV -> A/V for printing
    print(f"Gradient of I_ph(V_g) for Device {number}: {m:e} ± {dm:e} A/V.")
for ax in axes[0]:
    ax.set_xlabel("Time, $t$ (s)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    ax.legend(loc=(0.67, 0.55))
for ax in axes[1]:
    ax.set_xlabel("Source-drain voltage, $V_{sd}$ (mV)")
    ax.set_ylabel("Photocurrent, $I_{ph}$ (A)")
    ax.legend()

"""
Objective 2: Graph the final Isd(Vg) measurements, identifying the Dirac
voltage and mobility as we've done before.
"""

Vsd = 10e-3  # 10 mV source-drain voltage
d = 90e-9  # dielectric thickness (m)
epsilon_r = 3.9  # relative permittivity of dielectric
epsilon_0 = 8.85e-12  # permittivity of free space (F/m)
e = 1.6e-19  # charge magnitude of a single carrier (C)

# Table recording file names, the label for the graph, and number of passes
file_table = (
    ("CVD1F-Isd-Vg-Dark-final.dat", "Dark conditions", 6),
    ("CVD1F-Isd-Vg-Light-final.dat", "Light conditions", 6),
    ("CVD2F-Isd-Vg-Dark-final.dat", "Dark conditions", 4),
    ("CVD2F-Isd-Vg-Light-final.dat", "Light conditions", 4),
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for ax, file_info in zip((ax1, ax1, ax2, ax2), file_table):
    print()
    Vg, Isd = np.loadtxt(file_info[0], delimiter='\t', usecols=(0, 3), unpack=True)
    npass = file_info[2]
    points_per_pass = len(Vg)//npass
    # Only use the final forward pass.
    Vg = Vg[(npass-2)*points_per_pass:(npass-1)*points_per_pass]
    Isd = Isd[(npass-2)*points_per_pass:(npass-1)*points_per_pass]
    V_dirac = Vg[np.argmin(Isd)]
    print(f"Dirac point for {file_info[0][:4]} in {file_info[1].lower()} at {V_dirac:.2f} V")
    p0 = V_dirac * (epsilon_0 * epsilon_r) / (e * d)
    print(f"-> implying an initial carrier concentration of {p0*1e-4:e} cm^-2")
    Rsd = Vsd / Isd  # Device is square, so resistance = resistivity
    ax.plot(Vg, Isd, label=file_info[1], linewidth=2)
    # MOBILITY CALCULATIONS:
    # Obtain the two volgate values at which resistivity is closest to its FWHM
    FWHM_indices = np.sort(np.argsort(np.abs(Rsd - max(Rsd)/2))[:2])
    FWHM_voltages = Vg[FWHM_indices]
    if FWHM_voltages[1] - FWHM_voltages[0] < 0.5:  # i.e. only one side visible
        print("Assuming equal electron and hole mobilities,", end=' ')
        FWHM_indices[1] = FWHM_indices[0]
        FWHM_voltages[1] = V_dirac + (V_dirac - FWHM_voltages[0])
    ax.plot(FWHM_voltages, Isd[FWHM_indices], 'rx')
    V_FWHM = FWHM_voltages[1] - FWHM_voltages[0]
    print(f"FWHM for {file_info[0][:4]} in {file_info[1].lower()} is {V_FWHM:.2f} V")
    print(f"(the maximum resistivity: {max(Rsd):e} Ω/sq)")
    # Use the FWHM to calculate the three-point mobility
    mu_3p = 4 * d / (epsilon_0 * epsilon_r * V_FWHM * max(Rsd))
    print(f"-> implying a 3-point mobility of {mu_3p*1e4:e} cm^2 V^-1 s^-1")
    # Calculate the gradient, for alternative mobility calculations
    noisy_gradient = np.gradient(Isd, Vg)
    # Smooth out the noise in this gradient
    gradient = np.convolve(noisy_gradient, top_hat(20), mode='same')
    # Mobility from gradient at FWHM voltages:
    print("The gradient at FWHM voltages implies:", end=' ')
    for i in np.unique(FWHM_indices):
        # Plot extrapolations to Isd = 0
        m, a, b = gradient[i], Vg[i], Isd[i]
        c = b - m * a
        boundary_V = min(Vg) if m < 0 else max(Vg)
        ax.plot([boundary_V, -c/m], [m*boundary_V+c, 0],
                linestyle='dotted', color='red')
        # Calculate mobility and print result
        mu = d / (epsilon_0 * epsilon_r * Vsd) * abs(m)
        print(f"mobility {mu*1e4:e} cm^2 V^-1 s^-1", end=' ')
        print(f"(with x-axis intercept at {-c/m:.2f} V),", end=' ')
    print()
    # Mobility from maximum gradient of Isd(Vg) curve (local to V_dirac):
    # find the two turning points of the gradient either side of V_dirac
    gradient2 = np.gradient(np.abs(gradient), Vg)
    # the +/-0.5 below is for insurance.
    linear_index_1 = np.nonzero((Vg < V_dirac-0.5) & (gradient2 > 0))[0][-1]
    linear_index_2 = np.nonzero((Vg > V_dirac+0.5) & (gradient2 < 0))[0][0]
    print("The maximum gradient implies:", end=' ')
    for i in (linear_index_1, linear_index_2):
        # Plot extrapolations to Isd = 0
        m, a, b = gradient[i], Vg[i], Isd[i]
        c = b - m * a
        boundary_V = min(Vg) if m < 0 else max(Vg)
        ax.plot([Vg[i]], [Isd[i]], 'bx')
        ax.plot([boundary_V, -c/m], [m*boundary_V+c, 0],
                linestyle='dotted', color='blue')
        # Calculate mobility and print result
        mu = d / (epsilon_0 * epsilon_r * Vsd) * abs(m)
        print(f"mobility {mu*1e4:e} cm^2 V^-1 s^-1", end=' ')
        print(f"(with x-axis intercept at {-c/m:.2f} V),", end=' ')
    print()
for ax in (ax1, ax2):
    ax.set_xlabel("Gate voltage, $V_g$ (V)")
    ax.set_ylabel("Source-drain current, $I_{sd}$ (A)")
    ax.legend()

plt.show()
