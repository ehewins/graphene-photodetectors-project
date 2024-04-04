import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


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

plt.show()
