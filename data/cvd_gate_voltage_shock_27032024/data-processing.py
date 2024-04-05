import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


"""
Objective 1:
Determine what you can from the one measurement we got before disaster struck.
"""

time, current = np.loadtxt("CVD1F-time-power-300mV-Vg0-pre-shock.dat",
                           delimiter='\t', usecols=(1, 3), unpack=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Device 1 (Quantm Dots) with $V_g = 0$ V and $V_{sd} = 300$ mV")
ax1.plot(time, current)
ax1.set_xlabel("Time, $t$ (s)")
ax1.set_ylabel("Source-drain current, $I_{sd}$ (A)")
ax2.set_xlabel("Power on device, $P$ (W)")
ax2.set_ylabel("Photoresponsivity, $R$ (A/W)")
jump_indices = [29, 30, 56, 57, 81, 82, 107, 108,
                133, 134, 159, 160, 186, 187, 212, 213]

P0 = 2.33e-6  # Power incident on device, without optical filters
dP0 = 0.07e-6  # The corresponding uncertainty
ODs = [3.9, 2.9, 1.1, 0]
responsivity = []
power = []
for i in range(4):
    OD = ODs[i]
    filter_name = "OD" + str(OD)[0]
    initial_indices = jump_indices[4*i:4*(i+1):2]
    final_indices = jump_indices[4*i+1:4*(i+1)+1:2]
    photocurrents = np.abs(current[final_indices] - current[initial_indices])
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
        ax1.plot([time[initial], time[final]],
                 [current[initial], current[final]],
                 color=f'C{i+1}', linewidth=1.8,
                 label='Filter '+filter_name if jumps_labelled == 0 else None)
        jumps_labelled = 1
ax1.legend()
# Create log plot of responsivity versus power, and fit a straight line
(m, c), cv = curve_fit(linear_fit, np.log10(power), np.log10(responsivity))
print(f"Gradient of log(R_ph) vs log(P) plot: {m:e} ± {np.sqrt(cv[0, 0]):e}")
ax2.loglog(power, responsivity, 'bo', label='Data')
ax2.loglog(power, 10**c * np.array(power)**m, 'k-', label='Linear fit')
ax2.legend()

plt.show()
