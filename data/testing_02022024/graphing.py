"""
Script to produce plots of the data. No further data processing is carried out,
as these data were only collected for the sake of practising with the
measurement equipment.
"""

import numpy as np
import matplotlib.pyplot as plt

file_names = ("nothing-Isd-Vg.dat", "nothing-Isd-Vsd.dat",
              "OFET1-Isd-Vg-Vsd0.dat", "OFET1-Isd-Vg-Vsd1.dat",
              "OFET1-Isd-Vsd-Vg1.dat", "OFET1-Isd-Vsd-Vg5.dat",
              "OFET2-Isd-Vg-Vsd1.dat", "OFET2-Isd-Vsd-Vg1.dat",
              "OFET2-Isd-Vsd-Vg5.dat")

for file_name in file_names:
    data = np.loadtxt(file_name, delimiter='\t')
    x_data = data[:, 0]
    base = file_name.split(".", maxsplit=1)[0]
    plt.figure(figsize=(14, 6))
    plt.tight_layout()
    for y_data in data[:, 1::2].T:
        plt.plot(x_data, y_data)
    if base.split("-")[2] == "Vg":
        plt.xlabel("Gate Voltage, $V_g$ (V)")
        plt.legend(["Gate current, $I_g$",
                   "Source-Drain Current, $I_{sd}$"])
        const_V_type = "$V_{sd}$"
    else:
        plt.xlabel("Source-Drain Voltage, $V_{sd}$ (V)")
        plt.legend(["Source-Drain Current, $I_{sd}$",
                   "Gate current, $I_g$"])
        const_V_type = "$V_g$"
    final = base[-1]
    if final in ("0", "1", "5"):
        plt.title("Readings for " + base.split("-")[0] + " with " +
                  const_V_type + " = " + final + " V")
    else:
        plt.title("Readings with no device in test bed")
    plt.ylabel("Current (A)")
    # plt.savefig(base + ".png", dpi=500)

plt.show()
