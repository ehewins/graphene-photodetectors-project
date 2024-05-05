import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    return m * x + c


"""
In the interests of time, I'm copied and pasted the data here from the the
files `data/cvd_func_19032024/data-processing.py` and
`data/cvd_time_power_22032024/data-processing.py`, because I didn't want to
have to copy and paste all the code which generates it. We're just making a new
graph here, which displays the photoresponse as a function of power incident on
the device for the measurements from the 19th and 22nd of March.
"""

power_1 = np.array([2.33e-06, 1.8507847869075756e-07,
                    4.648961193877488e-09, 5.852695385417321e-10])
cvd1_responsivity_1 = np.array([0.1636577789699572, 1.8225221397221487,
                                28.716280784579567, 59.5236428788025])
cvd1_responsivity_1_err = np.array([1.169410e-02, 8.840402e-02,
                                    1.145998e+00, 2.142212e+00])
cvd2_responsivity_1 = np.array([0.16702950643776832, 1.6614033256334326,
                                28.129133487330527, 75.49299440748118])
cvd2_responsivity_1_err = np.array([5.524159e-03, 7.745496e-02,
                                    2.010396e+00, 1.070706e+01])
power_2 = np.array([4.648961193877488e-10, 5.85269538541732e-09,
                    1.8507847869075756e-07, 2.33e-06])
cvd1_responsivity_2 = np.array([226.52372349050566, 58.08306730724567,
                                4.341509643282623, 0.42208583690987145])
cvd1_responsivity_2_err = np.array([1.877089e+01, 1.891703e+00,
                                    1.304318e-01, 1.350277e-02])
cvd2_responsivity_2 = np.array([106.21369220797607, 25.425037559747942,
                                1.6721825367773298, 0.165424892703863])
cvd2_responsivity_2_err = np.array(21.217404e+01, 1.052386e+00,
                                   7.879564e-02, 1.295672e-02])
(cvd1_m1, cvd1_c1), cvd1_cv1 = curve_fit(linear_fit, np.log10(power_1),
                                         np.log10(cvd1_responsivity_1))
(cvd1_m2, cvd1_c2), cvd1_cv2 = curve_fit(linear_fit, np.log10(power_2),
                                         np.log10(cvd1_responsivity_2))
(cvd2_m1, cvd2_c1), cvd2_cv1 = curve_fit(linear_fit, np.log10(power_1),
                                         np.log10(cvd2_responsivity_1))
(cvd2_m2, cvd2_c2), cvd2_cv2 = curve_fit(linear_fit, np.log10(power_2),
                                         np.log10(cvd2_responsivity_2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.errorbar(power_1, cvd1_responsivity_1, yerr=cvd1_responsivity_1_err, fmt='o', color='C0', capsize=2, ecolor='C3', label='Before long expt.')
ax1.plot(power_1, 10**cvd1_c1 * power_1**cvd1_m1, 'k-')
ax1.errorbar(power_2, cvd1_responsivity_2, yerr=cvd1_responsivity_2_err, fmt='o', color='C1', capsize=2, ecolor='C3', label='After long expt.')
ax1.plot(power_2, 10**cvd1_c2 * power_2**cvd1_m2, 'k-', label='Fit line')
ax2.errorbar(power_1, cvd2_responsivity_1, yerr=cvd2_responsivity_1_err, fmt='o', color='C0', capsize=2, ecolor='C3', label='Before long expt.')
ax2.plot(power_1, 10**cvd2_c1 * power_1**cvd2_m1, 'k-')
ax2.errorbar(power_2, cvd2_responsivity_2, yerr=cvd2_responsivity_2_err, fmt='o', color='C1', capsize=2, ecolor='C3', label='After long expt.')
ax2.plot(power_2, 10**cvd2_c2 * power_2**cvd2_m2, 'k-', label='Fit line')

for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Power on device, $P$ (W)")
    ax.set_ylabel("Photoresponsivity, $R$ (A/W)")
    ax.legend()
ax1.set_title("Device 1 - Functionalised with Quantum Dots")
ax2.set_title("Device 2 - Functionalised with Perovskites")
fig.tight_layout()

plt.show()
