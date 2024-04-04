# CVD Graphene FET Measurements, After a Gate Voltage Shock

The measurement data in this folder was recorded on the 27th of March 2024.
All of the measurement data in this folder, except for one file, was recorded *after* an accidental gate voltage shock. The gate voltage was taken from 0 V to -11.5 V in an instant, (instead of gradully) which significantly reduced device performance. The gate still works, but the charge transfer efficiency is not what it once was.
The measurements of `CVD1F-time-power-300mV-Vg0-pre-shock.dat` were taken before this shock, after which we realised we had meant to take the measurements with a gate voltage of -11.5 V, not zero. The other measurements were taken after the accident to assess the extent of the damage.
The results were collected at room temperature and a pressure of 3x10^{-6} mbar. The GFETs have a Si/SiO2 substrate, with an oxide layer thickness of 90 nm. All current measurements are in units of Amperes, and all voltage measurements in Volts.

# About the Files

`CVD1F-time-power-300mV-Vg0-pre-shock.dat` is the one measurement we took before the accident, with the gate voltage still set to zero Volts. `CVD1F-time-power-300mV-Vg0.dat` is the same set of measurements, taken after the accident for comparison. `CVD1F-time-power-300mV-Vg-11.5.dat` has the gate voltage set to -11.5 V. I believe a gate voltage of -1.3 V was used for the measurements of `CVD2F-time-power-300mV.dat`, though I'm not certain. All these are measurements of the source-drain current versus time, where a series of filters are used to gradually increase the power for a series of periods of light exposure. The first few jumps in current correspond an OD3 filter, the next an OD2 filter, then an OD1 filter, and the finally no filter (OD0). The columns are the datapoint number, the time in seconds, the source-drain voltage and the source-drain current.

`CVD1F-time-300mV.dat` contains measurements equivalent to those in the file with the same name, in the `../cvd_high_current_26032024/` folder. The columns are the datapoint number, the time in seconds, the source-drain voltage and the source-drain current.

`CVD1F-Isd-Vg-Dark.dat`, `CVD1F-Isd-Vg-Light.dat`, `CVD2F-Isd-Vg-Dark.dat`, and `CVD2F-Isd-Vg-Light.dat` are another set of what has become our standard device characterisation measurements - source-drain current versus gate voltage measurements. The first column gives the gate voltage, the second the gate current, the third the source-drain voltage, and the fourth the source-drain current.
