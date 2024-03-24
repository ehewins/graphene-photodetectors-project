# CVD Graphene FET Measurements, Post-functionalisation

The measurement data in this folder was recorded on the 15th of March 2024.
These measurements were taken to investigate the properties of the graphene transistors, now that they have been functionalised with photosensitive materials. Device 1 has been functionalised with the quantum dots, and Device 2 with the perovskites. The dependence of the source-drain current on both the source drain voltage and the gate voltage is measured, in both dark and illuminated (405 nm, 4.5 mW light) conditions, as well as the photocurrent resulting from illumination.
The results were collected at room temperature and a pressure of 3x10^{-6} mbar. The GFETs have a Si/SiO2 substrate, with an oxide layer thickness of 90 nm. All current measurements are in units of Amperes, and all voltage measurements in Volts.

# About the Files

`CVD1F-Isd-Vsd-Dark.dat`, `CVD1F-Isd-Vsd-Light.dat`, `CVD2F-Isd-Vsd-Dark.dat`, and `CVD2F-Isd-Vsd-Light.dat` are all measurements of the source-drain current (second column) as a function of source-drain voltage (first column), with zero gate voltage. CVD1F / CVD2F refers to Device 1 or 2, and the Dark / Light part of the filename indicates whether the laser was shone onto the device during the measurements.

`CVD1F-Isd-Vg-Dark.dat`, `CVD1F-Isd-Vg-Light.dat`, `CVD2F-Isd-Vg-Dark.dat`, and `CVD2F-Isd-Vg-Light.dat` contain, primarily, measurements of the source-drain current as a function of gate voltage, with a constant source-drain voltage of 10 mV. The first column gives the gate voltage, the second the gate current, the third the source-drain voltage, and the fourth the source-drain current.

`CVD1F-time.dat` and `CVD2F-time.dat` contain time-series measurements of the source-drain current while we repeatedly uncovered and covered the window of the cryostat, exposing the device to the laser light described above. The columns are the datapoint number, the time in seconds, the source-drain voltage and the source-drain current.

`data-processing.py` graphs the data in the files above, as well as performing linear fitting to specific regions of data. It calculates the resistance, field effect mobility, and initial carrier concentration, of graphene chips, as well as the photocurrent.
