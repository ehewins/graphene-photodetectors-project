# CVD Graphene FET Measurements, Pre-functionalisation

The measurement data in this folder was recorded on the 12th of March 2024.
These measurements were taken to investigate the conduction properties of two CVD graphene field effect transistors, which we have named CVD1 and CVD1. The dependence of the source-drain current on both the source drain voltage and the gate voltage is measured, as well as the photocurrent when the devices are illuminated by 405 nm light (laser power 4.5 mW).
The FETs are mounted on a Hall bar device, which prompted us to measure the 4-terminal voltage/resistance. Unfortunately the contacts failed during the measurement process, so these results are not usable.
The results were collected at room temperature and a pressure of 3x10^{-6} mbar. The GFETs have a Si/SiO2 substrate, with an oxide layer thickness of 90 nm. All current measurements are in units of Amperes, and all voltage measurements in Volts.

# About the Files

`CVD1-Isd-Vg.dat` and `CVD2-Isd-Vg.dat` contain current measurements for a fixed source-drain voltage of 10 mV, where the gate voltage is swept between +&- 10 V. The columns give the gate voltage, the gate current, the source-drain voltage, the source-drain current, and the 4-terminal voltage.

`CVD1-Isd-Vsd.dat` and `CVD2-Isd-Vsd.dat` contain current measurements for zero gate voltage, where the source-drain voltage was swept betweem +&- 10mV. The columns give the source-drain voltage, the source-drain current, and the 4-terminal voltage.

`CVD1-time.dat` and `CVD2-time.dat` contain time-series measurements of the source-drain current while we repeatedly uncovered and covered the window of the cryostat, exposing the device to the laser light described above. The columns are the datapoint number, the time in seconds, the source-drain voltage and the source-drain current.

`data-processing.py` graphs the data in the files above, as well as performing linear fitting to specific regions of data. It calculates the resistance, field effect mobility, and initial carrier concentration, of graphene chips, as well as the photocurrent.
