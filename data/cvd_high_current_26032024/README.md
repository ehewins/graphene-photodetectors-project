# CVD Graphene FET Measurements, with High Source-Drain Voltages

The measurement data in this folder was recorded on the 26th of March 2024.
The purpose of these measurements was to test the boundaries of how high a source-drain current can be passed through the graphene channel, and to observe the effect higher current has on the photoresponsivity of the device. We also sought to maximise the measured photocurrent by optimising the gate voltage across the device based on previous measurements. Device 1 (CVD1) had gate voltage of -11.5 V across it, and Device 2 (CVD2) a voltage of -1.3 V.
After these measurements, a final set of source-drain current versus gate voltage measurements were taken for both devices in both dark and illuminated conditions.
The results were collected at room temperature and a pressure of 3x10^{-6} mbar. The GFETs have a Si/SiO2 substrate, with an oxide layer thickness of 90 nm. All current measurements are in units of Amperes, and all voltage measurements in Volts.

# About the Files

`CVD1F-time-100mV.dat` and `CVD2F-time-100mV.dat` are measurements of the source-drain current versus time, while the window of the cryostat was repeatedly covered and uncovered to expose the devices to 4.5mW 405 nm laser light. The source-drain voltage was 100 mV during these measurements. The columns are the datapoint number, the time in seconds, the source-drain voltage and the source-drain current.

`CVD1F-time-200mV.dat`, `CVD2F-time-200mV.dat`, `CVD1F-time-300mV.dat`, and `CVD2F-time-300mV.dat` contain similar measurements to those described above, except with progressively increasing source-drain voltages (indicated in the file name).

`CVD1F-Isd-Vg-Dark-final.dat`, `CVD1F-Isd-Vg-Light-final.dat`, `CVD2F-Isd-Vg-Dark-final.dat`, and `CVD2F-Isd-Vg-Light-final.dat` are a final set of source-drain current versus gate voltage measurements. These were taken to confirm the final state of the devices after the day's measurements. The first column gives the gate voltage, the second the gate current, the third the source-drain voltage, and the fourth the source-drain current.

`data-processing.py` graphs the data in the files above, and also determines the parameters of interest from each file.
NOTE: Add more detail here at some point.
