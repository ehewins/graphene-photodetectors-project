# CVD Graphene FET Measurements

The measurement data in this folder was (mostly) recorded on the 22nd of March 2024. The long-time experiments ran over from previous days.
The results were collected at room temperature and a pressure of 3x10^{-6} mbar. The GFETs have a Si/SiO2 substrate, with an oxide layer thickness of 90 nm. All current measurements are in units of Amperes, and all voltage measurements in Volts.

# About the Files

`CVD1F-long-time.dat` and `CVD1F-long-time-lowres.dat` are long running experiments measuring the source-drain current as a function of time, for an initial period (~1h) with the device in darkness, then a period under illumination (>1h), followed by another stint in darkness (>>1h). These measurements enable the characteristic times associated with the various equilibriation processes to be determined. The latter of these two files was an experiment run prior to the former, but technical problems mean the current data is stored with extremely low resolution, and we only have the datapoint number column (equivalent in theory to the time in seconds) and the source-drain current column. The columns in the first file are the datapoint number, the time in seconds, the source-drain voltage and the source-drain current. 

`CVD1F-Isd-Vg-Dark-post-long.dat` and `CVD2F-Isd-Vg-Dark-post-long.dat` are more sets of source-drain current versus gate voltage measurements. These were taken to establish the Dirac voltage following the long experiments above, during which the Dirac point could have shifted considerably. The first column gives the gate voltage, the second the gate current, the third the source-drain voltage, and the fourth the source-drain current.

`CVD1F-time-power.dat` and `CVD2F-time-power.dat` are measurements of the source-drain current versus time, where a series of filters are used to gradually increase the power for a series of periods of light exposure. The first three 'on' periods correspond an OD3 filter, the next two an OD2 filter, the next one and OD1 filter, and the final one no filter (OD0). The columns are the datapoint number, the time in seconds, the source-drain voltage and the source-drain current.

`CVD1F-Isd-Vg-Dark-final.dat`, `CVD1F-Isd-Vg-Light-final.dat`, `CVD2F-Isd-Vg-Dark-final.dat`, and `CVD2F-Isd-Vg-Light-final.dat` are a final set of source-drain current versus gate voltage measurements. These were taken to confirm the final state of the devices after the day's measurements, and to inform choices made about the next day's measurements. The first column gives the gate voltage, the second the gate current, the third the source-drain voltage, and the fourth the source-drain current.

`data-processing.py` graphs the data in the files above, and also determines the parameters of interest from each file. This includes the characteristic timescales of the long-time experiments, the Dirac voltage and full width half maximum of the measurements with a varied gate voltage, and the photoresponse as a function of power for the time-power experiments.
