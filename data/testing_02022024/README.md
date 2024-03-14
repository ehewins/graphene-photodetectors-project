# Test Measurements

The measurement data in this folder was recorded on the 2nd of February 2024. The purpose of these measurments was to familiarise ourselves with the measurement equipment, and the results here have no practical use for the project. The results were collected at room temperature and pressure. All current measurements are in units of Amperes, and all voltage measurements in Volts.

## About the Files

`nothing-Isd-Vg.dat` and `nothing-Isd-Vsd.dat` are current and voltage measurements for an empty header. The four columns in former are the gate voltage, the gate current, the source-drain voltage (constant 1 V) and the source-drain current. The two columns in the latter are the source-drain voltage and the source-drain current (with zero gate voltage).

`OFET1-Isd-Vg-Vsd0.dat` and `OFET1-Isd-Vg-Vsd1.dat` are current and voltage measurements for an unfunctionalised (bare) Organic Field Effect Transistor (OFET). The four columns correspond to gate voltage, gate current, source-drain voltage (held constant) and source-drain current measurements.
`OFET1-Isd-Vsd-Vg1.dat` and `OFET1-Isd-Vsd-Vg5.dat` are further measurements of the same OFET, this time with the gate voltage held constant and the source-drain voltage varied. The four columns are the source-drain voltage, source-drain current, gate voltage, and gate current.

`OFET2-Isd-Vg-Vsd1.dat`, `OFET2-Isd-Vsd-Vg1.dat` and `OFET2-Isd-Vsd-Vg5.dat` contain the same types of measurement data as their equivalently named OFET1 files, but correspond to a second OFET device. We stopped taking measurements early, because we realised this device was broken.

The Python file `graphing.py` produces plots of the recorded data. No further data processing is carried out.

