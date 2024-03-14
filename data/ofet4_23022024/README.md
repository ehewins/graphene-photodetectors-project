# Quantum Dot Measurements

The measurement data in this folder was recorded on the 23rd of February 2024.
These measurements were taken to investigate the electrical and optical properties of CdSe/ZnS core-shell type quantum dots - one of the materials we will be using to functionalise our hybrid graphene photodetectors. The measurements were taken by 'painting' a thin layer of the dots in solution onto an organic field effect transistor, before drying off the solvent in a cryostat.
The results were collected at room temperature and a pressure of 1x10^{-5} mbar. The OFETs are SiO2/Si based, with an oxide layer thickness of 230 nm. All current measurements are in units of Amperes, and all voltage measurements in Volts.

## About the Files

The files `OFET4-Isd-Vg.dat` and `OFET4-Isd-Vsd.dat` are measurements taken before the quantum dots were deposited onto the OFET.
The former contains measurments where the gate voltage was varied and the source-drain voltage was held at a constant 5 V. The columns give the gate voltage, gate current, source-drain voltage and source-drain current.
The latter contains measurements where the source-drain voltage was varied and the gate voltage was held at a constant 5 V. The columns give the source-drain voltage, source-drain current, gate voltage and gate current.

The files `OFET4F-Isd-Vg-Dark.dat` and `OFET4F-Isd-Vsd-Dark.dat` are measurements taken after the quantum dots were deposited onto the OFET, where the device was kept in dark conditions. They contain the same types of measurement as the two equivalently named files described above.
The files `OFET4F-Isd-Vg-Light.dat` and `OFET4F-Isd-Vsd-Light.dat` are measurements taken after the quantum dots were deposited onto the OFET, while the device was illuminated by 405 nm light (laser power: 4.5 mW). They contain the same types of measurement as the two equivalently named files described above.

The file `OFET4F-time.dat` contains time-series measurements of the source-drain current while we repeatedly uncovered and covered the window of the cryostat, exposing the device to the laser light described above. The columns are the datapoint number, the time ins seconds, the gate voltage, the gate current, the source-drain voltage and the source-drain current.
