# from the output of CP2K

#  Speed of light in vacuum [m/s]                             2.99792458000000E+08
#  Magnetic constant or permeability of vacuum [N/A**2]       1.25663706143592E-06
#  Electric constant or permittivity of vacuum [F/m]          8.85418781762039E-12
#  Planck constant (h) [J*s]                                  6.62606896000000E-34
#  Planck constant (h-bar) [J*s]                              1.05457162825177E-34
#  Elementary charge [C]                                      1.60217648700000E-19
#  Electron mass [kg]                                         9.10938215000000E-31
#  Electron g factor [ ]                                     -2.00231930436220E+00
#  Proton mass [kg]                                           1.67262163700000E-27
#  Fine-structure constant                                    7.29735253760000E-03
#  Rydberg constant [1/m]                                     1.09737315685270E+07
#  Avogadro constant [1/mol]                                  6.02214179000000E+23
#  Boltzmann constant [J/K]                                   1.38065040000000E-23
#  Atomic mass unit [kg]                                      1.66053878200000E-27
#  Bohr radius [m]                                            5.29177208590000E-11
#
#  *** Conversion factors ***
#
#  [u] -> [a.u.]                                              1.82288848426455E+03
#  [Angstrom] -> [Bohr] = [a.u.]                              1.88972613288564E+00
#  [a.u.] = [Bohr] -> [Angstrom]                              5.29177208590000E-01
#  [a.u.] -> [s]                                              2.41888432650478E-17
#  [a.u.] -> [fs]                                             2.41888432650478E-02
#  [a.u.] -> [J]                                              4.35974393937059E-18
#  [a.u.] -> [N]                                              8.23872205491840E-08
#  [a.u.] -> [K]                                              3.15774647902944E+05
#  [a.u.] -> [kJ/mol]                                         2.62549961709828E+03
#  [a.u.] -> [kcal/mol]                                       6.27509468713739E+02
#  [a.u.] -> [Pa]                                             2.94210107994716E+13
#  [a.u.] -> [bar]                                            2.94210107994716E+08
#  [a.u.] -> [atm]                                            2.90362800883016E+08
#  [a.u.] -> [eV]                                             2.72113838565563E+01
#  [a.u.] -> [Hz]                                             6.57968392072181E+15
#  [a.u.] -> [1/cm] (wave numbers)                            2.19474631370540E+05
#  [a.u./Bohr**2] -> [1/cm]                                   5.14048714338585E+03

# Energy
Ha_to_eV = 2.72113838565563e1
eV_to_Ha = 1/Ha_to_eV

# Length
Bohr_to_Angstrom = 5.29177208590000e-1
Angstrom_to_Bohr = 1/Bohr_to_Angstrom

# Force
Ha_B_to_eV_A = Ha_to_eV / Bohr_to_Angstrom
eV_A_to_Ha_B = 1/Ha_B_to_eV_A