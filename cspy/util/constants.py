import math

### constants ###
KB = 8.31446e-3  # Boltzmann constant in kJ mol^-1 K^-1
E = 2.7182818284  # Eulers constant
KE = 2.30708e-18  # coloumb constant in N Ang^2 e^(-2)
NA = 6.02214076e23  # Avogadro's number from NIST
PI = math.pi

### conversions ###
J2EV = 6.241509074e18  # J to eV from NIST
KCAL2KJ = 4.18401  # Kcal to J
HA2EV = 27.211386245981  # Ha to eV from NIST
EV2KCAL = 3.82929e-23  # eV to kcal
BOHR2ANGSTROM = 0.52917721092  # Bohr to Angstrom
EV2KJ = (J2EV * 1e3)**-1  # eV to kJ
EV2KJ_PER_MOL = EV2KJ * NA  # eV to kJ / NAs
KCAL2EV = 1/EV2KCAL  # kcal to eV
HA2KJ_PER_MOL = HA2EV * EV2KJ_PER_MOL  # Ha to kJ per mol

# Pressure conversions
EV_ANGSTROM3 = (1/J2EV)*1e30  # eV/Angstrom^3 to N/m^2=Pascal
EV_ANGSTROM3_TO_GPA = EV_ANGSTROM3 / 1e9  # GPa
HA_BOHR3_TO_GPA = HA2EV / BOHR2ANGSTROM**3 * EV_ANGSTROM3_TO_GPA  #GPa

# Maths conversions
DEG2RAD = 180 / PI