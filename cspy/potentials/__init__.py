__all__ = ["PotentialData", "available_potentials"]

import os

_DIRNAME = os.path.dirname(__file__)
from .potential_data import PotentialData

available_potentials = {
    "fit": ("F", os.path.join(_DIRNAME, "fit.pots")),
    "fit_disponly": ("F", os.path.join(_DIRNAME, "fit_disponly.pots")),
    "fit_reponly": ("F", os.path.join(_DIRNAME, "fit_reoponly.pots")),
    "w99": ("W", os.path.join(_DIRNAME, "w99.pots")),
    "fit_water_X": ("F", os.path.join(_DIRNAME, "fit_water_X.pots")),
    "Day_halobenzenes": ("C", os.path.join(_DIRNAME, "Day_halobenzenes.pots")),
    "w99_orig_Halogens": ("W", os.path.join(_DIRNAME, "w99_orig_Halogens.pots")),
    "w99_orig_H": ("W", os.path.join(_DIRNAME, "w99_orig_H.pots")),
    "w99rev_6311": ("W", os.path.join(_DIRNAME, "w99rev_6311.pots")),
    "w99rev_6311_s": ("W", os.path.join(_DIRNAME, "w99rev_6311_s.pots")),
    "w99rev_631": ("W", os.path.join(_DIRNAME, "w99rev_631.pots")),
    "w99rev_pcm_6311": ("W", os.path.join(_DIRNAME, "w99rev_pcm_6311.pots")),
    "w99_s_cl": ("W", os.path.join(_DIRNAME, "w99_s_cl.pots")),
    "w99sp": ("W", os.path.join(_DIRNAME, "w99sp.pots")),
    "w99rev_pcm_6311_and_Halides": ("W", os.path.join(_DIRNAME, "w99rev_pcm_6311_and_Halides.pots")),
    "isoPAHAP": ("F", os.path.join(_DIRNAME, "isoPAHAP.pots")), # From: https://doi.org/10.1039/C2CP23008A. Tested on pyrene, phenanthrene and perylene  
    "PAHAP": ("F", os.path.join(_DIRNAME, "PAHAP.pots")), # From: https://doi.org/10.1021/ct9004883. Tested on pyrene, phenanthrene and perylene
    "nothing": ("F", os.path.join(_DIRNAME, "nothing.pots")),
    "gaff2_LJ": ("GAFF", os.path.join(_DIRNAME, "GULP/", "gaff2_LJ.lib")), # General Amber Force Field version 2 - as bundled with GULP version 6.2
    "gaff2_fit": ("GAFF", os.path.join(_DIRNAME, "GULP/", "gaff2_fit.lib")), # As above but with LJ swapped for Fit (Piracetam only atm)
}
