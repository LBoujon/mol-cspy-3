__all__ = ["available_n2p2_potentials"]

import os

_DIRNAME = os.path.dirname(__file__)

available_n2p2_potentials = {
    "Faraday2024": {"n2p2_committee_dir" : os.path.join(_DIRNAME, "Faraday2024/committee"),
                    "n2p2_inputnn_filepath" : os.path.join(_DIRNAME, "Faraday2024/input.nn"),
                    "n2p2_scaling_filepath" : os.path.join(_DIRNAME, "Faraday2024/scaling.data")}, # DOI	https://doi.org/10.1039/D4FD00105B : delta-learning potential
}