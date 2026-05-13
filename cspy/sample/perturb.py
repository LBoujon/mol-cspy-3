import logging
import numpy as np

LOG = logging.getLogger(__name__)


class Perturb:
    """
    Perturb class based on Crystal object
    """

    def __init__(self, crystal_system):
        self.constrain = constrain
        self.p_types = {
            "translation": self.translation,
            "rotation": self.rotation,
            "cell_length": self.cell_length,
            "cell_angle": self.cell_angle,
            "volume": self.volume,
        }

    def translation(self, crystal, vector):
        return crystal

    def rotation(self, crystal, vector):
        return crystal

    def cell_length(self, crystal, vector):
        return crystal

    def cell_angle(self, crystal, vector):
        return crystal

    def volume(self, crystal, vector):
        return crystal

    def __call__(self, crystal, perturb_type, vector):
        return self.p_types[perturb_type](crystal, vector)
