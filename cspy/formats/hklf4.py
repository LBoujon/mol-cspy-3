import numpy as np
from cspy.util.path import Path


class Hklf4:
    def __init__(self, contents, **kwargs):
        self.lines = contents.split("\n")
        self.parse()

    def parse(self):
        hkl = []
        intensities = []
        for line in self.lines:
            if line.startswith("_"):
                break
            if line:
                tokens = line.strip().split()
                hkl.append([int(x) for x in tokens[:3]])
                intensities.append([float(x) for x in tokens[3:]])
        self.hkl = np.array(hkl)
        self.intensities = np.array(intensities)

    def d(self, reciprocal):
        return np.linalg.norm(np.dot(self.hkl, reciprocal), axis=1)

    def dhist(self, reciprocal, bins=100):
        return np.histogram(
            self.d(reciprocal),
            weights=self.intensities[:, 1],
            bins=bins,
            density=True,
            range=(0, 1.3),
        )

    @classmethod
    def from_file(cls, filename):
        return cls(Path(filename).read_text())
