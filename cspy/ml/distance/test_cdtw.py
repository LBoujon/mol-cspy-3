from cspy.ml.distance import cdtw_distance
import numpy as np


def test_zeros():
    z1 = np.zeros(1000)
    z2 = np.zeros(1000)
    diff = cdtw_distance(z1, z2, 0)
    assert diff == 0.0


if __name__ == "__main__":
    test_zeros()
