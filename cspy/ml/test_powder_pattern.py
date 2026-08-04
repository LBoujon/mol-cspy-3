from unittest.mock import Mock

import numpy as np

from cspy.ml.descriptors import calculate_crystal_powder_pattern


def test_configured_pymatgen_pattern_is_returned_as_an_array():
    crystal = Mock()
    crystal.calculate_powder_pattern.return_value = Mock(
        pattern=np.array([1.0, 2.0, 3.0])
    )

    result = calculate_crystal_powder_pattern(crystal, "pymatgen")

    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    crystal.calculate_powder_pattern.assert_called_once_with(method="pymatgen")


def test_descriptor_failure_keeps_the_crystal_usable(caplog):
    crystal = Mock()
    crystal.calculate_powder_pattern.side_effect = RuntimeError("broken CIF")

    result = calculate_crystal_powder_pattern(crystal, "pymatgen")

    assert result is None
    assert "keeping structure without the descriptor" in caplog.text


def test_disabled_descriptor_does_not_calculate_a_pattern():
    crystal = Mock()

    assert calculate_crystal_powder_pattern(crystal, None) is None
    crystal.calculate_powder_pattern.assert_not_called()
