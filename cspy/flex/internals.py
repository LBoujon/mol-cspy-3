from __future__ import print_function
import numpy as np
import copy
from typing import Union, Optional, Self


class Internal(object):
    """A class for describing the internal coordinates of a Molecule that are to be sampled
    during a flexible-molecule mol-dis calculation.

    Args:
        internal (Union[str, list]): Object describing the indices of the internal coordinate
        original_value (float): The original, unmodified value (length, angle, torsion) of the internal coordinate
        starting_value (Optional[float]]): The starting value (length, angle, torsion) of the internal coordinate
        initial_offset (float): The size of the offset to add to the starting_value
        number_of_steps (int): The number of steps used to sample the internal coordinate
        step_size (float): The step size used for each step when sampling the internal coordinate
    
    """
    def __init__(
        self,
        internal: Union[str, list],
        original_value: float,
        starting_value: Optional[float] = None,
        initial_offset: float = 0.0,
        number_of_steps: int = 0,
        step_size: float = 0.0,
    ) -> None:
        if isinstance(internal, str):
            internal = internal.split()
        self.internal = internal
        self.is_angle = len(internal) > 2
        self.original_value = original_value
        self.starting_value = starting_value
        self.initial_offset = initial_offset
        self.number_of_steps = number_of_steps
        self.step_size = step_size
        self.current_step = 0

    def get_value(self, radians: bool = True) -> float:
        """Function that calculates and returns the value of the internal coordinate
        that is to be distorted.

        Args:
            radians (bool, optional): Whether to use radians. Defaults to True.

        Returns:
            float: The value (angle, torsion, e.t.c) of the internal coordinate
        """
        if self.starting_value is not None:
            v = 0.0
        else:
            v = self.original_value
        v = v + self.get_current_offset()
        if self.is_angle:
            v = np.arctan2(np.sin(v), np.cos(v))
        return v

    def get_current_offset(self) -> float:
        """Function to return the current offset value

        Returns:
            float: The current offset value
        """
        v = self.current_step * self.step_size
        if self.starting_value is not None:
            v += -self.original_value + self.starting_value
        v += self.initial_offset
        if self.is_angle:
            v = np.arctan2(np.sin(v), np.cos(v))
        return v

    def get_step_value(self, step: int) -> float:
        """Function to return the value at a given step

        Args:
            step (int): The current sampling step number

        Returns:
            float: The value (angle, torsion, e.t.c) of the internal coordinate at
            the specified step.
        """
        ocs = self.current_step
        self.current_step = 0
        v = self.get_value() + self.step_size * float(step)
        self.current_step = ocs
        return v

    def set_value_as_single_step(self, value: float) -> None:
        """Sets the step size to a single value, while ensuring the number of steps
        and current step are set to 1.

        Args:
            value (float): The step size required for distorting the internal coordinate.
        """
        self.step_size = value
        self.number_of_steps = 1
        self.current_step = 1

    def fix(self) -> None:
        """Fixes the internal coordinate based on the current offset
        """
        self.starting_value = self.get_current_offset() + self.original_value
        self.original_value = self.original_value
        self.step_size = 0.0
        self.current_step = 0.0
        self.number_of_steps = 0
        self.initial_offset = 0.0

    def copy(self) -> Self:
        """Helper function for copying the class instance
        (08/09/25 - Unsure if used outside of this class)

        Returns:
            Self: A deep copy of the class instance
        """
        return copy.deepcopy(self)

    def scan_internals(self) -> list:
        """Function that iterates through each scan point and creates the distorted
        internal coordinate

        Returns:
            list: A list containing distorted internal coordinates from each scan point
        """
        internals = []
        for n in range(0, self.number_of_steps):
            internal = self.copy()
            internal.current_step = n
            internal.fix()
            internals.append(internal)
        return internals

    def __repr__(self) -> str:
        """Function that sets the representation of the Internal class

        Returns:
            str: A string representation of the Internal class
        """
        if self.number_of_steps == 0:
            return " ".join(map(str, self.internal + [self.get_value()]))
        else:
            return " ".join(
                map(
                    str,
                    self.internal
                    + [
                        "->",
                        self.starting_value,
                        "+",
                        self.step_size,
                        "x",
                        self.number_of_steps,
                    ],
                )
            )

    def __str__(self) -> str:
        """Function that sets the string representation of the Internal class

        Returns:
            str: The string representation of the Internal class
        """
        return self.__repr__()

    @property
    def n_points(self):
        """Property that returns the number of sampling points

        Returns:
            int: The number of sampling points
        """
        return self.number_of_steps + 1

