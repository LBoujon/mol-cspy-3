"""
Module to aid in the creation of supercells from `Crystal` objects in a consistent
manner. For example doubling along the shortest unit cell length.

The supercell creation method classes (GrowShortestMethod, DoubleShortestMethod,
SplitMethod, PrimeFactMethod) are all subclasses of SuperCellsCreatorBC. If you want to implement
your own supercell creation method it is recommended that you create a subclass of
SuperCellsCreatorBC and implement the required methods (see SuperCellsCreatorBC docs
to know what must be implemented).

For convenience, the function `create_supercells` is made available. It can be used
to create supercells from a list of `Crystal` objects.

```python
crystals: List[Crystal] = ...
supercells = create_supercells(crystals, niggli=True, target=8) # Apply Niggli reduction and target 8 molecules in the supercell
assert isintance(supercells, list) # The function returns None if supercells of 8 molecules
                                   # can't be created for all crystals
```

Classes
-------

- `SuperCellsCreatorBC`: Base class for supercell creator classes
- `GrowShortestMethod`: Supercell creation method
- `DoubleShortestMethod`: Supercell creation method
- `SplitMethod`: Supercell creation method
- `PrimeFactMethod`: Supercell creation method

Functions
---------

- `create_supercells`: Creates supercells of the crystals passed to the
function.

Named Tuples
------------

- `SuperCell`: To store the supercell size (n, m, l)

Constants
---------

- `METHOD_CLASSES`: A dictionary with keys as supercell creation methods and
values being the class associated with that method. It contains the 4 supercell
creation methods included in this module
"""

from cspy import Crystal
import sympy.ntheory
import logging
from pathlib import Path
from typing import List, Tuple, Union, Iterable, Dict, Type
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from collections import namedtuple


LOG = logging.getLogger(__name__)


SuperCell = namedtuple("SuperCell", ["n", "m", "l"])


class SuperCellsCreatorBC():
    """
    Base class for supercell generation methods.

    Usage
    -----

    To create your own supercell creator class first create a 
    class that inherits this one:
    
    ```
        class SuperCellsCreatorExample(SuperCellsCreatorBC):
            ...
    ```

    And then implement the methods `_supercell_calculator` for the 
    class. This method must accept the arguments `(self, target)`, where target
    is an integer that represents the target number of molecules that the supercell
    must have, and the crystals can be accessed with `self.crystals`. The method 
    returns `Union[List[SuperCell], None]`, meaning that it returns
    the supercell values if all crystals can create supercells with the target number of
    molecules and if one or more cannot create a supercell with that target number of
    molecules it will return None.

    > If the supercells can be created for all crystals, this method **MUST**
    > update the self.supercells attribute with the new valid
    > supercells. `self.supercells` is of type `Union[List[SuperCell], None]`

    Depending on the method you are implementing you might need to pass
    more data to the `__init__` method of the class, in this case remember to 
    call `super().__init__(crystals, niggli)` within your new `__init__` method
    so that the class data is created correctly.
    """
    def __init__(self, crystals: List[Crystal], niggli: bool = False) -> None:
        """
        Remember to call `super().__init__(crystals, niggli)`

        Arguments
        ---------

        crystals : List[Crystals]
            The crystals to check for supercells

        niggli : bool
            Apply the niggli cell reduction to the input crystals. Default is False
        """
        if niggli:
            niggli_crystals = []
            LOG.info("Creating niggli cells of all crystals")
            for crystal in crystals:
                niggli_crystal = crystal.as_primitive_P1()
                with TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    niggli_crystal.save(tmpdir_path / "tmp.res")
                    niggli_crystal: Crystal = Crystal.load(tmpdir_path / "tmp.res")
                niggli_crystal.titl = f"{crystal.titl}-niggli"
                niggli_crystals.append(niggli_crystal)
            self.crystals = niggli_crystals
        else:
            self.crystals = crystals
        self.method = "supercell_creation_method"
        
        # get the number of molecules in each crystal unit cell
        self.num_molecules: List[int] = []
        self.min_num_molecules: int = 1e10
        self.max_num_molecules: int = -1e10
        for crystal in self.crystals:
            mols = len(crystal.unit_cell_molecules())
            if not mols > 0:
                raise ValueError(f"{crystal.titl} has an invalid number of molecules: {mols}")
            LOG.debug(f"Counted {mols} molecules in {crystal.titl}")

            self.num_molecules.append(mols)
            if mols < self.min_num_molecules: self.min_num_molecules = mols
            if mols > self.max_num_molecules: self.max_num_molecules = mols
        self.unique_num_molecules = set(self.num_molecules)
        # get lcm of num of molecules
        self.lcm: int = np.lcm.reduce(self.num_molecules)
        self.supercells: Union[List[SuperCell], None] = None

    
    def _supercell_calculator(self, target: int) -> Union[List[SuperCell], None]:
        """
        Method to create supercells with `target` number of 
        molecules of `self.crystals`. It is not always possible to create the supercells
        for all crystals, in that case it returns `None`.

        > If the supercells can be created for all crystals, this method **MUST**
        > update the self.supercells attribute with the new valid
        > supercells. `self.supercells` is of type `Union[List[SuperCell], None]`.

        Arguments
        ---------

        target : int
            The target number of molecules in the supercell

        Returns
        -------

        Union[List[SuperCell], None]
            A list of SuperCell objects with the valid supercells for the contents
            of `self.crystals`, or None in case that not all crystals can have supercells
            with `target` number of molecules
        """
        raise NotImplementedError("Implement method to create the supercells")


    def check_target(self, target: int) -> Union[List[SuperCell], None]:
        """
        Check if a supercell with `target` number of molecules
        can be created with all crystals.

        Arguments
        ---------

        target : int
            Target number of molecules in supercells

        Returns
        -------

        Union[List[SuperCell], None]
            A list of SuperCell objects with the valid supercells for the contents
            of `self.crystals`, or None in case that not all crystals can have supercells
            with `target` number of molecules
        """
        # if target is not divisible by num mols of crystals 
        # it will definitely not be possible to create supercell
        for mols in self.unique_num_molecules:
            if target % mols != 0:
                return None
            
        supercells_res = self._supercell_calculator(target)

        if not supercells_res is None:
            for i, supercell_res in enumerate(supercells_res):
                assert np.prod(supercell_res)*self.num_molecules[i] == target
        return supercells_res
    

    def check_targets(self, start: int, end: int, increment: int = 1) -> Iterable[Tuple[int, Union[List[SuperCell], None]]]:
        """
        Check a range of target molecules in supercells

        Arguments
        ---------

        start : int
            Start target value

        end : int
            End target value

        increment : int
            Increase of target value, default is 1

        Returns
        -------

        Iterable[Tuple[int, Union[List[SuperCell], None]]]
            A tuple with the target number of molecules and the valid supercells or None
        """
        if end < start:
            raise ValueError(f"start ({start}) must be larger than end ({end})")
        
        if increment <= 0:
            raise ValueError(f"increment ({increment}) must be larger than 0")
        
        for target in range(start, end+1, increment):
            yield target, self.check_target(target)

    
    def get_P1_supercell_crystals(self) -> List[Crystal]:
        """
        Generate P1 supercells of `self.crystals` using the
        contents of `self.supercells`.

        Returns
        -------

        p1_crystals : List[Crystal]
            The P1 supercell crystals

        Raises
        ------

        ValueError
            When `self.supercells` is not a valid List[SuperCell]. Populate it
            by calling the `self.check_target()` or `self.check_targets()` methods
        """
        if self.supercells is None:
            raise ValueError("No valid supercells stored")

        p1_crystals: List[Crystal] = []
        for crystal, supercell in zip(self.crystals, self.supercells):
            supercell_crystal = crystal.as_P1_supercell(supercell)
            supercell_mols = len(supercell_crystal.unit_cell_molecules())
            original_mols = len(crystal.unit_cell_molecules())
            assert supercell_mols == original_mols * np.prod(supercell), \
                f"CRITICAL! Number of molecules ({supercell_mols}) does not match expected value ({original_mols * np.prod(supercell)}) in supercell of {crystal.titl}. " \
                "Something has gone wrong in creation of supercell"
            p1_crystals.append(supercell_crystal)
        
        return p1_crystals
    

    def save_mutliple_results_to_csv(
            self, 
            results: Iterable[Tuple[int, Union[List[SuperCell], None]]],
            file_ending: str,
        ) -> None:
        """
        Save the results of multiple target molecule counts to .csv
        files.

        Arguments
        ---------

        results : Iterable[Tuple[int, Union[List[SuperCell], None]]]
            a tuple of target molecules and the possible valid supercells

        file_ending : str
            Append this to the output filename: "{self.method}_{file_ending}", 
            the '.csv' extension will be added if it is missing
        """
        if not file_ending.endswith(".csv"):
            file_ending += ".csv"
        file_name = f"{self.method}_{file_ending}"

        dict_res = {
            "mols": [],
            "supercells": [],
        }
        for mols, supercells in results:
            if not supercells is None:
                dict_res["mols"].append(mols)
                dict_res["supercells"].append(supercells)

        df = pd.DataFrame(data = dict_res)
        df.to_csv(file_name, index_label="idx")
        LOG.info(f"Results saved to file: {file_name}")


    @classmethod
    def cell_length(cls, initial_cell: List[float], supercell: SuperCell) -> List[float]:
        """
        Calculate the length of the cell sides of a supercell.

        Arguments
        ---------

        initial_cell : List[float]
            The cell lengths of the unit cell

        supercell : SuperCell
            The size of the supercell

        Returns
        -------

        List[float]
            The length of the supercell sides
        """
        return [a*b for a, b in zip(initial_cell, supercell)]


class GrowShortestMethod(SuperCellsCreatorBC):
    """
    Create supercells by adding a unit cell iteratively
    along the direction of shortest supercell side.
    """
    def __init__(self, crystals: List[Crystal], niggli: bool) -> None:
        super().__init__(crystals, niggli)
        self.method = "grow_shortest"


    def _supercell_calculator(self, target: int) -> Union[List[SuperCell], None]:
        supercells: List[SuperCell] = []
        for idx, crystal in enumerate(self.crystals):
            LOG.debug(f"Checking {crystal.titl}")
            crystal_mols = self.num_molecules[idx]
            LOG.debug(f"Initial number of molecules in crystal: {crystal_mols}")

            cell_lengths: List[float] = list(crystal.unit_cell.lengths)
            initial_cell_lengths: Tuple[float, float, float] = crystal.unit_cell.lengths
            LOG.debug(f"Initial cell lengths: {cell_lengths}")

            supercell = [1, 1, 1] # to keep track of changes
            mol_count = crystal_mols
            while mol_count < target:
                smallest = min(cell_lengths)
                smallest_idx = cell_lengths.index(smallest)
                LOG.debug(f"Smallest cell length is {smallest} in position {smallest_idx}")

                # grow the length of the smallest
                cell_lengths[smallest_idx] += initial_cell_lengths[smallest_idx]
                supercell[smallest_idx] += 1
                LOG.debug(f"New cell lengths after iteration: {cell_lengths}")
                
                mol_count = np.prod(supercell)*crystal_mols
                LOG.debug(f"Number of molecules in current supercell: {mol_count}")

            if mol_count > target:
                LOG.debug(f"Growing by shortest side of unit cell cannot create supercell with {target} molecules in crystal {crystal.titl}")
                self.supercells = None
                return None
            
            supercells.append(SuperCell(supercell[0], supercell[1], supercell[2]))

        self.supercells = supercells
        return supercells
    

class DoubleShortestMethod(SuperCellsCreatorBC):
    """
    Create supercells by doubling the length along the 
    direction of shortest supercell side.
    """
    def __init__(self, crystals: List[Crystal], niggli: bool) -> None:
        super().__init__(crystals, niggli)
        self.method = "double_shortest"


    def _supercell_calculator(self, target: int) -> Union[List[SuperCell], None]:
        supercells: List[SuperCell] = []
        for idx, crystal in enumerate(self.crystals):
            LOG.debug(f"Checking {crystal.titl}")
            crystal_mols = self.num_molecules[idx]
            LOG.debug(f"Initial number of molecules in crystal: {crystal_mols}")

            cell_lengths: List[float] = list(crystal.unit_cell.lengths)
            initial_cell_lengths: Tuple[float, float, float] = crystal.unit_cell.lengths
            LOG.debug(f"Initial cell lengths: {cell_lengths}")

            supercell = [1, 1, 1] # to keep track of changes
            mol_count = crystal_mols
            while mol_count < target:
                smallest = min(cell_lengths)
                smallest_idx = cell_lengths.index(smallest)
                LOG.debug(f"Smallest cell length is {smallest} in position {smallest_idx}")

                # grow the length of the smallest
                cell_lengths[smallest_idx] *= 2
                supercell[smallest_idx] *= 2
                LOG.debug(f"New cell lengths after iteration: {cell_lengths}")
                
                mol_count = np.prod(supercell)*crystal_mols
                LOG.debug(f"Number of molecules in current supercell: {mol_count}")

            if mol_count > target:
                LOG.debug(f"Growing by shortest side of unit cell cannot create supercell with {target} molecules in crystal {crystal.titl}")
                self.supercells = None
                return None
            
            supercells.append(SuperCell(supercell[0], supercell[1], supercell[2]))

        self.supercells = supercells
        return supercells


class SplitMethod(SuperCellsCreatorBC):
    """
    Create supercells by starting from a supercell along
    the shortest axis and splitting along the longest length.
    """
    def __init__(self, crystals: List[Crystal], niggli: bool) -> None:
        super().__init__(crystals, niggli)
        self.method = "split"


    def _supercell_calculator(self, target: int) -> Union[List[SuperCell], None]:
        """
        Method to create supercells with `target` number of 
        molecules. It is not always possible to create the supercells
        for all crystals, in that case it returns `None`.
        """
        supercells: List[SuperCell] = []
        for idx, crystal in enumerate(self.crystals):
            crystal_mols = self.num_molecules[idx]
            LOG.debug(f"Initial number of molecules in crystal: {crystal_mols}")

            ini_cell_lengths: List[float] = list(crystal.unit_cell.lengths)
            LOG.debug(f"Initial cell lengths: {ini_cell_lengths}")

            # initial number of cells
            ini_cells = int(target / crystal_mols)
            LOG.debug(f"Initial number of cells: {ini_cells}")

            # grow along shortest length initially
            smallest = min(ini_cell_lengths)
            smallest_idx = ini_cell_lengths.index(smallest)
            LOG.debug(f"Smallest cell length is {smallest} in position {smallest_idx}")
            supercell = [1, 1, 1]
            supercell[smallest_idx] *= ini_cells
            cell_lengths = self.cell_length(ini_cell_lengths, supercell)

            if ini_cells % 2 == 1:
                LOG.debug(f"Initial number of cells is odd so can't subdivide")
                previous_supercell = supercell
            else:
                previous_diff = 1e100
                diff = max(cell_lengths) - min(cell_lengths)
                while diff < previous_diff:
                    previous_supercell = supercell
                    previous_diff = diff

                    largest = max(cell_lengths)
                    largest_idx = cell_lengths.index(largest)
                    largest_cell = supercell[largest_idx]
                    smallest = min(cell_lengths)
                    smallest_idx = cell_lengths.index(smallest)
                    # smallest_cell = supercell[smallest_idx]
                    # other_idx = (smallest_idx + largest_idx) % 3

                    if largest_cell % 2 == 0:
                        supercell[largest_idx] /= 2
                        supercell[largest_idx] = int(supercell[largest_idx])
                        supercell[smallest_idx] *= 2
                    else:
                        # odd number of cells in longest direction
                        pass
                        
                    cell_lengths = self.cell_length(ini_cell_lengths, supercell)
                    diff = max(cell_lengths) - min(cell_lengths)

            mol_count = np.prod(supercell)*crystal_mols
            if mol_count > target:
                LOG.debug(f"Growing by splitting along longest side of unit cell cannot create supercell with {target} molecules in crystal {crystal.titl}")
                self.supercells = None
                return None

            supercells.append(SuperCell(previous_supercell[0], previous_supercell[1], previous_supercell[2]))

        self.supercells = supercells
        return supercells


class PrimeFactMethod(SuperCellsCreatorBC):
    """
    Create supercells similar to the `SplitMethod` class
    but using prime factorization. Not very well tested.
    """
    def __init__(self, crystals: List[Crystal], niggli: bool) -> None:
        super().__init__(crystals, niggli)
        self.method = "prime_fact"

    
    def _find_prime_factorisation(self, target: int) -> Union[List[List[int]], None]:
        values = []
        for mols in self.unique_num_molecules:
            if target % mols == 0:
                div = target // mols
            else:
                return None

            prime_factors: List[int] = sympy.ntheory.factorint(div, multiple=True)

            if len(prime_factors) < 3:
                for _ in range(3-len(prime_factors)):
                    prime_factors.append(1)
            assert len(prime_factors) >= 3

            values.append(prime_factors)

        return values


    def _supercell_calculator(self, target: int) -> Union[List[SuperCell], None]:
        prime_facts_list = self._find_prime_factorisation(target)
        if prime_facts_list is None:
            return None
        
        supercells: List[SuperCell] = []
        for crystal in self.crystals:
            unit_cell_sizes: List[float] = crystal.unit_cell.lengths
            num_mols = len(crystal.unit_cell_molecules())

            max_length = max(unit_cell_sizes)
            ratios = [max_length/length for length in unit_cell_sizes]

            idx = list(self.unique_num_molecules).index(num_mols)
            prime_facts = prime_facts_list[idx]
            descending_prime_facts = sorted(prime_facts, reverse=True)
            LOG.debug(f"Prime factorization: {prime_facts}")

            supercell: Dict[int, List[int]] = {}
            supercell[0] = [1]
            supercell[1] = [1]
            supercell[2] = [1]
            for prime_fact in descending_prime_facts:
                max_num = max([max(supercell[0]), max(supercell[1]), max(supercell[2])])
                current_ratios = [max_num/np.prod(supercell[0]), max_num/np.prod(supercell[1]), max_num/np.prod(supercell[2])]
                ratio_diffs = [abs(a-b) for a,b in zip(ratios, current_ratios)]
                idx_max = ratio_diffs.index(max(ratio_diffs))
                supercell[idx_max].append(prime_fact)

            supercell = [np.prod(supercell[0]), np.prod(supercell[1]), np.prod(supercell[2])]
            supercells.append(SuperCell(supercell[0], supercell[1], supercell[2]))

        self.supercells = supercells
        return supercells


METHOD_CLASSES: Dict[str, Type[SuperCellsCreatorBC]] = {
    "grow_shortest": GrowShortestMethod,
    "double_shortest": DoubleShortestMethod,
    "split": SplitMethod,
    "prime_fact": PrimeFactMethod,
}


def create_supercells(crystals: Iterable[Crystal], target: Union[int, None] = None, 
                      niggli: bool = False, creation_method: Type[SuperCellsCreatorBC] = DoubleShortestMethod) -> Union[List[Crystal], None]:
    """
    Create supercells of the passed `crystals` with `target` number
    of molecules. If the creation of such supercell fails for any of the passed
    crystals, `None` is returned.

    If no `target` number of molecules in the supercell is passed, the LCM of the 
    number of molecules in the passed crystals is selected as the target.

    Arguments
    ---------

    crystals : Iterable[Crystal]
        The crystals from which to create supercells

    target : Union[int, None]
        The target number of molecules in the resulting supercells. If set to None
        the LCM of the number of molecules in `crystals` is set as the target. Default is 
        None

    niggli : bool
        Create Niggli reduced cells of the crystals before doing the supercells. Niggli
        reduction is also done before calculation of the LCM. Default is False

    creation_method : Type[SuperCellsCreatorBC]
        The class defining the method for creating the supercells. Default is DoubleShortestMethod

    Returns
    -------

    p1_supercells : Union[List[Crystal], None]
        A list of P1 crystal supercells, or `None` if the supercells could not be created
    """
    LOG.debug(f"Supercell creation method selected: {creation_method.__name__}")
    supercell_creator = creation_method(crystals, niggli=niggli)
    if target is None:
        target = supercell_creator.lcm
    LOG.debug(f"Checking if creation of supercells of {target} number of molecules is possible for all crystals passed")
    supercells = supercell_creator.check_target(target)
    if supercells is None:
        LOG.info(f"Unable to create supercells of {target} molecules for all crystals")
        return None
    
    return supercell_creator.get_P1_supercell_crystals()

