class MolBuilderException(Exception):
    """A general exception for molbuilder errors.

    """


class PropertyCalculationFailure(MolBuilderException):
    """Property calculation failure exception.

    """


class CarrierNotCorrectlyDefined(MolBuilderException):
    """Carrier not correctly defined for reorganization energy
    calculation.

    """


class UFFOptimizationFailure(PropertyCalculationFailure):
    """RDKit UFF optimization failure exception.

    """


class GaussianCalculationFailure(PropertyCalculationFailure):
    """Gaussian calculation failure exception.

    """


class FormchkFailure(GaussianCalculationFailure):
    """formchk failure exception.

    """


class CP2KCalculationFailure(PropertyCalculationFailure):
    """CP2K calculation failure exception.

    """


class AOMCalculationFailure(PropertyCalculationFailure):
    """AOM calculation failure exception.

    """


class DimerXYZFileContainsOddNumberOfAtoms(AOMCalculationFailure):
    """Unable to write AOM input because there was an odd number
    of atoms.

    """


class MoleculeModificationFailure(MolBuilderException):
    """Molecule Creation Failure exception.

    """


class FragmentAdditionFailure(MoleculeModificationFailure):
    """Fragment Addition Failure exception.

    """


class MoleculeMutationFailure(MoleculeModificationFailure):
    """Molecule Mutation Failure exception.

    """


class MoleculeFragmentationFailure(MoleculeModificationFailure):
    """Molecule Fragmentation Failure exception.

    """


class MoleculeRecombineFailure(MoleculeModificationFailure):
    """Molecule Recombine Failure exception.

    """


class MobilityCalculationFailure(MolBuilderException):
    """Mobility calculation failure exception.

    """
