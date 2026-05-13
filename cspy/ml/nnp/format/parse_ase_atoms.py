import ase
from cspy import Crystal, Molecule
from cspy.chem import Element
from typing import Union


def to_ase_atoms(structure: Union[Crystal, Molecule], **kwargs) -> ase.Atoms:
    """Convert cspy objects to ase objects.

    Args:
        structure: cspy Crystal or Molecule object to convert.

    Returns:
        ase.Atoms: ASE Atoms object corresponding to the cspy object.
    """
    if isinstance(structure, Crystal):
        return cspy_crystal_to_ase_atoms(structure, **kwargs)
    if isinstance(structure, Molecule):
        return cspy_molecule_to_ase_atoms(structure, **kwargs)
    raise NotImplementedError(
        f"Unable to convert object of type: {type(structure)}"
    )


def from_ase_atoms(atoms: ase.Atoms, **kwargs) -> Union[Crystal, Molecule]:
    """Convert ase atoms objects to cspy objects.

    Args:
        atoms (ase.Atoms): ase Atoms object to convert

    Returns:
        Union[Crystal, Molecule]: cspy Crystal or Molecule object corresponding to the ase Atoms object.
    """
    assert isinstance(atoms, ase.Atoms), "Not an ase Atoms object"

    if atoms.pbc.all():
        return ase_crystal_to_cspy_crystal(atoms, **kwargs)
    elif not atoms.pbc.any():
        return ase_molecule_to_cspy_molecule(atoms, **kwargs)
    else:
        raise NotImplementedError(
            "Unable to convert intermediate periodic structure"
        )


def cspy_crystal_to_ase_atoms(crystal: Crystal, **kwargs) -> ase.Atoms:
    """Convert cspy crystal object to ASE Atoms object. NOTE: ase Atoms objects are P1.

    Args:
        crystal (cspy.Crystal): cspy Crystal object to convert.

    Returns:
        ase.Atoms: ASE Atoms object corresponding to the cspy Crystal object.
    """
    unit_cell_atoms = crystal.unit_cell_atoms()
    symbols = [
        Element.from_atomic_number(x).symbol
        for x in unit_cell_atoms["element"]
    ]

    return ase.Atoms(
        symbols=symbols,
        positions=unit_cell_atoms["cart_pos"],
        cell=crystal.unit_cell.lattice,
        pbc=True,
        **kwargs
    )


def cspy_molecule_to_ase_atoms(molecule: Molecule, **kwargs) -> ase.Atoms:
    """Convert cspy Molecule object to ASE Atoms object.

    Args:
        molecule (cspy.Molecule): cspy Molecule object to convert.

    Returns:
        ase.Atoms: ASE Atoms object corresponding to the cspy Molecule object.
    """
    return ase.Atoms(
        symbols=[e.symbol for e in molecule.elements],
        positions=molecule.positions,
        pbc=False,
        **kwargs
    )

def ase_crystal_to_cspy_crystal(atoms: ase.Atoms, **kwargs) -> Crystal:
    """Convert ASE Atoms object to cspy Crystal.

    Args:
        atoms (ase.Atoms): ASE Atoms object to convert.

    Returns:
        cspy.Crystal: cspy Crystal object corresponding to the ASE Atoms object.
    """
    from cspy.crystal import UnitCell, AsymmetricUnit, SpaceGroup

    unit_cell = UnitCell(atoms.cell)
    asymmetric_unit = AsymmetricUnit(
        [Element.from_atomic_number(x) for x in atoms.get_atomic_numbers()],
        atoms.get_scaled_positions(),
    )
    space_group = SpaceGroup(1)

    return Crystal(
        unit_cell=unit_cell,
        asymmetric_unit=asymmetric_unit,
        space_group=space_group,
        **kwargs,
    )


def ase_molecule_to_cspy_molecule(atoms: ase.Atoms, **kwargs) -> Molecule:
    """Convert ASE Atoms object to cspy Molecule.

    Args:
        atoms (ase.Atoms): ASE Atoms object to convert.

    Returns:
        cspy.Molecule: cspy Molecule object corresponding to the ASE Atoms object.
    """
    return Molecule(
        positions=atoms.get_positions(),
        elements=[
            Element.from_atomic_number(x)
            for x in atoms.get_atomic_numbers()
        ],
        **kwargs
    )


def from_extended_xyz(fpath: str, **kwargs) -> Union[Crystal, Molecule]:
    """Read an extended XYZ file and convert it to a cspy Crystal or Molecule object.

    Args:
        fpath (str): Path to the extended XYZ file.
        **kwargs: Additional keyword arguments to pass to the conversion function.

    Returns:
        Union[Crystal, Molecule]: cspy Crystal or Molecule object corresponding to the extended XYZ file.
    """
    from ase import io
    import os
    from cspy.ml.nnp.format.parse_ase_atoms import from_ase_atoms

    name = "_".join(os.path.basename(fpath).split(".")[:-1])
    atoms = io.read(fpath, format="extxyz")

    return from_ase_atoms(atoms, titl=name, **kwargs)


def to_extended_xyz(
    crystal: Crystal,
    outname: str = None,
    **kwargs
) -> None:
    """Convert a cspy Crystal object to an extended XYZ file.

    Args:
        crystal (Crystal): cspy Crystal object to convert.
        outname (str, optional): Name of the output file. Defaults to None, which uses the crystal title.
        **kwargs: Additional keyword arguments to pass to the conversion function.

    Returns:
        None: The function writes the extended XYZ file to disk.
    """
    from cspy.ml.nnp.format.parse_ase_atoms import to_ase_atoms
    from ase.calculators.calculator import Calculator

    outname = crystal.titl if outname is None else outname
    atoms = to_ase_atoms(crystal, **kwargs)
    atoms.calc = Calculator()
    if "lattice_energy" in crystal.properties:
        energy = crystal.properties["lattice_energy"]
    elif "energy" in crystal.properties:
        energy = crystal.properties["energy"]
    else:
        energy = None
    atoms.calc.results = {
        "energy": energy,
        "forces": crystal.properties.get("atom_forces", None),
    }
    atoms.write(outname + ".xyz", format="extxyz")
