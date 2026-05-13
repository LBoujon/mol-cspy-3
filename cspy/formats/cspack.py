from struct import pack, unpack
from scipy.spatial.transform import Rotation
from cspy.crystal import UnitCell, SpaceGroup, Crystal, AsymmetricUnit
import numpy as np
import logging


LOG = logging.getLogger(__name__)
_PACKED_UNIQUE_PARAMS = (6, 4, 3, 2, 2, 2, 1)
_CELL_TYPES = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "rhombohedral",
    "hexagonal",
    "cubic",
)


def rotation_from_matrix(mat):
    if hasattr(Rotation, "from_matrix"):
        return Rotation.from_matrix(mat)
    else:
        return Rotation.from_dcm(mat)


def pack_header(n, d, i, c):
    header = 0
    flags = 1 if d == np.float64 else 0
    return pack("4B", c, i, flags, n)


def unpack_header(header):
    c = header
    return unpack("4B", header)


def pack_crystal(crystal, numeric_type="f"):
    """Packs a crystal structure into a header, unit cell and
    rotation/translations for each component in asymmetric unit. i.e.

    This assumes all asymmetric unit references are known for unpacking
    i.e. starting positions. as the rotations/translations are relative to them

    | HEADER | UNIT CELL | TRANSFORMATION * N |
      32b      variable    6 * 32b * nmols

    Header is packed as 32 bits as follows:

    I = international tables number
    C = cell type
    N = number of components in asymmetric unit
    F = flags (currently numeric type (0/1 = float/double))

    Transformations are packed as 3 Euler angles + Cartesian translations x y z
    which are floats/doubles depending on numeric_type
    """
    dtype = np.float32 if "f" in numeric_type else np.float64
    d = "f" if dtype == np.float32 else "d"
    cell_type = crystal.unit_cell.cell_type_index
    params = crystal.unit_cell.unique_parameters
    transformations = [
        np.hstack(
            (rotation_from_matrix(m.axes()).inv().as_euler("xyz"), m.center_of_mass)
        ).astype(dtype)
        for m in crystal.symmetry_unique_molecules()
    ]
    n = len(transformations)
    l = len(params)
    transformations = np.reshape(transformations, (6 * n))
    header = pack_header(
        n, dtype, crystal.space_group.international_tables_number, cell_type
    )
    return header + pack(f"{l}{d}{n*6}{d}", *params, *transformations)


def unpack_crystal(data, molecules):
    header, data = data[:4], data[4:]
    c, i, d, n = unpack_header(header)
    dtype = "d" if d else "f"
    l = _PACKED_UNIQUE_PARAMS[c]
    unpacked = unpack(f"{l}{dtype}{6 * n}{dtype}", data)
    params = unpacked[:l]
    transformations = np.reshape(unpacked[l:], (-1, 6))
    uc = UnitCell.from_unique_parameters(params, cell_type=_CELL_TYPES[c])
    sg = SpaceGroup(i)
    labels, elements, positions = [], [], []
    for mol, t in zip(molecules, transformations):
        labels.append(mol.labels)
        elements.append(mol.elements)
        positions.append(Rotation.from_euler("xyz", t[:3]).apply(mol.positions) + t[3:])
    return Crystal(
        uc,
        sg,
        AsymmetricUnit(
            np.hstack(elements),
            uc.to_fractional(np.vstack(positions)),
            labels=np.hstack(labels),
        ),
    )
