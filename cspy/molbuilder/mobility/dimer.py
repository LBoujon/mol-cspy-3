import copy
import math
import logging
import itertools as it
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
from cspy.molbuilder.exceptions import MobilityCalculationFailure


LOG = logging.getLogger(__name__)


def active_molecules(crystal, len_target, vdw_buffer):
    """Get the molecules of the unit cell and the molecules
    surrounding it to be used to calculate electron couplings and
    other related information.

    :param crystal: The crystal structure.
    :param len_target: The box length that determines the size of the
        trial supercell that should be created so that it is able to
        contain this box.
    :param vdw_buffer: The vdw buffer used to determine if two
        molecules are in close contact. If the a surrounding molecule
        is in close contact with a unit cell molecule it is included
        in the active molecule list.
    :return: The active molecules, their site indices, full indices
        and the trail expansion used to obtain the active molecules.
    """
    a, b, c = crystal.unit_cell.lengths
    angles = crystal.unit_cell.angles
    ca, cb, cg = np.cos(angles)
    sg = np.sin(angles[2])
    v = crystal.unit_cell.volume()

    max_w = abs(len_target / (v / (a * b * sg)))
    max_v = abs(
        (len_target + abs(max_w * (c * (ca - cb * cg) / sg))) / (b * sg)
    )
    max_u = abs(
        (len_target + abs(max_v * b * cg) + abs(max_w * c * cb)) / a
    )
    max_u = int(math.ceil(max_u))
    max_v = int(math.ceil(max_v))
    max_w = int(math.ceil(max_w))
    u = np.arange(-max_u, max_u + 1)
    v = np.arange(-max_v, max_v + 1)
    w = np.arange(-max_w, max_w + 1)

    LOG.info('Determining depths for surrounding molecules in close '
             'contact trying (%s, %s, %s)', max_u, max_v, max_w)

    uc_mols = crystal.unit_cell_molecules()
    uc_pos = np.vstack([i.positions for i in uc_mols])
    uc_vdws = [[i.vdw for i in mol.elements] for mol in uc_mols]
    uc_vdws = np.array([i for j in uc_vdws for i in j])
    max_vdw = np.max(uc_vdws)
    tree_1 = KDTree(uc_pos)

    a, b, c = 0, 0, 0
    for q, r, s in it.product(u, v, w):
        tran_pos = uc_pos + np.asarray([q, r, s]) @ crystal.unit_cell.direct
        tree_2 = KDTree(tran_pos)
        contacts = tree_1.query_ball_tree(tree_2, max_vdw * 2 + vdw_buffer)
        if any([len(i) != 0 for i in contacts]):
            a = max(a, abs(q))
            b = max(b, abs(r))
            c = max(c, abs(s))

    LOG.info('Determined depths for surrounding molecules in close '
             'contact to be (%s, %s, %s)', a, b, c)
    u = np.arange(-a, a + 1)
    v = np.arange(-b, b + 1)
    w = np.arange(-c, c + 1)

    mols = []
    site_idxs = list(range(len(uc_mols)))
    full_idxs = list(range(len(uc_mols)))
    full_idx = len(uc_mols)
    for q, r, s in it.product(u, v, w):
        if [q, r, s] == [0, 0, 0]:
            continue
        for i, uc_mol in enumerate(uc_mols):
            trans = np.asarray([q, r, s]) @ crystal.unit_cell.lattice
            tran_pos = uc_mol.positions + trans
            tree_2 = KDTree(tran_pos)
            contacts = tree_2.query_ball_tree(tree_1, max_vdw * 2 + vdw_buffer)
            for j, js in enumerate(contacts):
                if len(js) == 0:
                    continue
                diff = tran_pos[j] - uc_pos[js]
                dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                sum_radii = uc_mol.elements[j].vdw + uc_vdws[js] + vdw_buffer
                if np.any(dist < sum_radii):
                    mols.append(uc_mol.translated(trans))
                    site_idxs.append(i)
                    full_idxs.append(full_idx)
                    break
            full_idx += 1

    LOG.info('Found %s surrounding molecules in close contact', len(mols))
    return uc_mols + mols, site_idxs, full_idxs, (a, b, c)


def perturbed_molecules(crystal, idx, axis, magnitude, derivative,
                        depths, full_mol_idxs):
    """Perturb one of the molecules in the unit cell and generate a
    list of perturbed molecules that have indices in the full_idxs list.

    :param crystal: The crystal structure.
    :param idx: Index of the molecule to perturb.
    :param axis: The rotation axis or translation vector to carry out
        the perturbation.
    :param magnitude: The magnitude of the perturbation.
    :param derivative: Specifies whether it will be a rotation or
        translation type perturbation.
    :param depths: The depths to to expand to get the molecules in
        the full_mol_idxs list.
    :param full_mol_idxs: A list of indices of the molecules to return.
    :return: A list of perturbed molecules.
    """
    perturbed_uc_mols = crystal.unit_cell_molecules()
    num_uc_mols = len(perturbed_uc_mols)
    target_mol = perturbed_uc_mols[idx]
    perturbed_mol = copy.deepcopy(target_mol)
    surr_mol_idxs = full_mol_idxs[num_uc_mols:]

    if derivative == 'translation':
        perturbed_mol.translate(magnitude * axis)
    elif derivative == 'rotation':
        axis_x, axis_y, axis_z = axis
        perturbed_mol.translate(-target_mol.center_of_mass)
        pos = perturbed_mol.positions
        pos = pos - (pos @ axis * np.tile(axis, (pos.shape[0], 1)).T).T
        max_dist = np.max(np.sqrt(np.einsum('ij,ij->i', pos, pos)))
        angle = np.arcsin(magnitude / max_dist)
        sin_rota = np.sin(angle / 2)
        rota_matrix = R.from_quat([
            axis_x * sin_rota, axis_y * sin_rota,
            axis_z * sin_rota, np.cos(angle / 2)
        ]).as_matrix()
        perturbed_mol.transform(rota_matrix)
        perturbed_mol.translate(target_mol.center_of_mass)
    else:
        raise MobilityCalculationFailure(
            'Perturbation type %s not recognised.'.format(derivative)
        )
    perturbed_uc_mols[idx] = perturbed_mol

    max_u, max_v, max_w = depths
    u = np.arange(-max_u, max_u + 1)
    v = np.arange(-max_v, max_v + 1)
    w = np.arange(-max_w, max_w + 1)

    i = num_uc_mols
    surr_mols = []
    for q, r, s in it.product(u, v, w):
        if [q, r, s] == [0, 0, 0]:
            continue
        for uc_mol in perturbed_uc_mols:
            if i in surr_mol_idxs:
                surr_mols.append(uc_mol.translated(
                    np.asarray([q, r, s]) @ crystal.unit_cell.lattice
                ))
            i += 1

    return perturbed_uc_mols + surr_mols
