from rdkit import Chem
from rdkit.Chem.rdmolops import FindAllPathsOfLengthN
from rdkit.Chem.rdqueries import AtomNumEqualsQueryAtom
from .exceptions import MolBuilderException


def joined_symm_sssr(molecule):
    """Returns a list of atoms indices similar to GetSymmSSSR but for
    cages and rings rather then just rings.

    :param molecule: rdkit Mol object
    :return: a list of atom indices of each joined rings
    """
    symm_sssr = Chem.GetSymmSSSR(molecule)

    cage_atom_idxs_list = []
    for i, ring_atom_idxs_i in enumerate(symm_sssr):

        cage_atom_idxs = set(ring_atom_idxs_i)
        for j, ring_atom_idxs_j in enumerate(symm_sssr):

            if i == j:
                continue

            if len(set(ring_atom_idxs_i).intersection(ring_atom_idxs_j)) > 2:
                cage_atom_idxs = cage_atom_idxs.union(ring_atom_idxs_j)

        if sorted(cage_atom_idxs) not in cage_atom_idxs_list:
            cage_atom_idxs_list.append(sorted(cage_atom_idxs))

    return sorted(cage_atom_idxs_list)


def atom_in_num_joined_symm_sssr(molecule, atom_idx):
    """Returns the number of joined rings that a given atom index is in.

    :param molecule: rdkit Mol object
    :param atom_idx: the atom index
    :return: number of joined rings a atom is in
    """
    num = 0
    for cage_atom_idxs in joined_symm_sssr(molecule):
        if atom_idx in cage_atom_idxs:
            num += 1
    return num


def num_joined_symm_sssr(molecule):
    """Returns the number of joined rings in a molecule.

    :param molecule: rdkit Mol object
    :return: number of joined rings in the molecule
    """
    return len(joined_symm_sssr(molecule))


def remove_smallest_fragment(combo):
    """Removes the smallest fragment from an rdkit Mol object that
    contains two separate molecule fragments.

    :param combo: rdkit Mol object containing two separate
        molecule fragments
    """
    if len(Chem.GetMolFrags(combo)) == 2:
        atom_idxs_i, atom_idxs_j = Chem.GetMolFrags(combo)

        if len(atom_idxs_i) > len(atom_idxs_j):
            for atom_idx in reversed(atom_idxs_j):
                combo.RemoveAtom(atom_idx)

        elif len(atom_idxs_i) < len(atom_idxs_j):
            for atom_idx in reversed(atom_idxs_i):
                combo.RemoveAtom(atom_idx)

        else:
            raise MolBuilderException('Unable to remove smallest fragment as '
                                      'both fragments are the same size.')


def joined_symm_sssr_connectivities(molecule):
    """Returns a list of lists where the first list is the joined ring
    indices that the first joined ring is connected to and the second
    list is for the second joined ring and so on.

    :param molecule: rdkit Mol object
    :return: a list of joined rings connectivities
    """
    cage_atom_idxs_list = joined_symm_sssr(molecule)

    connectivities_list = []
    for i, cage_atom_idxs_i in enumerate(cage_atom_idxs_list):

        connectivities = []
        for j, cage_atom_idxs_j in enumerate(cage_atom_idxs_list):

            if i == j:
                continue

            if len(set(cage_atom_idxs_i).intersection(cage_atom_idxs_j)) == 2:
                connectivities.append(j)

        connectivities_list.append(connectivities)

    return connectivities_list


def joined_symm_sssr_idx_connections(connectivities_list, joined_ring_number,
                                     checked_joined_rings=None):
    """Returns a list of all joined ring indices that joined_ring_number
    is connected and all joined rings that those joined rings are
    connected to and so on.

    :param connectivities_list: connectivities list from the
        joined_symm_sssr_connectivities function
    :param joined_ring_number: the joined ring number of the joined ring
        we want the joined ring index connections for
    :param checked_joined_rings: checked_joined_rings used to keep track
        of the joined rings that this function has checked as this
        function is called recursively
    :return: a list of all joined ring connection for a given joined
        ring number
    """
    if checked_joined_rings is None:
        checked_joined_rings = []

    if joined_ring_number in checked_joined_rings:
        return []

    checked_joined_rings.append(joined_ring_number)

    connected_cages = []
    for i in connectivities_list[joined_ring_number]:
        if i in checked_joined_rings:
            continue
        connected_cages.append(i)

    for i in connected_cages:
        for j in joined_symm_sssr_idx_connections(
                connectivities_list, i, checked_joined_rings):
            if j not in connected_cages:
                connected_cages.append(j)

    return sorted(connected_cages)


def heteroatom_at_equal_dist(molecule, atom_idx_i, atom_idx_j):
    """This function is used to check if a given bond with atom_i and
    atom_j indices are at the same distances (number of bonds) to any
    heteroatoms in the molecule. This will be used to check which bonds
    in joined ring structure are able to be extended further with
    another joined ring fragment. The molecule must be kekulized before
    calling this function.

    :param molecule: rdkit Mol object of the molecule
    :param atom_idx_i: one the of bonds atom indices
    :param atom_idx_j: one the of bonds atom indices
    :return: a boolean of whether any heteroatoms are at the same
        distances for both atom indices
    """

    # get the cage index number for the bond we are looking at, since
    # this function is used for bonds that might be able to be extended
    # there should only be one cage
    cage_number = None
    for i, cage_atom_idxs_i in enumerate(joined_symm_sssr(molecule)):
        if atom_idx_i in cage_atom_idxs_i and atom_idx_j in cage_atom_idxs_i:
            cage_number = i
            break

    # get all cage indices that the bond is attached to since the
    # checked_joined_rings option is not defined for the
    # joined_symm_sssr_idx_connections function this will be the
    # entire molecule
    cage_connections = [cage_number] + joined_symm_sssr_idx_connections(
        joined_symm_sssr_connectivities(molecule), cage_number
    )

    # check for heteroatoms at equal distance for the entire molecule
    # for atom_idx_i and atom_idx_j
    return heteroatom_at_equal_dist_for_conn(molecule, cage_connections,
                                             atom_idx_i, atom_idx_j)


def heteroatom_at_equal_dist_for_conn(molecule, joined_ring_connections,
                                      atom_idx_i, atom_idx_j):
    """This function is used to check if a given bond with atom_idx_i
    and atom_idx_j indices are at the same distances (number of bonds)
    to any heteroatoms in a given joined_ring_connections of a molecule.
    This is used in function heteroatom_at_equal_distances when all ring
    connections are chosen and for crossover positions when checking a
    subsection of the molecule given by a subset of the ring
    connections. The molecule must be kekulized before calling
    this function.

    :param molecule: rdkit Mol object
    :param joined_ring_connections: joined ring connections of
        the molecule
    :param atom_idx_i: one the of bonds atom indices
    :param atom_idx_j: one the of bonds atom indices
    :return: a boolean of whether any heteroatoms are at the same
        distances for both atom indices
    """
    cage_atom_idxs_list = joined_symm_sssr(molecule)

    # get atoms indices associated to a joined ring connections
    connected_cage_atom_idxs = []
    for i in joined_ring_connections:
        # if there is a barrelene ring in the system return False since
        # there will be no aromatic conjugation
        if len(cage_atom_idxs_list[i]) > 6:
            return False
        connected_cage_atom_idxs += cage_atom_idxs_list[i]

    match = molecule.GetSubstructMatches(Chem.MolFromSmarts('[*]1-*=*-*=*-1'))
    five_membered_ring_center_positions = [i[0] for i in match]

    # get heteroatoms in the molecule in its ring connections
    heteroatoms = []
    for i in set(connected_cage_atom_idxs):
        if i in five_membered_ring_center_positions:
            heteroatoms.append(i)

    # using the distance_matrix determine whether any heteroatoms are at
    # the same distance for atom_i and atom_j
    dist = Chem.GetDistanceMatrix(molecule)
    for heteroatom in heteroatoms:
        if dist[heteroatom][atom_idx_i] == dist[heteroatom][atom_idx_j]:
            return True

    return False


def main_joined_ring_idx(molecule, atom_idxs):
    """Returns the index of the main ring of the crossover. This is the
    ring connected to the crossover position that contains the lowest
    atom index that is not in both rings for the two point crossover.
    There is only one main ring for three and four point crossover, this
    will be ring that contains all crossover atom positions.

    :param molecule: rdkit Mol object of the molecule
    :param atom_idxs: the crossover atom indices
    :return: the main ring index
    """
    main_ring_atom_idxs = None
    ring_idx = None
    for i, ring_atom_idxs in enumerate(joined_symm_sssr(molecule)):

        if not set(atom_idxs).issubset(ring_atom_idxs):
            continue

        if (main_ring_atom_idxs is None
        or min(set(main_ring_atom_idxs).difference(ring_atom_idxs))
        > min(set(ring_atom_idxs).difference(main_ring_atom_idxs))):
            main_ring_atom_idxs = ring_atom_idxs
            ring_idx = i

    return ring_idx


def joined_ring_connection_idxs(molecule, joined_ring_idx, crossover_idxs):
    """Returns a list of the joined ring indices for each of the two
    fragments created during the fragmentation stage of the crossover.

    :param molecule: rdkit Mol object of the molecule
    :param joined_ring_idx: joined main ring index of the crossover
    :param crossover_idxs: crossover atom indices
    :return: a list of the ring indices for each of the two fragments
        created during the fragmentation stage of the crossover
    """
    cage_atom_idxs_list = joined_symm_sssr(molecule)

    # get the neighboring ring indices of the rings next to the main
    # ring and crossover position
    neighbor_ring_idxs = []
    for i, cage_atom_idxs in enumerate(cage_atom_idxs_list):

        connected = len(set(crossover_idxs).intersection(cage_atom_idxs)) == 2

        if joined_ring_idx != i and connected:
            neighbor_ring_idxs.append(i)

    fragment_connections_i = sorted(
        [joined_ring_idx] + joined_symm_sssr_idx_connections(
            joined_symm_sssr_connectivities(molecule), joined_ring_idx,
            neighbor_ring_idxs
        )
    )
    fragment_connections_j = sorted(
        set(range(len(cage_atom_idxs_list))).difference(fragment_connections_i)
    )

    return fragment_connections_i, fragment_connections_j


def bonding_atom_idxs(molecule):
    """Get the dummy atom indices from a fragment and return them in the
    correct order to be used for attachment.

    :param molecule: rdkit Mol object
    :return: dummy atom indices for the fragment in the correct order
        for attachment
    """
    dummy_atoms = molecule.GetAtomsMatchingQuery(AtomNumEqualsQueryAtom(0))
    dummy_atom_idxs = [i.GetIdx() for i in dummy_atoms]
    length = len(dummy_atom_idxs)

    sorted_atom_idxs = []
    for path in FindAllPathsOfLengthN(molecule, length, useBonds=False):
        if set(path) == set(dummy_atom_idxs):
            sorted_atom_idxs = list(path)
            break

    if sorted_atom_idxs[0] > sorted_atom_idxs[-1]:
        sorted_atom_idxs.reverse()

    return sorted_atom_idxs


def set_dummies(molecule, atom_idxs):
    """Changes atom indices to dummies on molecule.

    :param molecule: rdkit Mol object
    :param atom_idxs: list of atom indices
    """
    for atom_idx in atom_idxs:
        aromatic = molecule.GetAtomWithIdx(atom_idx).GetIsAromatic()
        molecule.ReplaceAtom(
            atom_idx, Chem.Atom(0), updateLabel=True, preserveProps=True
        )
        if aromatic:
            molecule.GetAtomWithIdx(atom_idx).SetIsAromatic(True)


def replace_dummies(molecule):
    """Replaces all dummy atoms on the molecules to carbon atoms.

    :param molecule: rdkit Mol object
    """
    for atom in molecule.GetAtomsMatchingQuery(AtomNumEqualsQueryAtom(0)):
        aromatic = atom.GetIsAromatic()
        molecule.ReplaceAtom(
            atom.GetIdx(), Chem.Atom(6), updateLabel=True, preserveProps=True
        )
        if aromatic:
            molecule.GetAtomWithIdx(atom.GetIdx()).SetIsAromatic(True)


def sub_node_comm_num(rank_data, cores_per_job):
    """Returns a dictionary of the rank of the mpi proc, min cpu
    affinity and node name as the key and its sub node communication
    number as the value to be used to split the mpi comms to sub node
    comms useful for blocking mpi when multiprocessor jobs are running.

    :param rank_data: rank data info list which contains a list of the
        mpi ranks, the min cpu affinity and the node name
    :param cores_per_job: the number of cores of the
        multiprocessor job
    :return: dictionary of the sub node comm numbers as values with
        the workers mpi rank as the key
    """
    min_cpu_affs = set([data[1] for data in rank_data])
    nodes = set([data[2] for data in rank_data])

    data = {}
    for node in nodes:
        for min_cpu_aff in min_cpu_affs:
            data[(min_cpu_aff, node)] = []

    for rank, min_cpu_aff, node in rank_data:
        data[(min_cpu_aff, node)].append(rank)

    i = 0
    comm_num = 0
    comm_nums = {}
    for ranks in data.values():
        for rank in ranks:
            comm_nums[rank] = comm_num
            i += 1
            if i != 0 and i % cores_per_job == 0:
                comm_num += 1

    return comm_nums
