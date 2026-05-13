from scipy.spatial import cKDTree as KDTree
import numpy as np


class NearestAtoms:
    def __init__(self, symbols, positions):
        self.symbols = symbols
        self.positions = positions
        self.tree = KDTree(self.positions)

    def neighbours(self, pt, r, ncore = 1):
        idx = self.tree.query_ball_point(pt, r, n_jobs=ncore)
        return self.symbols[idx], self.positions[idx]


def hkl_finder(s, radius):

    """
    Samples the surface of a sphere around the centroid, and returns maximum
    and minimum of h,k, and l in a way that all the sampled points are inside
    the cluster (supercell).

    """

    from numpy import sin as sn, cos as cs

    centroid_xyz = np.mean(s.positions, axis = 0)
    thetas = np.linspace(0, np.pi, 18, endpoint=True)
    phis = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    surface_points = []
    for t in thetas:
        for ph in phis:
            surface_points.append([radius * sn(t) * cs (ph) + centroid_xyz[0],
                                  radius * sn(t) * sn(ph) + centroid_xyz[1],
                                  radius * cs(t) + centroid_xyz[2]])
    surface_points = np.array(surface_points)

    #Surface by lattice vectors = sblv
    sblv = s.to_fractional(surface_points)
    h_max, k_max, l_max = np.array([np.ceil(np.max(sblv[:,0])),
                                   np.ceil(np.max(sblv[:,1])),
                                   np.ceil(np.max(sblv[:,2]))])
    h_min, k_min, l_min = np.array([np.floor(np.min(sblv[:,0])),
                                   np.floor(np.min(sblv[:,1])),
                                   np.floor(np.min(sblv[:,2]))])
    return h_max, k_max, l_max, h_min, k_min, l_min


def atoms_in_radius(s, radius):

    centroid_xyz = np.mean(s.positions, axis=0)
    farthest_atom_xyz = np.max(np.linalg.norm(s.positions - centroid_xyz, axis=1))

    h_max, k_max, l_max, h_min, k_min, l_min = hkl_finder(s, radius + farthest_atom_xyz)
    full_unit_cell = s.positions
    atomic_positions = full_unit_cell
    for h in np.arange(h_min, h_max):
        for k in np.arange(k_min, k_max):
            for l in np.arange(l_min, l_max):
                if h == k == l == 0:
                    continue
                else:
                    atomic_positions = np.concatenate((atomic_positions,
                                                       full_unit_cell + np.dot(np.array([h,k,l]),
                                                                               np.array(s.lattice_vectors))),
                                                      axis=0)

    ncells = int(((h_max-h_min))*((k_max-k_min))*((l_max-l_min)))
    atomic_symbols = np.tile(s.symbols, ncells)
    return NearestAtoms(atomic_symbols, atomic_positions)


def RDF(snapshot, radius = 16, dr = 0.1, ncore = 1):

    nearest_atoms = atoms_in_radius(snapshot, radius)
    cm_xyz = snapshot.positions
    cm_symbols = np.array(snapshot.symbols).astype('str')

    # Initializing the rdf_dict
    rdf_dict = {}
    for s1 in np.unique(cm_symbols):
        for s2 in np.unique(cm_symbols):
            if s1 <= s2:
                rdf_dict[s1, s2] = np.zeros(int(radius / dr))

    # Finding CLOSE atom pairs considering their type
    for idx1 in range(cm_symbols.shape[0]):
        cluster_symb, cluster_pos = nearest_atoms.neighbours(cm_xyz[idx1],
                                                             radius,
                                                             ncore = ncore)

        # Removing atoms in the parent molecule
        for atom in cm_xyz:
            idx_r = np.where(np.linalg.norm(cluster_pos - atom, axis = 1) < 0.0001)
            cluster_symb = np.delete(cluster_symb, [idx_r])
            cluster_pos = np.delete(cluster_pos, idx_r, 0)

        for idx2 in range(cluster_symb.shape[0]):
            connecting_vector = cm_xyz[idx1] - cluster_pos[idx2]
            dist = np.linalg.norm(connecting_vector)

            if cm_symbols[idx1] <= cluster_symb[idx2]:
                rdf_dict[cm_symbols[idx1], cluster_symb[idx2]][int(dist/dr)] += 1
            else:
                rdf_dict[cluster_symb[idx2], cm_symbols[idx1]][int(dist/dr)] += 1

    for pair in rdf_dict.keys():
        n1 = snapshot.element_count[pair[0]]
        n2 = snapshot.element_count[pair[1]]
        vol = snapshot.volume

        #To avoid double counting
        double_counting_fac = 1
        if pair[0] != pair[1]:
            double_counting_fac = 2

        radii = np.arange(rdf_dict[pair].shape[0]) * dr + dr/2

        rdf_dict[pair] *= double_counting_fac
        rdf_dict[pair] /= (4 * np.pi * (radii)**2 * dr *
                           n1 * n2 / vol)

    return rdf_dict


if __name__ == '__main__':
    from lammps_to_n2p2 import snapshot_file_reader as sr
    from pathlib import Path
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(
        description='Takes lammps snapshots and return RDF of the data set'
                    )
    parser.add_argument('-sd', '--snapdir',
                        metavar='',
                        help='Path to to the folder where the snapshots are',
                        required=True)

    parser.add_argument('-r', '--radius',
                        metavar='',
                        help='Desired radius for calculation of RDF',
                        type=float,
                        default=16.0)

    parser.add_argument('-n', '--ncore',
                        metavar='',
                        help='Number of cores for parallel jobs',
                        type=int,
                        default=1)

    parser.add_argument('-w', '--width',
                        metavar='',
                        help='Width of histograms for calculation of RDF',
                        type=float,
                        default=0.1)

    parser.add_argument('-o', '--output',
                        metavar='',
                        help='Output filename appendix',
                        default='RDF.dat')

    args = parser.parse_args()

    snapshot_files = list(Path(args.snapdir).glob('snapshot*.dat'))
    s = sr(snapshot_files[0])
    cm_symbols = np.array(s.symbols).astype('str')

    rdf_total = {}
    for s1 in np.unique(cm_symbols):
        for s2 in np.unique(cm_symbols):
            if s1 <= s2:
                rdf_total[s1, s2] = np.zeros(int(args.radius/args.width))

    # for snapshot_file in sorted(snapshot_files):
    #     s = sr(snapshot_file)
    #     rdf_temp = RDF(s, radius = args.radius, dr = args.width)
    #     for key in rdf_total.keys():
    #         rdf_total[key] += rdf_temp[key]

    def jobWrapper(snapshot_file, radius, dr):
        s = sr(snapshot_file)
        return RDF(s, radius, dr)

    with ProcessPoolExecutor(args.ncore) as executor:
        RDFpool = {
            executor.submit(jobWrapper, snapshot_file, radius = args.radius, dr = args.width)
            for snapshot_file in snapshot_files
        }
        for future in as_completed(RDFpool):
            for key in rdf_total.keys():
                rdf_total[key] += future.result()[key] / len(snapshot_files)
                # normalized by the number of snapshots

    for key in rdf_total.keys():
        rdf_list = [np.arange(int(args.radius/args.width)) * args.width]
        rdf_list.append(rdf_total[key])
        np.savetxt('{}_{}_{}'.format(key[0], key[1], args.output),
                   np.array(rdf_list).T,
                   # fmt = '%-6.1f'
                   )