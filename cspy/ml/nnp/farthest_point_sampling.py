import numpy as np
from chainer import cuda

def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices, distances


if __name__ == '__main__':
    import argparse
    from cspy import Crystal
    import os
    import sys
    sys.path.append('/home/masih/.apps/average-minimum-distance/build/lib/')
    import amd

    parser = argparse.ArgumentParser(
        description='Takes res files and returns the sampled list by Farthest Point Sampling (FPS)'
    )

    parser.add_argument(
        '-s', '--seed',
        metavar='',
        help='The initial seed for maximin or minimax',
        default=0
    )

    parser.add_argument(
        "files",
        type=str,
        nargs='+',
        help='res files to convert',
    )

    parser.add_argument(
        '-n', '--npoints',
        metavar='',
        help='The number of points needs to be sampled.',
        default=50
    )

    parser.add_argument(
        '-k', '--amd_k',
        type=int,
        help='The length of AMD descriptor',
        default=200
    )

    parser.add_argument(
        '--sampled_output',
        metavar='',
        help='Output file for list of first n sampled points by FPS',
        default='sampled_points.txt'
    )

    parser.add_argument(
        '--unsampled_output',
        metavar='',
        help='Output file for list of points the are not sampled.',
        default='unsampled_points.txt'
    )


    parser.add_argument(
        '--faulty_ids',
        metavar='',
        help='The list of IDs of structures that for any reason are invalid and should not be '
        'included in the FPS sampling.',
    )

    parser.add_argument(
        '--excluded_ids',
        metavar='',
        help='The list of IDs of structures that for any reason (before or during FPS) were not '
             'included in the FPS sampling.',
        default='excluded_ids.txt'
    )

    args = parser.parse_args()

    faulty_ids = []
    if args.faulty_ids:
        with open(args.faulty_ids, 'r') as f:
            for line in f.readlines():
                faulty_ids.append(line.split()[0])

    amd_dict = {}
    excluded_ids = []
    for res_file in args.files:
        id_ = res_file.split('/')[-1].split('.')[0]

        if id_ in faulty_ids:
            excluded_ids.append(id_)
            continue

        try:
            crys = Crystal.load(res_file)
        except:
            print(f'Error! {id_} cannot be loaded by CSPy.')
            continue

        crys.save(f'{id_}.cif')

        try:
            amd_cif_input = list(amd.io.CifReader(f'{id_}.cif'))[0]
            amd_dict[id_] = amd.AMD(amd_cif_input, args.amd_k)
        except:
            excluded_ids.append(id_)
            print(f'{id_} was not considered because of error in AMD calculation')

        os.remove(f'{id_}.cif')

    ids_list = []
    amd_list = []
    for key in sorted(amd_dict.keys()):
        ids_list.append(key)
        amd_list.append(amd_dict[key])
    amd_list = np.array(amd_list)

    neighbours, _ = farthest_point_sampling(
        amd_list,
        int(args.npoints),
        initial_idx = args.seed
    )

    sampled_keys = []
    with open(args.sampled_output, 'w') as f:
        for neigh in neighbours[0]:
            sampled_keys.append(ids_list[neigh])
            f.write(f'{ids_list[neigh]}\n')

    with open(args.unsampled_output, 'w') as f:
        for key in ids_list:
            if key in sampled_keys:
                continue
            f.write(f'{key}\n')

    with open(args.excluded_ids, 'w') as f:
        for id_ in excluded_ids:
            f.write(f'{id_}\n')