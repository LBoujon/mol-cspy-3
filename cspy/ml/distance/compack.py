import numpy as np
from cspy.executable import Compack
from tempfile import TemporaryDirectory


def iterative_compack_batch(ref, comp, nmolecules=30, timeout=10.0, batchsize=20):
    BIG_RMS = 1e10
    rmsds = {i: BIG_RMS - 1 for i, res in enumerate(comp)}
    n = 4
    while n < nmolecules:
        n = min(round(n * 1.618), nmolecules)

        current_res = [i for i, v in rmsds.items() if v < BIG_RMS]
        if not current_res:
            break
        nbatches = int(len(comp) / batchsize) + 1
        for batchno, batch in enumerate(np.array_split(current_res, nbatches)):
            if len(batch) == 0:
                continue
            with TemporaryDirectory(prefix="/dev/shm/") as tempdir:
                c = Compack(
                    ref,
                    [comp[i] for i in batch],
                    working_directory=tempdir,
                    timeout=timeout,
                )
                c.run()
                for ref_idx, batch_idx, rms in c.result:
                    cr_idx = batch[batch_idx]
                    rmsds[batch[batch_idx]] = rms
    return rmsds
