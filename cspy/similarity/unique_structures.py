from rtree import Rtree
import logging
from cspy.ml.distance import iterative_cdtw_distance
from cspy.executable import Compack
from cspy.util.path import TemporaryDirectory
import time
import numpy as np
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from sklearn.metrics.pairwise import cosine_similarity

LOG = logging.getLogger("cspy.similarity")
#LOG.setLevel(logging.DEBUG)


class Structure:
    def __init__(self, id, coords, xrd, res, trial=None, errs=(1.0, 0.05)):
        self.id = id
        self.coords = coords
        self.res = res
        self.xrd = xrd / simps(xrd, dx=0.02)
        self.errs = errs
        if trial is not None:
            self.trial = int(trial)

    def same_structure_xrd(self, others, threshold=10.0, iterations=10):
        return [
            i
            for i, other in enumerate(others)
            if iterative_cdtw_distance(self.xrd, other.xrd, iterations, threshold)
        ]

    def same_structure_cosine(self, others, threshold=0.99):
        xrds = [self.xrd]
        for other in others:
            xrds.append(other.xrd)
        LOG.debug("about to do cosine_similarity")
        s = cosine_similarity(xrds)
        LOG.debug("s in same_structure_cosine: %s", s)
        return np.where(s[0, 1:] > 0.5)[0]

    def same_structure_compack(self, others, threshold=0.3):
        ref = [self.res]
        cmp = [other.res for other in others]
        with TemporaryDirectory() as tmpdir:
            cpk = Compack(ref, cmp)
            cpk.working_directory = tmpdir
            cpk.run()
            result = cpk.result()
        return [i for i, rmsd in enumerate(result) if rmsd[2] < threshold]

    @property
    def bounds(self):
        return (
            self.coords[0] - self.errs[0],
            self.coords[1] - self.errs[1],
            self.coords[0] + self.errs[0],
            self.coords[1] + self.errs[1],
        )


class UniqueStructures:
    def __init__(self):
        self.structures = []
        self.tree = Rtree()

    def bulk_insert(self, structures, comparison="cosine"):
        duplicates = []
        for i, s in enumerate(structures):
            t1 = time.time()
            duplicates += self.insert_without_duplicates(s, method=comparison)
            t2 = time.time()
        return duplicates

    def insert_without_duplicates(self, structure, method="cosine"):
        """Add a structure to the list of unique structures
        if it is unique. Otherwise, remove the higher energy
        duplicate structure(s) from this tree, and return them
        """
        bounds = structure.bounds
        hits = list(self.tree.intersection(bounds, objects="raw"))
        equivalent = []
        if hits:
            if method == "cosine":
                same = structure.same_structure_cosine(hits)
            elif method == "xrd":
                same = structure.same_structure_xrd(hits)
            else:
                same = structure.same_structure_compack(hits)
            equivalent = [hits[i] for i in same]
            for i, s in enumerate(equivalent):
                self.tree.delete(s.id, s.bounds)
        equivalent.append(structure)
        equivalent = sorted(equivalent, key=lambda x: x.coords[0])
        s = equivalent[0]
        self.tree.insert(s.id, s.bounds, s)
        return equivalent[1:]

    def contains(self, structure, method="cosine"):
        """
        check if list contains duplicates of the structure
        """
        bounds = structure.bounds
        hits = list(self.tree.intersection(bounds, objects="raw"))
        equivalent = []
        if hits:
            if method == "cosine":
                same = structure.same_structure_cosine(hits)
            elif method == "xrd":
                same = structure.same_structure_xrd(hits)
            else:
                same = structure.same_structure_compack(hits)
            equivalent = [hits[i] for i in same]

        return True if equivalent else False

    def insert_check_trial(self, structure, method="cosine"):
        """Add a structure to the list of unique structures
        if it is unique. Check whether duplicates are in same 
        trial (seed), if so return self_cover=True
        """
        bounds = structure.bounds
        hits = list(self.tree.intersection(bounds, objects="raw"))
        equivalent = []
        self_cover = False
        if hits:
            if method == "cosine":
                same = structure.same_structure_cosine(hits)
            elif method == "xrd":
                same = structure.same_structure_xrd(hits)
            else:
                same = structure.same_structure_compack(hits)
            equivalent = [hits[i] for i in same]
            for i, s in enumerate(equivalent):
                self.tree.delete(s.id, s.bounds)
        equivalent.append(structure)
        trial = min(equivalent, key=lambda x: x.trial).trial
        self_cover = trial == structure.trial
        equivalent = sorted(equivalent, key=lambda x: x.coords[0])
        s = equivalent[0]
        s.trial = trial
        LOG.debug(
            "On-the-fly clustering, coords %s, identity %s, trial %s",
            s.coords,
            s.trial,
            structure.trial,
        )
        self.tree.insert(s.id, s.bounds, obj=s)
        return equivalent[1:], self_cover

    @property
    def number_structures(self):
        return self.tree.count(self.tree.bounds)

    @property
    def clustered_structures(self):
        items = []
        objs = self.tree.intersection(self.tree.bounds, objects=True)
        for obj in objs:
            items.append(obj.object)
        return items
