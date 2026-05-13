import re
import logging
import numpy as np


LOG = logging.getLogger("dmacrys_summary")
NUMBER_REGEX = r"[-+]?[0-9]+[.]?[0-9]*"
KEYVAL_REGEX = re.compile(r"\s*(.+)\s*:\s*({})+".format(NUMBER_REGEX))


class DmacrysSummary:
    def __init__(self, contents):
        self.contents = contents
        #self.valid = False
        self._check_errors()
        if not self.error:
            self._parse()

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            return cls(f.read())
        
    def _check_errors(self):
        error = None
        #Buck_search = re.search("Invalid minimisation - negative curvature", self.contents)
        Buck_search1 = re.search("Initial Lattice Energy:   \*\*", self.contents)
        Buck_search2 = re.search("Final Lattice Energy:   \*\*", self.contents)
        It_search = re.search("too many iteration cycles", self.contents)
        if Buck_search1 or Buck_search2 or It_search:
            LOG.debug("Invalid minimization")
            LOG.debug("Summary contents:\n%s", self.contents)
            if Buck_search1 or Buck_search2:
                error = "BuckCat"
                LOG.exception("Error in Dmacrys: Invalid energy at end of optimisation. Probable Buckingham catastrophe.", stack_info=False, exc_info=False)
            if It_search:
                error = "MaxIts"
                LOG.exception("Error in Dmacrys: Total GO iterations exceeds max allowed GO iterations.", stack_info=False, exc_info=False)

        self.error = error

    def _parse(self):
        try:
            self._parse_keyvals()
            self._parse_tables()
        except Exception as e:
            LOG.exception("Error parsing dmacrys_summary %s", e)
            self.error = "Unknown"

    def _parse_keyvals(self):
        matches = re.findall(KEYVAL_REGEX, self.contents)
        desired_quantities = {
            "Final Lattice Energy": (float, "final_energy"),
            "Initial Lattice Energy": (float, "initial_energy"),
            "Number of Iterations": (int, "iterations"),
        }
        for k, val in matches:
            k = k.strip(".= ")
            if k in desired_quantities:
                conv, name = desired_quantities[k]
                setattr(self, name, conv(val))

    def _parse_tables(self):

        lineiter = enumerate(self.contents.splitlines(), start=1)
        for i, line in lineiter:
            if "Valid minimisation" in line:
                self.valid = True
            if "Final Lattice Energy" in line:
                next(lineiter)
                i, line = next(lineiter)
                cols = [x for x in line.split() if not "(" in x]
                rows = []
                for _ in range(4):
                    i, line = next(lineiter)
                    rows.append(line)
                for label, r in zip(["initial", "final"], rows):
                    for c, v in zip(cols, r.split()[2:]):
                        setattr(self, "{}_{}".format(label, c), float(v))
            if "density" in line:
                next(lineiter)
                i, initial_line = next(lineiter)
                self.initial_volume, self.initial_density = [
                    float(x) for x in initial_line.split()[-2:]
                ]
                i, final_line = next(lineiter)
                self.final_volume, self.final_density = [
                    float(x) for x in final_line.split()[-2:]
                ]
            if "Contributions" in line:
                z = int(line.split()[-1].strip(")"))
                lat_contributions = {}
                for _ in range(7):
                    i, line = next(lineiter)
                    tokens = line.split("=")
                    k = tokens[0].split(".")[0]
                    v_ev, v_kj = tokens[-1].split("[")
                    v_ev = float(v_ev.strip())
                    v_kj = float(v_kj.strip(" ]"))
                    lat_contributions[k] = v_kj
                self.z = z
                self.lattice_energy_contributions = lat_contributions
            if "Zone Centre" in line:
                phonon_frequencies = []
                i, line = next(lineiter)
                while len(line) != 0:
                    phonon_frequencies += [float(i) for i in list(line.split())]
                    i, line = next(lineiter)
                if any(i < 0 for i in phonon_frequencies):
                    LOG.warning(
                        "Phonon modes calculated with negative "
                        "vibrational frequencies: %s", phonon_frequencies
                    )
                self.phonon_frequencies = np.array(phonon_frequencies)

    @property
    def electrostatic_contribution(self):
        es_keys = (
            "Inter-molecular charge-charge energy",
            "Total charge-dipole+dipole-dipole energy",
            "Higher multipole interaction energy",
        )
        return sum(self.lattice_energy_contributions.get(k, 0.0) for k in es_keys)
