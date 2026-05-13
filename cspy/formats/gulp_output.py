import logging
import numpy as np
from pathlib import Path
import re
from typing import Tuple, Union, Optional

LOG = logging.getLogger(__name__)


class GulpOutputException(Exception):
    """Generic exception for Dmacrys output failures.

    """


class GulpOutput:

    def __init__(self, contents : str, parse: Optional[bool] = True, singlepoint: Optional[bool] = False):
        """Gulp output class for parsing the GULP standard output
        file and converting data into a more useful form.

        Parameters
        ----------
        contents : string
            string containing the contents of the gulp standard out
        parse : bool
            whether or not to parse the output
        singlepoint : bool
            whether or not the output is from a singlepoint calculation

        """
        self.contents = contents
        self._content_lines = self.contents.splitlines()
        self.parsed_contents = {}
        self.singlepoint = singlepoint

        self.parsed = False
        self.reached_maxcyc = False

        if parse:
            self._parse()


    def _parse(self):
        """ Iterate over lines in GULP output to identify 
        space group, unit cell, and asymmetric unit and return them
        to self.parsed_contents.
        """
        from cspy.crystal import Crystal, AsymmetricUnit, SpaceGroup, UnitCell
        from cspy.chem.element import Element
        content_lines_iter = iter(self._content_lines)
        for line in content_lines_iter:
            if "Space group" in line:
                k, v = self._parse_single_line_block(line)
                self.parsed_contents["space group"] = v.strip()

            if "**** Maximum number of function calls has been reached ****" in line:
                self.reached_maxcyc = True

            elif "Cartesian lattice vectors" in line or "Final Cartesian lattice vectors" in line:
                if "Final" in line or self.singlepoint:
                    array = np.array(
                        [list(map(float, x.split())) for x in self._parse_table(content_lines_iter)]
                    )
                    self.parsed_contents["unit cell"] = UnitCell(array)

            # The label preceeding the atomic coordinates changes depending on the GULP settings
            # the first 2 are used for singlepoints and the last 2 are for GOs
            # the 2nd and 4th are for P1 cells where the asymmetric unit == the unit cell
            elif "Fractional coordinates of asymmetric unit" in line or \
                "Fractional coordinates of atoms" in line or \
                "Final asymmetric unit coordinates" in line or \
                "Final fractional coordinates of atoms" in line:
                if "Final" in line or self.singlepoint:
                    for i in range(4):
                        next(content_lines_iter)
                    elements = []
                    positions = []
                    for l in self._parse_table(content_lines_iter):
                        tokens = l.split()
                        elements.append(Element.from_string(tokens[1]))
                        positions.append((float(tokens[3]), float(tokens[4]), float(tokens[5])))
                    self.parsed_contents["asymmetric unit"] = AsymmetricUnit(elements, positions)

        LOG.debug('Parsed content "%s"', self.parsed_contents)

    def _parse_table(self, content_lines_iter):
        """ Identify where a table begins in GULP and return only that

        Parameters
        ----------
        content_lines_iter : itertools object
            lines from GULP output

        Returns
        -------
        lines : list
            list of strings of lines in table
        """
            
        reading_table = False
        next(content_lines_iter)

        lines = []
        for line in content_lines_iter:
            if not line.strip():
                break
            if line.startswith("----------------------"): continue
            else:
                lines.append(line.strip())
        return lines


    def _parse_single_line_block(self, line : str):
        return re.split('\s*[:=]\s*', line)


    @classmethod
    def from_file(cls, filename):
        return cls(Path(filename).read_text())
