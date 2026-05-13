import logging
from pathlib import Path
import re

EXYZ_FIELD_REGEX = re.compile(r"""(\w+)=(".*?"|\S+)""")
LOG = logging.getLogger(__name__)


def parse_xyz_string(contents, extended_xyz=False, filename="string"):
    contents = contents.splitlines()
    natom = int(contents[0].strip())
    LOG.debug("Expecting %d atoms in %s", natom, filename)
    xyz_dict = {"comment": contents[1].strip(), "atoms": []}
    for line in contents[2:]:
        if not line.strip():
            break
        tokens = line.strip().split()
        el = tokens[0]
        xyz = tuple(float(x) for x in tokens[1:4])
        xyz_dict["atoms"].append((el, xyz))
    LOG.debug("Found %d atoms lines in %s", len(xyz_dict["atoms"]), filename)
    if extended_xyz:
        comment_line = xyz_dict.pop("comment")
        matches = EXYZ_FIELD_REGEX.findall(comment_line)
        for k, v in matches:
            xyz_dict[k] = v.strip("\"'")
    return xyz_dict


def parse_xyz_file(filename, **kwargs):
    contents = Path(filename).read_text()
    return parse_xyz_string(contents, filename=filename, **kwargs)
