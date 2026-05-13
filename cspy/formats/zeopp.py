import logging
from cspy.util.path import Path

LOG = logging.getLogger(__name__)


def parse_sa_file(path):
    LOG.debug("Reading zeo++ surface area file: %s", path)
    contents = Path(path).read_text()
    lines = contents.splitlines()
    sa_data = {}
    for i, line in enumerate(lines):
        tokens = line.split(" ")
        current_values = []
        key = None
        for token in tokens:
            if token.endswith(":"):
                if key is None:
                    if i == 0:
                        sa_data["filename"] = " ".join(current_values[1:])
                else:
                    sa_data[key] = current_values
                key = token.rstrip(":")
                current_values = []
            elif token:
                current_values.append(token)
        sa_data[key] = current_values

    for key in sa_data:
        if not sa_data[key]:
            sa_data[key] = None
        elif len(sa_data[key]) == 1:
            sa_data[key] = sa_data[key][0]
    return sa_data


def parse_chan_file(path):
    LOG.debug("Reading zeo++ channel file: %s", path)
    path = Path(path)
    contents = path.read_text()
    lines = contents.splitlines()
    chan_data = {"name": path.stem}
    tokens = lines[0].split()
    n = int(tokens[1])
    if n == 0:
        return chan_data

    dimensions = [int(x) for x in tokens[-n:]]
    for dim, line in zip(dimensions, lines[1:]):
        _, chan_id, Di, Df, Dif = line.split()
        channel = f"channel_{chan_id}"
        chan_data[f"{channel}_dimension"] = dim
        chan_data[f"{channel}_Di"] = Di
        chan_data[f"{channel}_Df"] = Df
        chan_data[f"{channel}_Dif"] = Dif
    return chan_data
