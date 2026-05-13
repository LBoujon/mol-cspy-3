#!/usr/bin/env python
import logging
from cspy.chem.multipole import DistributedMultipoles
from cspy.util.path import Path
from cspy.util.logging_config import DATEFMT, FORMATS

LOG = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dma_files", nargs="+", help="Molecular DMA files")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format=FORMATS[args.log_level], datefmt=DATEFMT
    )
    for fname in args.dma_files:
        LOG.info("Processing %s", fname)
        dm = DistributedMultipoles.from_dma_file(fname)
        dm.sort_atoms_for_neighcrys()
        dm.reorient_onto_molecular_axes()
        Path(fname).with_suffix(".oriented.dma").write_text(dm.to_dma_string())


if __name__ == "__main__":
    main()
