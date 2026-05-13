from os.path import dirname, join

CURRENT_SCHEMA_VERSION = "cspy2"
_DIR = dirname(__file__)

with open(join(_DIR, "cspy2.sql")) as f:
    cspy2 = f.read()

with open(join(_DIR, "cspy2_flex.sql")) as f:
    cspy2_flex = f.read()

with open(join(_DIR, "cspy2_mps.sql")) as f:
    cspy2_mps = f.read()

with open(join(_DIR, "cspy2_aus.sql")) as f:
    cspy2_aus = f.read()

with open(join(_DIR, "cspy2_molecules.sql")) as f:
    cspy2_molecules = f.read()

with open(join(_DIR, "deprecated.sql")) as f:
    deprecated = f.read()

current = cspy2
current_flex = cspy2_flex
current_molecules = cspy2_molecules
current_mps = cspy2_mps
current_aus = cspy2_aus

__all__ = ["cspy2", 
           "cspy2_flex", 
           "cspy2_mps", 
           "cspy2_aus", 
           "current", 
           "current_flex", 
           "current_molecules", 
           "current_mps", 
           "current_aus", 
           "CURRENT_SCHEMA_VERSION", 
           "deprecated"]
