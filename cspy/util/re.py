import re

NUMERIC_CONST_PATTERN = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
NUMERIC = re.compile(NUMERIC_CONST_PATTERN)
