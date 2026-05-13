from ._cdtw import cdtw_sakoe_chiba, iterative_cdtw_sakoe_chiba


def cdtw_distance(x, y, band_range):
    return cdtw_sakoe_chiba(x, y, band_range)


def iterative_cdtw_distance(x, y, band_range, threshold):
    return iterative_cdtw_sakoe_chiba(x, y, band_range, threshold)
