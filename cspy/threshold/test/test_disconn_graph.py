from cspy.threshold.disconnectivity_graph import DisconnectivityGraph
from pathlib import Path
from cspy.db import CspDataStore


PICKLED_GRAPHS = {
    "perylene_FIT": Path(__file__).parent / "disconn_data" / "perylene_FIT_1.pickle",
    "perylene_PAHAP": Path(__file__).parent / "disconn_data" / "perylene_PAHAP_1.pickle",
    "phenanthrene_PAHAP": Path(__file__).parent / "disconn_data" / "phenanthrene_PAHAP_1.pickle"
}

DATABASES = {
    "benzene": str(Path(__file__).parent / "databases" / "benzene.db")
}


def test_from_db() -> None:
    ds = CspDataStore(DATABASES["benzene"])
    disconn = DisconnectivityGraph(ds)
    assert len(disconn.basin_roots) == 1
    assert disconn.num_nodes == 8
    assert disconn.barrier_between_ids("benzene-Thre-1-0-3-0", "benzene-Thre-1-0-3-58") == 6
    assert disconn.barrier_between_ids("benzene-Thre-1-0-3-65", "benzene-Thre-1-0-3-54") == 7
    assert len(disconn.lid_ids_from_level(4)) == 1
    ini, min_ = disconn.initial_minima_ids()
    assert (len(ini), len(min_)) == (1, 7)
    assert len(disconn.equivalent_structs("benzene-Thre-1-0-3-58")) == 1
    assert len(disconn.equivalent_structs("benzene-Thre-1-0-3-0")) == 77
    assert len(disconn.structures_in_lid("thre-0-6")) == 4
    assert len(disconn.structures_in_lid("thre-0-3")) == 0
    ds.disconnect()


def test_from_pickle() -> None:
    disconn = DisconnectivityGraph(PICKLED_GRAPHS["perylene_PAHAP"])
    assert len(disconn.basin_roots) == 2
    assert disconn.num_nodes == 41
    ini, min_ = disconn.initial_minima_ids()
    assert (len(ini), len(min_)) == (2, 39)
    

def test_barrier_between_ids() -> None:
    disconn = DisconnectivityGraph(PICKLED_GRAPHS["perylene_PAHAP"])
    assert disconn.barrier_between_ids("perylene-SG-1-0-3-0", "perylene-SG-1-1-3-0") == None

    disconn = DisconnectivityGraph(PICKLED_GRAPHS["phenanthrene_PAHAP"])
    assert disconn.barrier_between_ids("pyrene-SG-1-0-3-0", "pyrene-SG-1-5-3-0") == 8
    assert disconn.barrier_between_ids("pyrene-SG-1-0-3-0", "pyrene-SG-1-1-3-0") == 10


def test_lid_ids_from_level() -> None:
    disconn = DisconnectivityGraph(PICKLED_GRAPHS["perylene_FIT"])
    assert len(disconn.lid_ids_from_level(4)) == 2

    disconn = DisconnectivityGraph(PICKLED_GRAPHS["phenanthrene_PAHAP"])
    assert len(disconn.lid_ids_from_level(4)) == 3


def test_disconn_matrix() -> None:
    disconn = DisconnectivityGraph(PICKLED_GRAPHS["perylene_FIT"])
    assert disconn.disconnectivity_matrix().shape == (2, 2)

    disconn = DisconnectivityGraph(PICKLED_GRAPHS["perylene_PAHAP"])
    assert disconn.disconnectivity_matrix().shape == (2, 2)

    disconn = DisconnectivityGraph(PICKLED_GRAPHS["phenanthrene_PAHAP"])
    assert disconn.disconnectivity_matrix().shape == (3, 3)
