from cspy.threshold import threshold_db
from cspy.db import CspDataStore
from pathlib import Path


DATABASES = {
    "benzene": Path(__file__).parent / "databases" / "benzene.db"
}


def test_parse_db() -> None:
    _, trials, crystals1, trial_structures, descriptors = threshold_db.parse_db(DATABASES["benzene"], range(0, 1))
    assert len(trials) == 1
    assert len(descriptors) == 94
    assert len(trial_structures) == len(crystals1)

    _, trials, crystals2, trial_structures, descriptors = threshold_db.parse_db(DATABASES["benzene"], range(3, 4), only_valid=False)
    assert trials[0][1] == 3

    assert len(crystals2) >= len(crystals1)


def test_sql_queries() -> None:
    ds = CspDataStore(str(DATABASES["benzene"]))
    data = ds.query(threshold_db.UNIQUE_STRUCTURES_SQL).fetchall()
    assert len(data) == 8

    data = ds.query(threshold_db.TRAJECTORY_STRUCTURES_SQL).fetchall()
    assert len(data) == 94


def test_get_data_by_trial() -> None:
    trials, _crystals, _trial_structures, descriptors = threshold_db.get_data_by_trial(DATABASES["benzene"], 0)
    assert trials[0][1] == 0
    assert len(descriptors) == 94
