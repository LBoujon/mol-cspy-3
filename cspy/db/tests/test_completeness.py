import csv
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from cspy.db.completeness import (
    DEFAULT_WINDOWS,
    combine_database_observations,
    convergence,
    load_database_observations,
    main as completeness_main,
    summarize,
)
from cspy.db.dump import main as dump_main
from cspy.db.clustering import main as clustering_main


def create_clustered_database(
    path: Path, representative_layout: str, prefix: str = ""
) -> None:
    def sid(value: str) -> str:
        return f"{prefix}{value}"

    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE crystal (
            id TEXT PRIMARY KEY,
            spacegroup INTEGER,
            density REAL,
            energy REAL,
            molecule_id TEXT,
            file_content TEXT
        );
        CREATE TABLE trial_structure (
            id TEXT,
            minimization_step INTEGER,
            trial_number INTEGER,
            valid BOOLEAN,
            minimization_time REAL,
            metadata TEXT
        );
        CREATE TABLE equivalent_to (unique_id TEXT, equivalent_id TEXT);
        CREATE TABLE descriptor (
            id TEXT, name TEXT, value BLOB, metadata TEXT
        );
        CREATE TABLE meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT,
            description TEXT,
            modification_time TEXT
        );
        CREATE TABLE trial (
            trial_id TEXT,
            trial_number INTEGER,
            iterations INTEGER,
            state TEXT,
            metadata TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO crystal VALUES (?, ?, ?, ?, ?, ?)",
        [
            (sid("a"), 2, 1.1, -10.0, "mol", "TITL a"),
            (sid("b"), 4, 1.1, -9.8, "mol", "TITL b"),
            (sid("c"), 4, 1.1, -8.0, "mol", "TITL c"),
            (sid("d"), 2, 1.1, -7.0, "mol", "TITL d"),
            (sid("invalid"), 2, 1.1, -20.0, "mol", "TITL invalid"),
        ],
    )
    final_rows = [
        (sid("a"), 3, 1, 1, 1.0, None),
        (sid("a"), 3, 1, 1, 1.0, None),
        (sid("b"), 3, 2, 1, 1.0, None),
        (sid("c"), 3, 3, 1, 1.0, None),
        (sid("d"), 3, 4, 1, 1.0, None),
        (sid("invalid"), 3, 5, 0, 1.0, None),
    ]
    connection.executemany(
        "INSERT INTO trial_structure VALUES (?, ?, ?, ?, ?, ?)",
        [(sid("a"), 2, 1, 1, 1.0, None), *final_rows],
    )
    if representative_layout == "null":
        equivalences = [
            (sid("a"), None),
            (sid("a"), sid("b")),
            (sid("c"), None),
            (sid("d"), None),
        ]
    elif representative_layout == "self":
        equivalences = [
            (sid("a"), sid("a")),
            (sid("a"), sid("b")),
            (sid("c"), sid("c")),
            (sid("d"), sid("d")),
        ]
    else:
        raise ValueError(representative_layout)
    connection.executemany(
        "INSERT INTO equivalent_to VALUES (?, ?)", equivalences
    )
    connection.commit()
    connection.close()


@pytest.mark.parametrize("representative_layout", ["null", "self"])
def test_database_layouts_and_duplicate_trial_rows(tmp_path, representative_layout):
    database = tmp_path / "clustered.db"
    create_clustered_database(database, representative_layout)

    observations = load_database_observations(database)

    assert len(observations) == 4
    by_id = {item.structure_id: item for item in observations}
    assert by_id["a"].representative_id == "a"
    assert by_id["b"].representative_id == "a"
    assert by_id["a"].multiplicity == 2
    assert by_id["b"].multiplicity == 2
    assert by_id["c"].multiplicity == 1
    assert "invalid" not in by_id


def test_global_summary_combines_rediscoveries_across_spacegroups(tmp_path):
    database = tmp_path / "clustered.db"
    create_clustered_database(database, "null")
    observations = load_database_observations(database)

    rows = summarize(observations)
    total = next(
        row for row in rows if row.scope == "all" and row.spacegroup == "ALL"
    )

    assert (total.N, total.S_obs, total.f1, total.f2) == (4, 3, 2, 1)
    assert total.good_turing_coverage == pytest.approx(0.5)
    assert total.chao_unseen == pytest.approx(2.0)
    sg4 = next(
        row for row in rows if row.scope == "all" and row.spacegroup == "4"
    )
    assert (sg4.N, sg4.S_obs, sg4.f1, sg4.f2) == (2, 2, 2, 0)
    assert sorted(
        row.energy_window_kj_mol
        for row in rows
        if row.scope == "window" and row.spacegroup == "ALL"
    ) == list(DEFAULT_WINDOWS)


def test_custom_window_uses_representative_energy(tmp_path):
    database = tmp_path / "clustered.db"
    create_clustered_database(database, "self")
    observations = load_database_observations(database)

    rows = summarize(observations, windows=[1.0], by_spacegroup=False)
    window = next(row for row in rows if row.scope == "window")

    assert (window.N, window.S_obs, window.f1, window.f2) == (2, 1, 0, 1)
    assert window.good_turing_coverage == 1.0
    assert window.energy_cutoff == -9.0


def test_combine_databases_composes_local_and_global_clusters(tmp_path):
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    global_clusters = tmp_path / "output.db"
    create_clustered_database(first, "null", prefix="x")
    create_clustered_database(second, "null", prefix="y")
    create_clustered_database(global_clusters, "null")

    connection = sqlite3.connect(global_clusters)
    connection.executescript(
        """
        DELETE FROM crystal;
        DELETE FROM trial_structure;
        DELETE FROM equivalent_to;
        """
    )
    representatives = ["xa", "xc", "xd", "ya", "yc", "yd"]
    connection.executemany(
        "INSERT INTO crystal VALUES (?, 2, 1.1, ?, 'mol', 'TITL')",
        [
            (structure_id, -10.0 + index)
            for index, structure_id in enumerate(representatives)
        ],
    )
    connection.executemany(
        "INSERT INTO trial_structure VALUES (?, 3, ?, 1, 1.0, NULL)",
        [(structure_id, index) for index, structure_id in enumerate(representatives)],
    )
    connection.executemany(
        "INSERT INTO equivalent_to VALUES (?, ?)",
        [
            ("xa", None),
            ("xa", "ya"),
            ("xc", None),
            ("xd", None),
            ("yc", None),
            ("yd", None),
        ],
    )
    connection.commit()
    connection.close()

    observations = combine_database_observations(
        [first, second], global_clusters=global_clusters
    )
    total = summarize(observations, windows=[], by_spacegroup=False)[0]

    assert (total.N, total.S_obs, total.f1, total.f2) == (8, 5, 4, 0)
    assert total.good_turing_coverage == pytest.approx(0.5)
    assert {
        item.representative_id
        for item in observations
        if item.structure_id in {"xa", "xb", "ya", "yb"}
    } == {"xa"}


def test_multiple_databases_require_global_clusters(tmp_path):
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    create_clustered_database(first, "null", prefix="x")
    create_clustered_database(second, "null", prefix="y")

    with pytest.raises(ValueError, match="--global-clusters"):
        combine_database_observations([first, second])


def test_incomplete_cluster_assignments_are_rejected_by_default(tmp_path):
    database = tmp_path / "clustered.db"
    create_clustered_database(database, "null")
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM equivalent_to WHERE unique_id = 'd'")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="no assignment"):
        load_database_observations(database)

    observations = load_database_observations(
        database, strict_assignments=False
    )
    assert {item.structure_id for item in observations} == {"a", "b", "c", "d"}


def test_bootstrap_intervals_are_reproducible(tmp_path):
    database = tmp_path / "clustered.db"
    create_clustered_database(database, "null")
    observations = load_database_observations(database)

    first = summarize(
        observations,
        windows=[],
        by_spacegroup=False,
        bootstrap_samples=100,
        random_seed=42,
    )[0]
    second = summarize(
        observations,
        windows=[],
        by_spacegroup=False,
        bootstrap_samples=100,
        random_seed=42,
    )[0]
    with_spacegroups = summarize(
        observations,
        windows=[],
        by_spacegroup=True,
        bootstrap_samples=100,
        random_seed=42,
    )[0]

    assert first == second
    assert first == with_spacegroups
    assert first.good_turing_ci_low is not None
    assert first.good_turing_ci_high is not None
    assert first.chao_S_ci_low is not None
    assert first.chao_S_ci_high is not None
    assert (
        first.good_turing_ci_low
        <= first.good_turing_coverage
        <= first.good_turing_ci_high
    )
    assert first.chao_S_ci_low <= first.chao_S <= first.chao_S_ci_high


def test_convergence_curve_ends_at_full_summary(tmp_path):
    database = tmp_path / "clustered.db"
    create_clustered_database(database, "null")
    observations = load_database_observations(database)

    curve = convergence(observations, windows=[1.0], points=3)
    full = next(
        row
        for row in curve
        if row.sample_size == 4
        and row.scope == "all"
        and row.spacegroup == "ALL"
    )

    assert [
        row.sample_size
        for row in curve
        if row.scope == "all" and row.spacegroup == "ALL"
    ] == [2, 3, 4]
    assert (full.N, full.S_obs, full.f1, full.f2) == (4, 3, 2, 1)


def test_completeness_cli_accepts_custom_windows(tmp_path):
    database = tmp_path / "clustered.db"
    output = tmp_path / "completeness.csv"
    create_clustered_database(database, "null")

    assert (
        completeness_main(
            [
                str(database),
                "--windows",
                "2.5",
                "7.5",
                "--total-only",
                "-o",
                str(output),
            ]
        )
        == 0
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["energy_window_kj_mol"] for row in rows[1:]] == ["2.5", "7.5"]


@patch("cspy.db.clustering.find_equivalent_structures")
def test_cluster_can_write_completeness_report(find_equivalent, tmp_path):
    database = tmp_path / "clustered.db"
    output = tmp_path / "cluster-completeness.csv"
    create_clustered_database(database, "null")
    find_equivalent.return_value = (str(database), {"a": [], "c": [], "d": []})

    clustering_main(
        [
            str(database),
            "-o",
            str(database),
            "-m",
            "pymatgen",
            "--completeness",
            "--completeness-windows",
            "2.5",
            "--completeness-total-only",
            "--completeness-output",
            str(output),
        ]
    )

    find_equivalent.assert_called_once()
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[1]["energy_window_kj_mol"] == "2.5"


def test_dump_cluster_info_table_only(tmp_path):
    database = tmp_path / "clustered.db"
    table_output = tmp_path / "observations.csv"
    structure_output = tmp_path / "structures.zip"
    create_clustered_database(database, "self")

    dump_main(
        [
            str(database),
            "--cluster-info",
            "--table-only",
            "--table-output",
            str(table_output),
            "--structure-output",
            str(structure_output),
        ]
    )

    with table_output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert not structure_output.exists()
    by_id = {row["id"]: row for row in rows}
    assert by_id["b"]["representative_id"] == "a"
    assert by_id["b"]["representative_spacegroup"] == "2"
    assert by_id["b"]["representative_energy"] == "-10.0"
    assert by_id["b"]["multiplicity"] == "2"
