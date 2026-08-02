from cspy.cspympi.cspy_tasks import (
    _cleanup_flexible_setup_files,
    _mean_conformational_energy_correction,
)


def test_flexible_energy_correction_averages_each_molecule_once():
    correction = _mean_conformational_energy_correction(
        energies=[10.0, 20.0], minimum_energies=[5.0, 10.0]
    )

    assert correction == 7.5


def test_flexible_cleanup_is_seed_exact_and_accepts_duplicate_names(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    own_file = tmp_path / "14_1464_conformer.xyz"
    other_seed_file = tmp_path / "14_14642_conformer.xyz"
    own_file.write_text("temporary", encoding="utf-8")
    other_seed_file.write_text("must survive", encoding="utf-8")

    _cleanup_flexible_setup_files(
        [own_file.name, own_file.name], spacegroup=14, seed=1464
    )

    assert not own_file.exists()
    assert other_seed_file.read_text(encoding="utf-8") == "must survive"
