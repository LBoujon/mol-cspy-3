from pathlib import Path

# Get location of this file
init_location = Path(__file__).parent.resolve()

# Find list of custom calculators
custom_calculators = {
    file.stem: str(file)
    for file in init_location.glob("*.py")
    if file.name != "__init__.py"
}

other_models = {
    "mace": {
        "energy": "eV",
        "length": "Ang",
        "normalisation": "cell",
        "energy_corr": None,
    }
}