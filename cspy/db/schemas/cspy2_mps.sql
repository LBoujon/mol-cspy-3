pragma journal_mode = wal;
pragma foreign_keys = ON;

create table if not exists pairs (
    id string PRIMARY KEY, /* Unique identifier for the pair */
    min_energy float, /* Minimum energy of crystals found to contain pair */
    energy_list JSON DEFAULT('[]'), /* List of energies of crystals containing pair*/
    min_density float, /* Minimum density of crystals found to contain pair */
    max_density float, /* Maximum density of crystals found to contain pair */
    median_density float, /* Median density of crystals found to contain pair */
    density_list JSON DEFAULT('[]'), /* List of densities of crystals containing pair*/
    frequency integer, /* Number of crystals found to contain pair */
    xyz_coordinates string, /* File contents the xyz coordinates describing this pair*/
    mol2_coordinates string, /* File contents the mol2 coordinates describing this pair*/
    asymmetric boolean, /* Whether the pair is from an asymmetric unit*/
    symmetric boolean, /* Whether the pair is from outside an asymmetric unit*/
    molecule1 string, /* Unique identifier for molecule1 */
    molecule2 string, /* Unique identifier for molecule1 */
    FOREIGN KEY(molecule1) REFERENCES molecules(id),
    FOREIGN KEY(molecule2) REFERENCES molecules(id)
);

create table if not exists molecules (
    id string PRIMARY KEY, /* Unique identifier for the molecule */
    xyz_coordinates string, /* File contents the xyz coordinates describing this molecule*/
    equivalent_atoms JSON DEFAULT('[]') /* List of symmetry equivalent atoms*/
);