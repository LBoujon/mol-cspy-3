pragma journal_mode = wal;
pragma foreign_keys = ON;

create table if not exists asym_units (
    id string PRIMARY KEY, /* Unique identifier for the asymmetric unit */
    energy float, /* Energy of asymmetric unit */
    xyz_coordinates string, /* File contents the xyz coordinates describing this asymmetric unit*/
    molecule_ids JSON DEFAULT('[]') /* List of ids of molecules in asymmetric unit*/
);

create table if not exists molecules (
    id string PRIMARY KEY, /* Unique identifier for the molecule */
    xyz_coordinates string, /* File contents the xyz coordinates describing this molecule*/
    equivalent_atoms JSON DEFAULT('[]'), /* List of symmetry equivalent atoms*/
    atom_elements JSON DEFAULT('[]') /* List of elements of atoms as ordered in xyz_coordinates*/
);