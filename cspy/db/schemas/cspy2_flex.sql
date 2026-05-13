pragma journal_mode = wal;
pragma foreign_keys = ON;

create table if not exists distorted_molecules (
    id string PRIMARY KEY, /* Unique identifier for the distorted_molecule */
    energy float, /* Energy of each distorted molecule */
    xyz_coordinates string, /* File contents the xyz coordinates describing this molecule*/
    mults string, /* File contents the molecular multipoles */
    axes string, /* File contents the molecular axis definition (NEIGHCRYS/DMACRYS) format */
    charges string, /* File contents the molecular charges in the same format */
    res string /* res file of the molecule */
);
