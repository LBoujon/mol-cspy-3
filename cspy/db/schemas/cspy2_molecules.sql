pragma journal_mode = wal;
pragma foreign_keys = ON;

create table if not exists molecules (
    id string PRIMARY KEY, /* Unique identifier for the distorted_molecule */
    xyz_coordinates string, /* File contents the xyz coordinates describing this molecule*/
    total_charge float, /* Total charge on molecule (ion) */
    charges string, /* File contents the molecular charges in the same format */
    mults string, /* File contents the molecular multipoles */
    axes string /* File contents the molecular axis definition (NEIGHCRYS/DMACRYS) format */
);
