pragma journal_mode = wal;
pragma foreign_keys = ON;

create table if not exists meta (
    id integer PRIMARY KEY AUTOINCREMENT,
    version string, /* version number for this database */
    description string, /* what the modification was */
    modification_time datetime DEFAULT CURRENT_TIMESTAMP /* modification time */
);

create table if not exists crystal (
    id string PRIMARY KEY, /* unique identifier for the crystal */
    spacegroup integer DEFAULT 1, /* 
        international tables number,
        but fcontents should fully
        describe setting, choice etc.
    */
    density float, /* density in g/cm^3 */
    energy float, /* total lattice energy */
    molecule_id string, /* ID of the molecule used to generate the crystal */
    file_content string /* file contents for e.g. a shelx .res file describing this crystal*/
);

create table if not exists trial_structure (
    id string, 
    minimization_step integer, /* associated minimization step */
    trial_number integer, /* associated quasirandom seed or trial number*/
    valid boolean,
    minimization_time float, /* time in secs for minimization */
    metadata string, /* information about minimization/comments */
    FOREIGN KEY(id) REFERENCES crystal(id)
);

create table if not exists descriptor (
    id string, 
    name string, /* descriptor name e.g. xrd, etc. */
    value blob, /* binary blob of the descriptor, probably np.ndarray */
    metadata string, /* how was the descriptor calculated e.g wavelength, range etc. */
    PRIMARY KEY(id, name, metadata),
    FOREIGN KEY(id) REFERENCES crystal(id)
);

create table if not exists equivalent_to (
    unique_id string,
    equivalent_id string,
    PRIMARY KEY(unique_id, equivalent_id),
    FOREIGN KEY(unique_id) REFERENCES crystal(id),
    FOREIGN KEY(equivalent_id) REFERENCES crystal(id)
);

create table if not exists trial (
    trial_id string PRIMARY KEY,
    trial_number integer,
    iterations integer DEFAULT 0,
    state string,
    metadata string
);
