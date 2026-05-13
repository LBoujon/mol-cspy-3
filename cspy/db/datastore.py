from collections import defaultdict
import logging
import json
import sqlite3
from .schemas import current as CURRENT_SCHEMA
from typing import Any, Iterable, List, Tuple, Union

LOG = logging.getLogger(__name__)


class DataStore:

    _SCHEMA = ""

    def __init__(self, filename, connect=True, execute_schema=False):
        """Open an sqlite3 database file, connecting if required"""
        assert isinstance(filename, str)
        self.filename = filename
        self._connection = None
        self._cursor = None
        self.table_names = set()
        self.schema_version = "unknown"

        if connect:
            self.connect(execute_schema=execute_schema)
            self.load_table_names()
            self.guess_schema_version()

    @property
    def connected(self):
        """Check if we have a connection to the database"""
        return self.connection is not None

    @property
    def connection(self):
        return self._connection

    def connect(self, execute_schema=False, **kwargs):
        """Establish a connection to the database,
        returning a cursor for that connection. If a
        connection already exists, use that one"""

        if self.connected:
            return self.connection.cursor()

        LOG.debug("Establishing connection to database %s", self.filename)
        conn = sqlite3.connect(self.filename, timeout=30, **kwargs)
        if execute_schema:
            try:
                conn.executescript(self._SCHEMA)
            except sqlite3.DatabaseError as e:
                LOG.error("Error establishing connection to %s: %s", self.filename, e)
                raise e
        self._connection = conn
        return self.cursor()

    def commit(self):
        """Force a commit on the database connection"""
        if self.connected:
            self.connection.commit()

    def table_exists(self, table_name):
        """Check if a table with the given name exists"""
        return table_name in self.table_names

    def cursor(self, new=False):
        assert self.connected, "Must connect to a database before acquiring cursor"
        if new:
            return self.connection.cursor()
        if not hasattr(self, "_cursor"):
            self._cursor = self.connection.cursor()
        elif self._cursor is None:
            self._cursor = self.connection.cursor()
        return self._cursor

    def load_table_names(self):
        """Load the set of current table names in the database"""
        cur = self.cursor()
        cur.execute("select name from sqlite_master where type='table'")
        self.table_names = {row[0] for row in cur.fetchall()}
        LOG.debug("Loaded table_names: %s", self.table_names)

    def guess_schema_version(self):
        """Figure out which version of cspy was used to create this database"""
        self.n_tables = len(self.table_names)
        if self.table_exists("res_data"):
            self.schema_version = "deprecated"
        if self.table_exists("meta"):
            self.schema_version = "cspy2"
            if not self.table_exists("crystal"):
                self.schema_version = "intermediate"
        LOG.debug("Guessed schema version: '%s'", self.schema_version)
        return self.schema_version

    def create_tables(self):
        """Connect to the database, create each table in the schema,
        then load the table names and guess the schema version"""
        cur = self.cursor()
        LOG.debug("Creating tables from schema")
        cur.executescript(self._SCHEMA)
        self.load_table_names()
        self.guess_schema_version()

    def close(self):
        """Close the database connection, clearing class variables"""
        self.disconnect()

    def disconnect(self):
        """Close the database connection, clearing class variables"""
        if self.connected:
            self.connection.close()
        self._cursor = None
        self._connection = None

    def insert_one(self, table, row, columns=None, commit=True, replace=False):
        stmt = "insert {replace} into {table} {columns} values ({placeholders})"
        if isinstance(row, dict):
            columns, values = zip(*row.items())
            placeholders = ",".join("?" for _ in values)
            columns = "({})".format(",".join(columns))
        else:
            values = row
            columns = "(" + ",".join(columns) + ")" if columns else ""
            placeholders = ",".join("?" for _ in values)

        return self.query(
            stmt.format(
                table=table,
                columns=columns,
                placeholders=placeholders,
                replace="or replace" if replace else "",
            ),
            values,
            commit=commit,
        )

    def insert_many(self, table, rows, columns=None, commit=True, replace=False):
        stmt = "insert {replace} into {table} {columns} values ({placeholders})"
        if isinstance(rows, dict):
            columns, values = zip(*rows.items())
            placeholders = ",".join("?" for _ in values)
            values = (x for x in zip(*values))
            columns = "({})".format(",".join(columns))
        else:
            values = list(rows)
            assert len(values) > 0
            columns = "(" + ",".join(columns) + ")" if columns else ""
            placeholders = ",".join("?" for _ in values[0])

        return self.query_many(
            stmt.format(
                table=table,
                columns=columns,
                placeholders=placeholders,
                replace="or replace" if replace else "",
            ),
            values,
            commit=commit,
        )

    def select(self, table, columns=None, additional=""):
        stmt = "select {columns} from {table} {additional}"
        if columns is not None:
            columns = ", ".join("{}.{}".format(table, col) for col in columns)
        else:
            columns = "*"
        return self.query(
            stmt.format(columns=columns, table=table, additional=additional)
        )

    def query(self, query_text, *args, commit=True, **kwargs):
        """Query the database with a custom query"""
        response = self.cursor().execute(query_text, *args, **kwargs)
        if commit:
            self.commit()
        return response

    def query_many(self, query_text, *args, commit=True):
        """Query the database with a custom query"""
        response = self.cursor().executemany(query_text, *args)
        if commit:
            self.commit()
        return response

    @staticmethod
    def create_and_connect(filename):
        """Create a new datastore object, establishing
        connection and creating tables"""
        dstore = DataStore(filename, connect=True, execute_schema=True)
        return dstore
    
    def get_table_columns(self, table: str) -> List[str]:
        """
        Get the column names of a table in the database.

        Arguments
        ---------

        table : str
            The table name in the database

        Returns
        -------

        columns : List[str]
            A list of column names in the table
        """
        query_text = f"PRAGMA table_info({table})"
        return [row[1] for row in self.query(query_text).fetchall()]

    def copy_schema_to(self, filename):
        import os
        if os.path.exists(filename):
            LOG.info(f"{filename} exists. Won't overwrite existing file.")
            return
        else:
            destination_ds = self.create_and_connect(filename)
            destination_ds.connect()
        for table in self.table_names:
            if table == 'sqlite_sequence':
                continue
            columns_info = self.query(f"PRAGMA table_info({table});").fetchall()
            column_definitions = [f"{column_info[1]} {column_info[2]}" for column_info in columns_info]
            columns_definition_str = ', '.join(column_definitions)
            create_table_query = f"CREATE TABLE IF NOT EXISTS {table} ({columns_definition_str});"
            destination_ds.query(create_table_query)
        LOG.info(f"Schema of {self.filename} was successfully copied to {destination_ds.filename}.")
        destination_ds.close()

    def copy_schema_to(self, filename):
        import os
        if os.path.exists(filename):
            LOG.info(f"{filename} exists. Won't overwrite existing file.")
            return
        else:
            destination_ds = self.create_and_connect(filename)
            destination_ds.connect()
        for table in self.table_names:
            if table == 'sqlite_sequence':
                continue
            columns_info = self.query(f"PRAGMA table_info({table});").fetchall()
            column_definitions = [f"{column_info[1]} {column_info[2]}" for column_info in columns_info]
            columns_definition_str = ', '.join(column_definitions)
            create_table_query = f"CREATE TABLE IF NOT EXISTS {table} ({columns_definition_str});"
            destination_ds.query(create_table_query)
        LOG.info(f"Schema of {self.filename} was successfully copied to {destination_ds.filename}.")
        destination_ds.close()


class CspDataStore(DataStore):

    _SCHEMA = CURRENT_SCHEMA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, connect=True, execute_schema=True, **kwargs)

    def update_unique_structures(self, equivalence):
        pass

    def unique_structures(self, with_file_content=False, with_trial_data=False, fallback=False, **kwargs) -> sqlite3.Cursor:
        """
        Get the unique structures in the database crystal table. To get the N lowest energy structures
        you can use fetchmany(N) when consuming the cursor, instead of using fetchall().

        Arguments
        ---------

        with_file_content : bool
            Return the file_content column from the crystal table. Default is False

        with_trial_data : bool
            Include the trial_structure data with each crystal. Default is False

        fallback : bool
            If database has not been clustered, return final minimizations instead. Default
            is False

        **kwargs : Dict[str, Any]
            Extra filters to the SQL query. See notes below for accepted keys
        
        Returns
        -------

        cursor : sqlite3.Cursor
            An sqlite3 cursor with the defined query 

        Raises
        ------

        ValueError
            When both `included_spgs` and `excluded_spgs` kwargs are passed

        Notes
        -----

        Available kwargs:

        - `included_spgs` (List[int]): Returned crystals will only be in these space groups. Mutually
        exclusive option with `excluded_spgs`
        - `excluded_spgs` (List[int]): Returned crystals will not be in these space groups. Mutually
        exclusive option with `included_spgs`
        - `max_energy` (float): Only structures within `max_energy` from the global energy minimum will
        be returned
        """
        num_equivalent_to = self.query("select count(*) from equivalent_to").fetchone()[0]
        if num_equivalent_to == 0:
            LOG.info(
                "No data in equivalent_to table: run `cspy-db cluster` to populate this"
            )
            if fallback:
                LOG.info("Falling back to final minimizations")
                return self.final_minimizations(with_file_content=with_file_content, with_trial_data=with_trial_data, **kwargs)
            else:
                return []
        else:
            extra_filter_txt = []
            included_spgs = kwargs.get("included_spgs", None)
            excluded_spgs = kwargs.get("excluded_spgs", None)
            if included_spgs is not None and excluded_spgs is not None:
                raise ValueError("Cannot set both `included_spgs` and `excluded_spgs` in kwargs")
            if included_spgs is not None:
                spgs_txt = ", ".join([str(sg) for sg in included_spgs])
                extra_filter_txt.append(f"C.spacegroup in ({spgs_txt}) ")
            if excluded_spgs is not None:
                spgs_txt = ", ".join([str(sg) for sg in excluded_spgs])
                extra_filter_txt.append(f"C.spacegroup not in ({spgs_txt}) ")
            max_energy = kwargs.get("max_energy", None)
            if max_energy is not None:
                extra_filter_txt.append(f"C.energy <= (select min(energy) from crystal) + {max_energy} ")

            extra_txt = ""
            if len(extra_filter_txt) > 0:
                extra_txt = "where "
                extra_txt += "and ".join(extra_filter_txt)

            if with_file_content:
                select_C = "select C.*"
            else:
                select_C = "select C.id, C.spacegroup, C.density, C.energy, C.molecule_id"

            query_text = (
                select_C + " from crystal C join "
                "(select distinct unique_id from equivalent_to) "
                "on unique_id == C.id " + extra_txt +
                "order by energy asc"
            )
            if with_trial_data:
                query_text = (
                    select_C + ", T.minimization_step, T.trial_number, T.minimization_time "
                    "from crystal C join "
                    "(select distinct unique_id from equivalent_to) "
                    "on unique_id == C.id"
                    " join trial_structure T on C.id = T.id " + extra_txt +
                    "order by energy asc"
                )
        return self.query(query_text)
    
    def number_of_unique_structures(self):
        # We just want the number of structures and file_content increases the RAM cost
        return len(list(self.unique_structures(with_file_content=False)))

    def get_total_minimization_times(self):
        query_text = (
            "select trial_number, sum(minimization_time) from trial_structure "
            "group by trial_number"
        )
        return self.query(query_text)

    def equivalent_trial_numbers(self):
        query_text = (
            "select t1.trial_number, coalesce(t2.trial_number, -1)"
            "from equivalent_to "
            "left join trial_structure t1 on (t1.id = equivalent_to.unique_id) "
            "left join trial_structure t2 on (t2.id = equivalent_to.equivalent_id)"
        )
        return self.query(query_text)

    def equivalent_ids(self) -> list[Tuple[str, str]]:
        """ For each non-unique crystal structure in database,
            return a tuple of the equivalent unique crystal 
            structure's ID, and the non-unique crystal structure
            ID.

            Parameters
            ----------
            None

            Returns
            -------
            list[Tuple[str, str]]
                Each tuple in the list represents a unique pair.
                The first element in the tuple is a unique crystal
                structure ID. This will appear multiple times.
                The second element is an equivalent crystal structure
                ID that was removed by clustering.
            """
        query_text = (
            "select t1.id, coalesce(t2.id, -1)"
            "from equivalent_to "
            "left join trial_structure t1 on (t1.id = equivalent_to.unique_id) "
            "left join trial_structure t2 on (t2.id = equivalent_to.equivalent_id)"
        )
        return self.query(query_text)

    def final_minimization_step(self):
        return self.query(
            "select max(minimization_step) from trial_structure"
        ).fetchone()[0]

    def final_minimizations(self, with_file_content=True, with_trial_data=False, **kwargs) -> sqlite3.Cursor:
        """
        Get the structures from the final minimization step from the crystal table. To get the N lowest energy structures
        you can use fetchmany(N) when consuming the cursor, instead of using fetchall().

        Arguments
        ---------

        with_file_content : bool
            Return the file_content column from the crystal table. Default is False

        with_trial_data : bool
            Include the trial_structure data with each crystal. Default is False

        **kwargs : Dict[str, Any]
            Extra filters to the SQL query. See notes below for accepted keys
        
        Returns
        -------

        cursor : sqlite3.Cursor
            An sqlite3 cursor with the defined query 

        Raises
        ------

        ValueError
            When both `included_spgs` and `excluded_spgs` kwargs are passed

        Notes
        -----

        Available kwargs:

        - `included_spgs` (List[int]): Returned crystals will only be in these space groups. Mutually
        exclusive option with `excluded_spgs`
        - `excluded_spgs` (List[int]): Returned crystals will not be in these space groups. Mutually
        exclusive option with `included_spgs`
        - `max_energy` (float): Only structures within `max_energy` from the global energy minimum will
        be returned
        """
        max_step = self.final_minimization_step()
        if max_step is None:
            return []
        
        if with_file_content:
            select_C = "select C.*"
        else:
            select_C = "select C.id, C.spacegroup, C.density, C.energy, C.molecule_id"

        extra_filter_txt = []
        included_spgs = kwargs.get("included_spgs", None)
        excluded_spgs = kwargs.get("excluded_spgs", None)
        if included_spgs is not None and excluded_spgs is not None:
            raise ValueError("Cannot set both `included_spgs` and `excluded_spgs` in kwargs")
        if included_spgs is not None:
            spgs_txt = ", ".join([str(sg) for sg in included_spgs])
            extra_filter_txt.append(f"C.spacegroup in ({spgs_txt}) ")
        if excluded_spgs is not None:
            spgs_txt = ", ".join([str(sg) for sg in excluded_spgs])
            extra_filter_txt.append(f"C.spacegroup not in ({spgs_txt}) ")
        max_energy = kwargs.get("max_energy", None)
        if max_energy is not None:
            extra_filter_txt.append(f"C.energy <= (select min(energy) from crystal) + {max_energy} ")

        extra_txt = ""
        if len(extra_filter_txt) > 0:
            extra_txt = "where "
            extra_txt += "and ".join(extra_filter_txt)

        query_text = (
            select_C + " from crystal C join "
            "(select id from trial_structure where minimization_step >= {max_step}) "
            "T on C.id = T.id " + extra_txt +
            "order by energy asc"
        ).format(max_step=max_step)

        if with_trial_data:
            query_text = (
                select_C + ", T.minimization_step, T.trial_number, T.minimization_time "
                "from crystal C join "
                "(select * from trial_structure where minimization_step >= {max_step}) "
                "T on C.id = T.id " + extra_txt +
                "order by energy asc"
            ).format(max_step=max_step)

        return self.query(query_text, commit=False)

    def number_of_final_minimizations(self):
        max_step = self.final_minimization_step()
        if max_step is None:
            return 0
        query_text = (
            "select count(C.id) from crystal C join "
            "(select id from trial_structure where minimization_step >= {max_step}) "
            "T on C.id = T.id "
        ).format(max_step=max_step)
        return list(self.query(query_text))[0][0]
        
    def step_minimizations(self, step, with_file_content=True, with_trial_data=False):
        max_step = self.final_minimization_step()
        if max_step < step:
            LOG.error("Looking for minimisations at step %d but max step is %d. Returning empty list.", step, max_step)
            return []
        
        if with_file_content:
            select_C = "select C.*"
        else:
            select_C = "select C.id, C.spacegroup, C.density, C.energy, C.molecule_id"

        query_text = (
            select_C + " from crystal C join "
            "(select id from trial_structure where minimization_step == {step}) "
            "T on C.id = T.id "
            "order by energy asc"
        ).format(step=step)

        if with_trial_data:
            query_text = (
                select_C + ", T.minimization_step, T.trial_number, T.minimization_time "
                "from crystal C join "
                "(select * from trial_structure where minimization_step == {step}) "
                "T on C.id = T.id "
            ).format(step=step)

        return self.query(query_text, commit=False)

    def copy_unique_structures_to(self, dbname):
        if self.has_unique_structure_information():
            other = CspDataStore(dbname)
            other.close()
            self.query(f"attach '{dbname}' as tmp")
            self.query(
                "insert or replace into tmp.crystal select crystal.* from crystal join "
                "(select distinct unique_id from equivalent_to) "
                "on unique_id == crystal.id"
            )
            self.query(
                "insert or replace into tmp.descriptor select descriptor.* from descriptor join "
                "(select distinct unique_id from equivalent_to) "
                "on unique_id == descriptor.id"
            )
            self.query(
                "insert or replace into tmp.trial_structure select trial_structure.* from trial_structure join "
                "(select distinct unique_id from equivalent_to) "
                "on unique_id == trial_structure.id"
            )
            self.commit()
            self.query("detach tmp")
        else:
            LOG.error("%s has no unique structure information. Cannot comply.", dbname)

    def copy_unique_structures_within_range(self, dbname, 
                                            energy_min: float = None, energy_max: float = None, 
                                            density_min: float = None, density_max: float = None) -> None:
        
        """
        Take all crystal structures within specified energy-density range and copy to a new database.

        Arguments
        ---------

        dbname : string
            Name of new database to copy data to

        energy_min : float
            Do not copy structures below this relative energy value. If not specified, no minimum is enforced.

        energy_max : float
            Do not copy structures above this relative energy value. If not specified, no maximum is enforced.

        density_min : float
            Do not copy structures below this density value. If not specified, no minimum is enforced.

        density_max : float
            Do not copy structures above this density value. If not specified, no maximum is enforced.
        """
        
        other = CspDataStore(dbname)
        other.close()
        self.query(f"attach '{dbname}' as tmp")

        conditions = []

        if energy_min is not None:
            conditions.append(("crystal.energy >= (select min(crystal.energy) from crystal) + {} "
                                ).format(energy_min))
        if energy_max is not None:
            conditions.append(("crystal.energy <= (select min(crystal.energy) from crystal) + {} "
                                ).format(energy_max))
        if density_min is not None:
            conditions.append(("crystal.density >= {} "
                                ).format(density_min))
        if density_max is not None:
            conditions.append(("crystal.density <= {} "
                                ).format(density_max))

        if conditions:
            property_range_sql = " where " + "and ".join(conditions)
    
            if self.has_unique_structure_information():
                self.query(
                    "insert or replace into tmp.crystal select crystal.* from crystal join "
                    "(select distinct unique_id from equivalent_to) "
                    "on unique_id == crystal.id "+ property_range_sql
                )
            else:
                LOG.warning("%s has no unique structure information. Copying ALL structures within specified range.", dbname)
                self.query(
                    "insert or replace into tmp.crystal select crystal.* from crystal "+ property_range_sql
                )
            self.query(
                "insert or replace into tmp.descriptor select descriptor.* from descriptor "
                "where descriptor.id in  (select id from tmp.crystal)" 
            )
            self.query(
                "insert or replace into tmp.trial_structure select trial_structure.* from trial_structure "
                "where trial_structure.id in (select id from tmp.crystal)"
            )
            self.commit()
            self.query("detach tmp") 
        else:
            LOG.error("No tolerances provided. Must specify energy_min, energy_max, density_min, or density_max. Refusing to proceed...")

    def copy_equivalence_table_to(self, dbname):
        if self.has_unique_structure_information():
            other = CspDataStore(dbname)
            other.close()
            self.query(f"attach '{dbname}' as tmp")
            self.query(
                "insert or replace into tmp.equivalent_to select * from equivalent_to"
            )
            self.commit()
            self.query("detach tmp")
        else:
            LOG.error("%s has no unique structure information. Cannot comply.", dbname)

    def crystals(self):
        return self.select("crystal")

    def add_equivalent_structures(self, equivalent_structures):
        rows = []
        for k, v in equivalent_structures.items():
            if len(v):
                rows += [(k, x) for x in set(v)]
            else:
                rows.append((k, None))
        self.query("delete from equivalent_to")
        self.insert_many("equivalent_to", rows, columns=("unique_id", "equivalent_id"))
        nuniques = self.query(
            "select count(distinct unique_id) from equivalent_to"
        ).fetchone()[0]
        ndups = self.query(
            "select count(distinct equivalent_id) from equivalent_to"
        ).fetchone()[0]
        LOG.debug("%s has %d uniques, %d duplicates", self.filename, nuniques, ndups)
        self.commit()

    def add_csp_results(self, results):
        crystals = defaultdict(list)
        molecule_to_crystals = defaultdict(list)
        trial_structures = defaultdict(list)
        descriptors = defaultdict(list)

        for r in results:
            crystals["id"].append(r.id)
            crystals["spacegroup"].append(r.spacegroup)
            crystals["energy"].append(r.energy)
            crystals["density"].append(r.density)
#RC 
            try:
                crystals["molecule_id"].append(r.molecule_id)
            except Exception as e:
                crystals["molecule_id"].append('no mol_id exists')
            crystals["file_content"].append(r.file_content)
            trial_structures["id"].append(r.id)
            trial_structures["minimization_step"].append(r.minimization_step)
            trial_structures["trial_number"].append(r.trial_number)
            trial_structures["valid"].append(True)
            trial_structures["minimization_time"].append(r.time)
            if r.xrd is not None:
                descriptors["id"].append(r.id)
                descriptors["name"].append("xrd")
                descriptors["value"].append(r.xrd)
                descriptors["metadata"].append("{'two_theta': (0, 20), 'sep': 0.02}")

        if crystals:
            self.insert_many("crystal", crystals, replace=True)
        if trial_structures:
            self.insert_many("trial_structure", trial_structures, replace=True)
        if descriptors:
            self.insert_many("descriptor", descriptors, replace=True)
        self.commit()

    def add_qrbh_results(self, results):
        crystals = defaultdict(list)
        molecule_to_crystals = defaultdict(list)
        trial_structures = defaultdict(list)
        descriptors = defaultdict(list)

        for r in results:
            crystals["id"].append(r.id)
            crystals["spacegroup"].append(r.spacegroup)
            crystals["energy"].append(r.energy)
            crystals["density"].append(r.density)
#RC         
            try:
                crystals["molecule_id"].append(r.molecule_id)
            except Exception as e:
                crystals["molecule_id"].append('no mol_id exists')
            crystals["file_content"].append(r.file_content)
            trial_structures["id"].append(r.id)
            trial_structures["minimization_step"].append(r.minimization_step)
            trial_structures["trial_number"].append(r.trial_number)
            trial_structures["valid"].append(r.accept)
            trial_structures["minimization_time"].append(r.time)
            meta = {"mc_step": r.mc_step, "unique_index": r.unique_index}
            if hasattr(r, 'metadata'):
                meta.update(r.metadata)
            trial_structures["metadata"].append(json.dumps(meta))
            if r.xrd is not None:
                descriptors["id"].append(r.id)
                descriptors["name"].append("xrd")
                descriptors["value"].append(r.xrd)
                descriptors["metadata"].append("{'two_theta': (0, 20), 'sep': 0.02}")

        if crystals:
            self.insert_many("crystal", crystals, replace=True)
        if trial_structures:
            self.insert_many("trial_structure", trial_structures, replace=True)
        if descriptors:
            self.insert_many("descriptor", descriptors, replace=True)
        self.commit()

    def add_trials(self, trials):
        trials_dict = defaultdict(list)
        for trial in trials:
            id, seed, iterations, state, meta = trial
            trials_dict["trial_id"].append(id)
            trials_dict["trial_number"].append(seed)
            trials_dict["iterations"].append(iterations)
            trials_dict["state"].append(state)
            trials_dict["metadata"].append(json.dumps(meta))
        if trials_dict:
            self.insert_many("trial", trials_dict, replace=True)

    def add_crystal_structures(self, structures, **kwargs):
        self.insert_many("crystal", structures, **kwargs)
        self.add_metadata("added {} crystals".format(len(structures)))

    def add_descriptors(self, kind, values, metadata="", **kwargs):
        descriptors = defaultdict(list)
        for i, val in values.items():
            descriptors["id"].append(i)
            descriptors["value"].append(sqlite3.Binary(val))
            descriptors["name"].append(kind)
            descriptors["metadata"].append(metadata)

        self.insert_many("descriptor", descriptors, **kwargs)
        self.add_metadata("added {} descriptors".format(len(values)))
        self.commit()

    def add_metadata(self, description):
        self.insert_one(
            "meta", {"version": self.schema_version, "description": description}
        )

    def has_unique_structure_information(self):
        return (
            self.query("select count(*) from equivalent_to limit 1").fetchone()[0] > 1
        )
    

    def equivalent_structures(self, unique_id: str) -> List[str]:
        """
        Return the crystal IDs of the structures that are
        equivalent to the structure with `unique_id`.

        Arguments
        ---------

        unique_id : str
            The unique_id of a structure

        Returns
        -------

        matches : List[str]
            A list of the crystal IDs that are equivalent to `unique_id`
        """
        matches = [x[0] for x in self.query("select T.equivalent_id from equivalent_to T where T.unique_id = '{}'".format(unique_id)).fetchall() if x[0] is not None]
        return matches
    

    def get_data_from_table(self, table: str, columns: Union[Iterable[str], None] = None) -> List[Tuple[Any, ...]]:
        """
        Get data from a table in the database. You can select which columns
        to get by passing them through `columns` argument, or set it to None and
        all columns in the table will be returned.

        Arguments
        ---------

        table : str
            The table name in the database

        columns : Union[Iterable[str], None]
            The column names to return. If set to None all data from the table is
            returned. Default is None

        Returns
        -------
        data : List[Tuple[Any, ...]]
            A list containing the data of the table
        """
        if columns is None:
            cols_txt = "*"
        else:
            cols_txt = ', '.join(columns)

        data = self.query(f"select {cols_txt} from {table}").fetchall()
        return data


    def get_descriptor_data(self, columns: Union[Iterable[str], None] = None) -> List[Tuple[Any, ...]]:
        """
        Get data from the descriptor table. You can filter which columns you 
        want in the output by passing them through the `columns` parameter. If 
        no column names are passed, it will return all the data of the table.

        The column names in CSPy 2.0 schema are: id, name, value, metadata

        Arguments
        ---------

        columns : Union[Iterable[str], None]
            The column names to return. If set to None all data from the table is
            returned. Default is None

        Returns
        -------
        descriptor_data : List[Tuple[Any, ...]]
            A list containing the data of the table
        """        
        descriptor_data = self.get_data_from_table("descriptor", columns)
        return descriptor_data


    def get_trial_data(self, columns: Union[Iterable[str], None] = None) -> List[Tuple[Any, ...]]:
        """
        Get data from the trial table. You can filter which columns you 
        want in the output by passing them through the `columns` parameter. If 
        no column names are passed, it will return all the data of the table.

        The column names in CSPy 2.0 schema are: trial_id, trial_number, iterations,
        state, metadata

        Arguments
        ---------

        columns : Union[Iterable[str], None]
            The column names to return. If set to None all data from the table is
            returned. Default is None

        Returns
        -------
        trial_data : List[Tuple[Any, ...]]
            A list containing the data of the table
        """
        trial_data = self.get_data_from_table("trial", columns)
        return trial_data    
    

    def get_trial_structure_data(self, columns: Union[Iterable[str], None] = None) -> List[Tuple[Any, ...]]:
        """
        Get data from the trial_structure table. You can filter which columns you 
        want in the output by passing them through the `columns` parameter. If 
        no column names are passed, it will return all the data of the table.

        The column names in CSPy 2.0 schema are: id, minimization_step, trial_number,
        valid, minimization_time, metadata

        Arguments
        ---------

        columns : Union[Iterable[str], None]
            The column names to return. If set to None all data from the table is
            returned. Default is None

        Returns
        -------
        trial_structure_data : List[Tuple[Any, ...]]
            A list containing the data of the table
        """
        trial_structure_data = self.get_data_from_table("trial_structure", columns)
        return trial_structure_data    
    

    def get_crystal_data(self, columns: Union[Iterable[str], None] = None) -> List[Tuple[Any, ...]]:
        """
        Get data from the crystal table. You can filter which columns you 
        want in the output by passing them through the `columns` parameter. If 
        no column names are passed, it will return all the data of the table.

        The column names in CSPy 2.0 schema are: id, spacegroup, density, energy,
        molecule_id, file_content

        Arguments
        ---------

        columns : Union[Iterable[str], None]
            The column names to return. If set to None all data from the table is
            returned. Default is None

        Returns
        -------
        crystal_data : List[Tuple[Any, ...]]
            A list containing the data of the table
        """
        crystal_data = self.get_data_from_table("crystal", columns)
        return crystal_data   
    

    def get_equivalent_to_data(self, columns: Union[Iterable[str], None] = None) -> List[Tuple[Any, ...]]:
        """
        Get data from the equivalent_to table. You can filter which columns you 
        want in the output by passing them through the `columns` parameter. If 
        no column names are passed, it will return all the data of the table.

        The column names in CSPy 2.0 schema are: unique_id, equivalent_id

        Arguments
        ---------

        columns : Union[Iterable[str], None]
            The column names to return. If set to None all data from the table is
            returned. Default is None

        Returns
        -------
        equivalent_to_data : List[Tuple[Any, ...]]
            A list containing the data of the table
        """
        equivalent_to_data = self.get_data_from_table("equivalent_to", columns)
        return equivalent_to_data   
    

    def get_meta_data(self, columns: Union[Iterable[str], None] = None) -> List[Tuple[Any, ...]]:
        """
        Get data from the meta table. You can filter which columns you 
        want in the output by passing them through the `columns` parameter. If 
        no column names are passed, it will return all the data of the table.

        The column names in CSPy 2.0 schema are: id, version, description, modification_time

        Arguments
        ---------

        columns : Union[Iterable[str], None]
            The column names to return. If set to None all data from the table is
            returned. Default is None

        Returns
        -------
        meta_data : List[Tuple[Any, ...]]
            A list containing the data of the table
        """
        meta_data = self.get_data_from_table("meta", columns)
        return meta_data
    
