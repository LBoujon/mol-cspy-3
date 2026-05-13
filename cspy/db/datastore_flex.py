from collections import defaultdict
import logging
import json
import sqlite3
from .schemas import current_flex as CURRENT_SCHEMA_FLEX

LOG = logging.getLogger(__name__)


class DataStoreFlex:

    _SCHEMA_FLEX = ""

    def __init__(self, filename, connect=True, execute_schema=False):
        """Open a sqlite3 database file for flexibility, connecting if required"""
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
    
    import time

    def connect(self, execute_schema=False, retries=5, **kwargs):
        """Establish a connection to the database,
        returning a cursor for that connection. If a
        connection already exists, use that one"""

        if self.connected:
            return self.connection.cursor()

        LOG.debug("Establishing connection to database %s", self.filename)

        for attempt in range(retries):
            try:
                conn = sqlite3.connect(self.filename, **kwargs)
                if execute_schema:
                    conn.executescript(self._SCHEMA_FLEX)
                self._connection = conn
                return self.cursor()
            except sqlite3.DatabaseError as e:
                LOG.error("Error establishing connection to %s: %s", self.filename, e)
                if "database is locked" in str(e).lower() and attempt < retries - 1:
                    LOG.debug("Retrying connection attempt %d", attempt + 1)
                else:
                    raise e

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
      #  LOG.debug("Loaded table_names: %s", self.table_names)

    def guess_schema_version(self):
        """Figure out which version of cspy was used to create this database"""
        self.n_tables = len(self.table_names)
        if self.table_exists("distorted_molecules"):
            self.schema_version = "cspy2_flex"
        if self.table_exists("res_data"):
            self.schema_version = "deprecated"
        if self.table_exists("meta"):
            self.schema_version = "cspy2"
            if not self.table_exists("crystal"):
                self.schema_version = "intermediate"
     #   LOG.debug("Guessed schema version: '%s'", self.schema_version)
        return self.schema_version

    def create_tables(self):
        """Connect to the database, create each table in the schema,
        then load the table names and guess the schema version"""
        cur = self.cursor()
        LOG.debug("Creating tables from schema")
        cur.executescript(self._SCHEMA_FLEX)
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
#------ Is rows a dictionary?
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
          #  columns = ", ".join("{}.{}".format(table, col) for col in columns)
            columns = ", ".join("{}".format(col) for col in columns)
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
        dstore = DataStoreFlex(filename, connect=True, execute_schema=True)
        return dstore


class CspDataStoreFlex(DataStoreFlex):

    _SCHEMA_FLEX = CURRENT_SCHEMA_FLEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, connect=True, execute_schema=True, **kwargs)

    def update_unique_structures(self, equivalence):
        pass

    def unique_structures(self, with_trial_data=False, fallback=False):
        num_unique = self.query("select count(*) from equivalent_to").fetchone()[0]
        if num_unique == 0:
            LOG.info(
                "No data in equivalent_to table: run `cspy-db cluster` to populate this"
            )
            if fallback:
                LOG.info("Falling back to final minimizations")
                return self.final_minimizations(with_trial_data=with_trial_data)
            else:
                return []
        else:
            query_text = (
                "select C.* from crystal C join "
                "(select distinct unique_id from equivalent_to) "
                "on unique_id == C.id "
                "order by energy asc"
            )
            if with_trial_data:
                query_text = (
                    "select C.*, T.minimization_step, T.trial_number, T.minimization_time "
                    "from crystal C join "
                    "(select distinct unique_id from equivalent_to) "
                    "on unique_id == C.id"
                    " join trial_structure T on C.id = T.id "
                    "order by energy asc"
                )
        return self.query(query_text)

    def add_flex_results(self, results):
        distorted_molecules = defaultdict(list)

#------ Let's create a dictionary from results that is a tupla 
#       with 'id' and 'energy' as keys, and the values of all id's and their 
#       corresponding energies will be ir values will be in a list
        for i, r in results.items():
            distorted_molecules["id"].append(i)
            distorted_molecules["energy"].append(r[-5])
            distorted_molecules["xyz_coordinates"].append(r[-6])
            distorted_molecules["mults"].append(r[-4])
            distorted_molecules["axes"].append(r[-3])
            distorted_molecules["charges"].append(r[-2])
            distorted_molecules["res"].append(r[-1])

        columns, values = zip(*distorted_molecules.items())

        self.insert_many("distorted_molecules", distorted_molecules, replace=True)
        self.commit()

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
