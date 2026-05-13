from collections import defaultdict
import logging
import json
import sqlite3
from .schemas import current_mps as CURRENT_SCHEMA_MPS

LOG = logging.getLogger(__name__)


class DataStoreMPS:

    _SCHEMA_MPS = ""

    def __init__(self, filename, connect=True, execute_schema=False):
        """Open a sqlite3 database file for molecular pair seeding, connecting if required"""
        assert isinstance(filename, str)
        self.filename = filename
        self._connection = None
        self._cursor = None
        self.table_names = set()
        self.schema_version = "cspy2_mps"

        if connect:
            self.connect(execute_schema=execute_schema)
            self.load_table_names()

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

      #  LOG.debug("Establishing connection to database %s", self.filename)
        conn = sqlite3.connect(self.filename, **kwargs)
        if execute_schema:
            try:
                conn.executescript(self._SCHEMA_MPS)
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
      #  LOG.debug("Loaded table_names: %s", self.table_names)

    def create_tables(self):
        """Connect to the database, create each table in the schema,
        then load the table names and guess the schema version"""
        cur = self.cursor()
        LOG.debug("Creating tables from schema")
        cur.executescript(self._SCHEMA_MPS)
        self.load_table_names()

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
        dstore = DataStoreMPS(filename, connect=True, execute_schema=True)
        return dstore


class CspDataStoreMPS(DataStoreMPS):

    _SCHEMA_MPS = CURRENT_SCHEMA_MPS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, connect=True, execute_schema=True, **kwargs)

    def add_mps_properties(self, properties):
        import json
        pairs = defaultdict(list)

#------ Let's create a dictionary from results that is a tupla 
#       with 'id' and 'energy' as keys, and the values of all id's and their 
#       corresponding energies will be ir values will be in a list
        for i, p in properties.items():
            pairs["id"].append(i)
            for key in p.keys():
                property = p[key]
                if key == "energy_list" or key == "density_list" or key == "equivalent_atoms":
                    if isinstance(property, list):
                        property = json.dumps(property)
                pairs[key].append(property)

        columns, values = zip(*pairs.items())

        if "equivalent_atoms" in p.keys():
            self.insert_many("molecules", pairs, replace=True)
        else:
            self.insert_many("pairs", pairs, replace=True)
        self.commit()

    def add_crystal_structures(self, structures, **kwargs):
        self.insert_many("crystal", structures, **kwargs)
        self.add_metadata("added {} crystals".format(len(structures)))

    def add_metadata(self, description):
        self.insert_one(
            "meta", {"version": self.schema_version, "description": description}
        )
