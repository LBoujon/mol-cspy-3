cspy-db command
===============

The ``cspy-db`` command is used to process and analyse the SQLite3 database files that a produced from a crystal structure prediction simulation.

.. sphinx_argparse_cli::
  :module: cspy.apps.db
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-db

Finding redundant structures via PXRD comparison
------------------------------------------------

After a CSP calculation, you will often (almost always) find the same
structure multiple times. These redundant structures may be found by
using the ``cluster`` subprogram in ``cspy-db``:

.. code:: bash

   cspy-db cluster *.db

This will find redundant structures within all the database files,
combine the unique structures into a new database file (defaulting to
``output.db``), then find unique structures within the combined file.

Finding redundant structures with COMPACK
-----------------------------------------

.. note::
   This section is only relevant to those which have a license for the CSD Python API and followed the
   installation instructions in :ref:`csd-python-api-installation`.

In addition to PXRD clustering, we are able to perform clustering with the COMPACK
algorithm on a csp database. This uses the CSD Python API and requires
a *conda* environment which combines the CSD Python API and mol-CSPy
environments into one.

Clustering crystal structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To cluster a database using COMPACK, the following command can be used:

.. code:: bash

   cspy-db cluster input.db -m compack 

Other optional flags include:

.. code:: bash

     -cdt CLUSTER_DENSITY_THRESHOLD, --cluster-density-threshold CLUSTER_DENSITY_THRESHOLD
                           The density threshold used in clustering, within which
                           structures are considered the same.

     -cet CLUSTER_ENERGY_THRESHOLD, --cluster-energy-threshold CLUSTER_ENERGY_THRESHOLD
                           The energy threshold used in clustering, within which
                           structures are considered the same.

     -j JOBS, --jobs JOBS  Number of parallel processes/threads to use for
                           xrd/compack clustering

     -rms CLUSTER_RMS_THRESHOLD, --cluster-rms-threshold CLUSTER_RMS_THRESHOLD
                           RMS difference threshold used in compack clustering.

There are a number of COMPACK search settings that can be tweaked. The
defaults of these are recorded in the ``configuration.py`` file.
Alternatively, user defined values can be read from a ``cspy.toml``
file. See below for an example:

.. code:: toml

   [compack]
   angle_tolerance = 30
   distance_tolerance = 0.3
   packing_shell_size = 60
   ignore_hydrogen_counts = true
   ignore_hydrogen_positions = true

Searching the database for a match
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To search through a database or series of databases and compare to a
given structure, the following command should be employed:

.. code:: bash

   cspy-db cluster input.db -m compack --compack_exp_str NAME_OF_STRUCTURE

Where ``--compack_exp_str`` ``NAME_OF_STRUCTURE`` is  the filename containing the comparison crystal structure (typically an experimental SCXRD structure). 
Alternatively, the user may specify the CSD reference code (**The user should be aware that some CSD structures may contain disordered atoms or solvent molecules that
will affect the overlay comparison**).


Extracting crystal structures from a database
------------------------------------------------

If you’d prefer to work with a csv file, you can dump out the data about unique structures by using the
``dump`` subprogram in ``cspy-db``:

.. code:: bash

   cspy-db dump output.db # only unique structures
   cspy-db dump output.db --include-duplicates # all structures in the database

This will result in a data table being written to ``structures.csv``, and an archive of SHELX res files being written to ``structures.zip``.
By default this will only export unique structures.

If you want to dump structures within a specificed energy range from the global minimium structure, this can be done with the ``-e`` flag:

.. code:: bash

   cspy-db dump output.db -e 7 # only unique structures 7 kJ mol^-1 from the global minimum

If you would prefer to have these structures in a database format, add the ``--copy-db`` flag.


Summarising information in a database
------------------------------------------------

mol-CSPy's CSP databases contain a lot of information and it is often more useful to summarise it.
The ``info`` subprogram will return:
- The number of structures
- The number/percentage of unique structures (if the information is available)
- Statistics for energies and density
- Details of the five lowest energy structures

The percentage of unique structures is a convenient (albeit crude) means of measuring whether enough structures were sampled during CSP.

.. code:: bash

   cspy-db info database.db


Plotting a landscape from a database
------------------------------------------------

mol-CSPy's CSP databases contain all the information necessary to plot and render a crystal landscape.
This is a useful for on-the-fly analysis of CSP outputs.
The contents of one or more databases may be plotted together as:

.. code:: bash

   cspy-db plot database0.db database1.db


Labelling by space group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, data points are coloured are labelled according to the database they were sourced from.
If you would instead prefer to colour and labell according to spacegroup, you need only append the ``--spg`` flag.
Additionally, a list of space group numbers may optionally be provided after the ``--spg`` to instruct ``cspy-db plot`` to plot only data points belonging to those space groups.

To plot all space groups:

.. code:: bash

    cspy-db plot database0.db --spg

To plot only space groups 1 and 14:

.. code:: bash

    cspy-db plot database0.db --spg 1 14

Plotting from clustered databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cspy-db plot`` will automatically search for clustering information in a database and adapt it's behaviour if this information is found.

.. note::
   The output databases from ``cspy-db cluster`` contain only unique crystal structures.
   Input databases (aftering clustering) keep track of which crystal structures are equivalent.
   This section refers to these databases

By default, only unique crystal structures will be plotted. If you wish to plot all crystal structures, append the ``--ignore-clustering`` flag.

Additionally, the ``--equivalents`` can be added to colour each unique crystal structures according to the number of equivalent structures in the database. 
This is another (crude) means of measuring whether enough structures were sampled during CSP. If the lowest energy structures have all only been found once, it may be worth running the CSP for longer.