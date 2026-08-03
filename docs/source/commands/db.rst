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

Pattern validation of structural matches with critic2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the external ``critic2`` executable is available, candidates that pass the
structural RMSD threshold can subsequently be compared by their calculated
powder patterns:

.. code:: bash

   cspy-db cluster input.db -m pymatgen \
       --compack_exp_str experimental.res \
       --cluster-rms-threshold 0.3 \
       --critic2-patterns \
       --critic2-output pattern_matches.csv

The order of the calculation is fixed: the structural comparison is performed
first and only matches with an RMSD below ``--cluster-rms-threshold`` are sent
to ``critic2``. The output CSV contains the structural RMSD followed by direct
GPWDF and global variable-cell GVCPWDF scores. For GVCPWDF, the simulated
candidate is deformed towards the experimental reference. Lower values indicate
a closer match. The variable-cell score is useful when small differences in
lattice parameters shift otherwise similar diffraction peaks. No GPWDF or
GVCPWDF threshold is applied: these scores are reported for interpretation by
the user and do not alter the structural match decision.

``critic2`` is optional and is only required when ``--critic2-patterns`` is
specified. GVCPWDF requires a ``critic2`` executable compiled with NLopt
support; mol-cspy checks this before starting the comparisons. The executable
is resolved in the following order:

#. ``--critic2-executable``;
#. the ``CSPY_CRITIC2_EXECUTABLE`` environment variable;
#. ``critic2`` on the job's ``PATH``.

This makes the same input portable between a workstation and clusters such as
Aloe and Neon without storing machine-specific paths in ``cspy.toml``. Conda
activation and environment modules both update ``PATH``, so mol-cspy finds the
binary automatically. If Critic2 belongs to the mol-cspy Conda environment, a
Slurm job can prepare and verify the environment as follows:

.. code:: bash

   cd "$SLURM_SUBMIT_DIR"
   eval "$(/opt/software/conda/bin/conda shell.bash hook)"
   conda activate mol-cspy-lucas

   command -v critic2
   critic2 --version | grep -qi NLopt || {
       echo "Critic2 with NLopt is not available in this job" >&2
       exit 1
   }

   mpiexec -n "$SLURM_NTASKS" python "$(command -v cspy-csp)" ...

If the cluster provides Critic2 separately, load its site-specific module after
activating Conda and before starting mol-cspy:

.. code:: bash

   conda activate mol-cspy-lucas
   module load critic2  # replace with the module name provided by the site
   critic2 --version | grep -qi NLopt || exit 1

No executable entry is needed in ``cspy.toml``. The optional
``CSPY_CRITIC2_EXECUTABLE`` and ``--critic2-executable`` overrides remain useful
for debugging or exceptional installations, but are not required in normal
Conda or module-based jobs.

Each candidate runs in a separate temporary directory because ``critic2``
creates fixed-name auxiliary files. The output CSV is updated atomically after
each candidate and is reused on restart. Successful rows are skipped and failed
rows are retried. Use ``--no-critic2-resume``,
``--no-critic2-retry-errors`` or ``--critic2-fail-fast`` to change those
policies. Individual failures otherwise remain in the CSV with ``status=error``
and an explanatory ``error`` field, without cancelling other candidates.

Use ``--critic2-timeout`` to change the per-comparison timeout. If no structure
passes the RMSD threshold, an empty CSV (with headers) is written and Critic2 is
not invoked.

PXRD backend consistency
^^^^^^^^^^^^^^^^^^^^^^^^

When PXRD descriptors are missing, ``--pxrd-backend`` controls their source:
``platon``, ``pymatgen`` or ``auto``. The selected backend is recorded in the
descriptor metadata. ``auto`` tries Platon first and falls back to pymatgen,
with a warning if that occurs. For a production clustering run, prefer an
explicit backend so every descriptor is generated consistently. Platon and
pymatgen patterns should not be mixed without validating the clustering
thresholds on representative structures.

The pymatgen backend uses the CSPy 2 powder-profile convention: unscaled Bragg
intensities are broadened with a Lorentzian profile of FWHM 0.05 degrees on the
regular two-theta grid. This broadening preserves overlap between slightly
shifted peaks for the cosine prefilter used by ``cdtw_cos``.

For a single input database with missing descriptors, ``--jobs`` controls the
number of worker processes used to generate them. Successful patterns are
committed to the input database every 100 structures, so an interrupted command
can be restarted and will calculate only the descriptors that are still absent.


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

Exporting cluster multiplicities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Good--Turing and related completeness estimators require one row per valid
final observation together with its cluster representative. The input database
passed to ``cspy-db cluster`` retains this information; the reduced output
database contains only unique representatives and is therefore insufficient
for reconstructing multiplicities by itself.

Use ``--cluster-info`` to include the representative ID, representative energy,
representative space group and global cluster multiplicity. ``--table-only``
avoids creating an unnecessary archive of structure files:

.. code:: bash

   cspy-db dump input.db --cluster-info --table-only \
       --table-output observations.csv

The export collapses repeated ``trial_structure`` rows by crystal ID and uses
only valid structures from the final minimization step.

Estimating sampling completeness
---------------------------------

The ``completeness`` subprogram calculates Good--Turing sample coverage and the
Chao lower-bound estimator directly from a clustered input database:

.. code:: bash

   cspy-db completeness input.db -o completeness.csv

The report can also be produced as part of clustering, after the equivalence
table has been written:

.. code:: bash

   cspy-db cluster input.db -m pymatgen --completeness \
       --completeness-output completeness.csv

By default, results are reported for the complete sample and for windows of 5,
10 and 15 kJ mol\ :sup:`-1` above the global minimum. Custom windows can be
provided as a space-separated list:

.. code:: bash

   cspy-db completeness input.db --windows 2 7.5 20 \
       -o completeness.csv

Global results combine rediscoveries across space groups. Additional rows show
the coverage of observations originating in each space group; pass
``--total-only`` to omit those rows. A clustered database that still contains
all final observations must be used, rather than the reduced unique-structure
output database.

The CSV includes reproducible confidence intervals obtained from 1000
parametric bootstrap resamples by default. The bootstrap reserves the
Good--Turing estimate of unseen probability mass instead of naively resampling
singletons as if they were known rediscoveries. Change this with ``--bootstrap-samples``,
``--confidence-level`` and ``--random-seed``; use zero bootstrap samples to
disable intervals. These are uncertainty estimates for the observed sampling
process, not proof that no physically relevant structure remains unseen. In
particular, a point estimate of 100 percent should be reported together with
its interval, Chao estimate and convergence behaviour.

A trial-ordered curve suitable for plotting in a report can be written with:

.. code:: bash

   cspy-db completeness input.db -o completeness.csv \
       --convergence-output completeness-convergence.csv \
       --convergence-points 25

The curve contains a pooled ``ALL`` series and one series per space group, so
independently sampled groups retain a meaningful local trial order. As in the
summary, ``--total-only`` omits the per-space-group series.

By default every valid final structure must occur in ``equivalent_to``. This
detects incomplete or interrupted clustering tables instead of silently
treating absent assignments as new singletons. The compatibility option
``--allow-incomplete-clusters`` restores the permissive behaviour when
inspecting a known legacy database.

For several source databases (for example, one per space group), the reduced
output database is also required to compose within-database and cross-database
equivalences:

.. code:: bash

   cspy-db completeness sg*.db --global-clusters output.db \
       -o completeness.csv

Using ``--completeness`` directly with ``cspy-db cluster sg*.db`` performs this
composition automatically. The corresponding options
``--completeness-windows`` and ``--completeness-total-only`` customize the
report; the default windows remain 5, 10 and 15 kJ mol\ :sup:`-1`.
The corresponding clustering options also include
``--completeness-bootstrap-samples``,
``--completeness-confidence-level`` and
``--completeness-convergence-output``.


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
