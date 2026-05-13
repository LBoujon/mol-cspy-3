cspy-reoptimize command
-----------------------

The ``cspy-reoptimize`` command can be used to reoptimize crystal structures in a SQLite3 database after running ``cspy-csp``

.. sphinx_argparse_cli::
  :module: cspy.distributed.reoptimization
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-reoptimize

.. note::
   The source code for the command does not exist in the cspy.apps module. It can be located in the cspy.distributed module.

The current implementation of the ``cspy-reoptimize`` command allows the user to reoptimize the crystal structures in the SQL database
using DMACRYS, PMIN and dftb+. 

In a similar fashion to the ``cspy-csp`` command, re-optimization steps can be specified by using a ``cspy.toml`` file as follows:

.. code:: toml

  [[csp_minimization_step]]
  kind = "dmacrys"
  electrostatics = "multipoles"

Similarly, if we wanted to run a dftb+ calculation:

.. code:: toml

  [[csp_minimization_step]]
  kind = "dftb"

  [dftb]
  max_steps = 2500
  max_force = 0.00058
  kpoint_spacing = 0.05
  scc_tol = 1e-5
  timeout = 3600
  single_point = false          # Bool to control whether we run a single-point energy calculation
  atomic_pos_opt = true         # Bool to control whether we run a geometry optimisation on the atomic positions
  lattice_and_atoms_opt = false # Bool to control whether we run a geometry optimisation on the atomic positions and lattice parameters

Please note that there are a number of different keywords that can be specified under the ``[dftb]`` header. The defaults for these can be found in ``cspy/configuration.py``

.. note::

  As of V1.1, The ``cspy-reoptimize`` command no longer expects the same inputs as the ``cspy-csp`` command. i.e. the user does not need to specify an axis file,
  xyz files, dmacrys charges file and a dmacrys multipoles file if they are running a dftb calculation. For more information, please see the worked
  example below.

Example Usage
^^^^^^^^^^^^^

Please consult the following link to the acetic acid worked example - :ref:`reoptimize`.