cspy-opt command
================

The ``cspy-opt`` command can be used to perform crystal structure optimisations on one or several input crystal structures using DMACRYS.

.. sphinx_argparse_cli::
  :module: cspy.apps.opt
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-opt

.. note::
   If a set of multipole files are not provided, then the ``cspy-opt`` command will invoke the same functions that are used
   by the ``cspy-dma`` command to perform distributed multipole analysis. To do this, a ``.fchk`` file will be generated using 
   Gaussian with the basis set and level of theory read from the relevent optional arguments (--basis-set and --method)

.. note::
   The positional argument *crystal_structures* can be in either SHELX or CIF format. 
   Furthermore, several files can be listed sequentially in order to perform
   optimisations on each crystal structure. However, the ``cspy-opt`` command is not parallelised 
   over crystal structures, hence, these optimisations will happen in serial.



Crystal structure optimisation with DMACRYS
-------------------------------------------

Below is an example of the ``cspy-opt`` command:

.. code:: bash

   cspy-opt test.res --basis-set '6-311G**' --method 'PBE1PBE' -j 4 # Use four CPU cores for the gaussian calculation


If the user already possess a set of multipoles for the input crystal structure, these can be specified using the ``--multipole`` flag
which will read the file and bypass a new DMA calculation:

.. code:: bash

   cspy-opt test.res --multipole test.dma

.. note::
   Optimisation settings for ``cspy-opt`` are less flexible than those in ``cspy-csp``. Be aware that, if using ``cspy-opt`` with DMACRYS, settings written within a cspy.toml file may not be implemented as expected.


Crystal structure optimisation with dftb+
-----------------------------------------

The ``cspy-opt`` command is not limited to DMACRYS optimisations and can also be used to perform single-point energy calculations or crystal structure re-optimisations using dftb+.
To perform these calculations, the user should ensure that they have downloaded dftb+ and make sure it is located in their ``$PATH`` variable. For more info, please visit the
dftb+ github repository using the following `link. <https://github.com/dftbplus/dftbplus>`_

In order to customise the type of calculation and simulation parameters used in the dftb+ calculation, it is useful to include a ``cspy.toml`` file in the working directory such as
the one below:

.. code:: toml

  [[csp_minimization_step]]
  kind = "dftb"

  [dftb]
  max_steps = 2500
  max_force = 0.00058
  kpoint_spacing = 0.05
  scc_tol = 1e-5
  timeout = 3600
  single_point =  false         # Bool to control whether we run a single-point energy calculation
  atomic_pos_opt =  false       # Bool to control whether we run a geometry optimisation on the atomic positions
  lattice_and_atoms_opt =  true # Bool to control whether we run a geometry optimisation on the atomic positions and lattice parameters

Please note that there are a number of different keywords that can be specified under the ``[dftb]`` header. The defaults for these can be found in ``cspy/configuration.py``

The user can start the calculation with the following command:

.. code:: bash

   cspy-opt test.res --dftb-calc

This will create a ``Crystal`` object from the ``test.res`` file and then use this to create the necessary input files for the dftb+ calculation. These are run 
inside a temporary folder that is created in ``/dev/shm``. Should the user wish to keep the output files, the ``--outputs`` flag should be included which will
copy these from the temporary folder to the working directory.