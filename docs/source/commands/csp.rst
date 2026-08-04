.. _csp-app:

cspy-csp command
================

The ``cspy-csp`` command is used to perform crystal structure prediction calculations.

.. sphinx_argparse_cli::
  :module: cspy.apps.csp
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-csp


To use this app, there are a few prerequisites:

1. Generate 3D geometries for the molecule(s) in question, and choose an
   appropriate potential type for the interactions in question.

2. Generate multipoles and a molecular axis file in the appropriate
   formats using ``cspy-dma`` - see :ref:`dma-app`.


Workflow
--------

The ``cspy-csp`` command uses the mpi4py package to set up a parent-worker model. The parent (MPI Rank 0) acts as a controller 
process setting up a work queue with jobs that are then assigned to workers. The workers then perform these tasks and return 
their results back to the parent process. 

Tasks are seperated into two categories:

1. Structure generation
2. Structure minimization

In structure generation tasks, the worker uses the built-in structure generator to create a crystal structure and returns it
to the work queue. In structure minization tasks, the worker receive a crystal structure resulting from the structure generation tasks and attempts
to minimize the structure with respect to its lattice energy. By default, the minimization task involves 
three structure minimizations steps which use PMIN and DMACRYS. However, the user can change any of these steps by 
supplying a ``cspy.toml`` file. For more information, please refer to the :ref:`config` section. If the minimization 
is successful, the worker returns the optimised crystal structure and this is stored into a SQL database.

The status of the ``cspy-csp`` calculation can be tracked in the ``status.txt`` file which typically looks like this:

.. code:: text

               target  valid qr_fail invalid  seed running  ttime      mtime
      61       20000  20000    5386    5618  25619       0  310916.57  248231.03
      14       20000  20000    5058    5146  25147       0  210038.67  166707.11
      19       20000  20000    1577    1626  21627       0  194878.41  153034.97
      2        20000  20000   13080   13179  33180       0  185668.93  147034.07
      4        20000  20000    1350    1417  21418       0  139030.08  119672.93
      15       20000  20000   13202   13438  33439       0  311009.15  215207.89
      33       20000  20000    1214    1244  21245       0  152953.92  136771.76
      9        20000  20000     427     609  20610       0  146190.67   118860.5
      29       20000  20000    1572    1712  21713       0  177565.81  136421.29
      5        20000  20000    9259    9573  29574       0   221863.9  128954.27

* ``1st column`` is the spacegroup number
* ``target`` is the number of valid structures that was requested
* ``valid`` is the number of valid structures that have been obtained
* ``qr_fail`` is the number of trial crystal structures that failed the crystal structure generation step
* ``invalid`` is the number of trial crystal structures that passed the crystal structure generation step but failed the subsequent minimization step.
* ``seed`` is the maximum sobol seed that has been used so far
* ``running`` is a the number of active tasks in the work queue
* ``ttime`` is the total amount of time in seconds that the workers have spent on all tasks
* ``mtime`` is the total amount of tie in seconds that the workers have spent on structure minimization tasks


In addition, a file labelled ``errors.txt`` will be generated. This file is designed to give the user more insight into the trial crystal structures which failed the minimization step
and populate the ``invalid`` column of the status file. An example of this is shown below:

.. code:: text
   
              timeout max_its buck_cat mol_clash unknown total
      61         1182      14       43        81     122  1442
      14          976       2       12       425     103  1518
      19          287       9        8        31      12   347
      2           392       0        0       920     102  1414
      4           163       1        1        79      10   254
      15         1271      12       15       263     232  1793
      33          134       0        4        20       8   166
      9           106       1        0        19      29   155
      29          412       2        8        24      25   471
      5           786       2        8       355     152  1303
      summary    5709      43       99      2217     795  8863

* The 1st column is the spacegroup number
* ``summary`` in the final row gives the total for each error type
* ``timeout`` occurs when a trial structure exceeds the set timeout for the geometry optimisation
* ``max_its`` occurs when a trial structure exceeds the maximum number of geometry optimisation iterations (steps) in DMACRYS  
* ``buck_cat`` occurs when the atoms of a trial structure move too close together and become unphysically bound. This can occur when performing a geometry optimisation using a Buckingham potential which is unphysically atrractive at small interatomic distances.
* ``mol_clash`` occurs when there is an overlap between two molecules and is often associated with a buckingham catastrosphe but where the geometry optimisation step passes.
* ``unknown`` occurs when an error does not fall into any of the previous categories (this is likely to be updated in the future for cases that are identified)
* ``total`` is the total number of errors for each spacegroup

Command line usage of ``cspy-csp``
----------------------------------

Running a local ``cspy-csp`` calculation can be done as follows,

.. code:: bash

   mpiexec -np 4 cspy-csp molecule.xyz -c molecule_rank0.dma -m molecule.dma -a molecule.mols -g 33 -n 100

where ``molecule.xyz``, ``molecule_rank0.dma``, ``molecule.dma`` and ``molecule.mols`` are the XYZ geometry file, molecular 
axis definition, molecular charges and molecular multipoles, respectively. We recommend naming the files similarly 
to this example where *molecule* is replaced with the name of the molecule that you want to perform CSP one. 
This calculation will use 4 processors, 1 for the controller process and 3 workers. The controller process will assign 
structure generation and structure  optimisation jobs to the workers. In addition, the ``-g`` and ``-n`` parameters are 
used to control the space group that we want to generate structures in, and the number of valid structures, respectively. 
In this example, we will quasi-randomly generate structures in spacegroup 33, and minimize them until we have 100 *valid* structures. 
We could parallelize this further with the ``-np`` flag up to the number of cores on the local machine.

Distributed usage of ``cspy-csp``
---------------------------------

You can also run ``cspy-csp`` on a fixed node allocation. An example submission script for SLURM is as follows,

.. code:: bash

   #!/bin/bash
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=40
   #SBATCH --time=24:00:00

   cd $SLURM_SUBMIT_DIR

   module load conda/py3-latest
   source activate cspy
   mpiexec cspy-csp molecule.xyz -c molecule_rank0.dma -m molecule.dma -a molecule.mols -g fine10


.. note::
   If the following error occurs with the above submission script change ``source activate cspy`` to ``conda activate cspy``:

   *Could not find conda environment: cspy. You can list all discoverable environments with `conda info --envs`.*


In this example the ``-g`` parameter is a string: ``fine10``. This refers to a customized setting set inside the
cspy.configuration module in which the ten most frequent space groups (as observed in the Cambridge Structural Database) are sampled with 
10,000 *valid* structures in each space group. Users may define their own custom sampling settings by appending them
to the ``COMMON_SAMPLING_SETTINGS`` dictionary in ``cspy/configuration.py``.

Below is the example of the dictionary input for the ``co-crystal_fine`` sampling setting. This is made up from the 
10 most common spacegroups for co-crystals. The number of *valid* structures is typically larger for spacegroups that 
are observed more frequently such as P2\ :sub:`1` \/c (14) and C2/c (15).

.. code:: python

   "co-crystals_fine": {
        # 10 most common spacegroups for co-crystals
        "space_group": {2, 14, 15, 4, 19, 61, 1, 33, 5, 9},
        "number_structures": {
            2: 10000,
            19: 10000,
            4: 20000,
            61: 20000,
            14: 50000,
            15: 50000,
            1: 10000,
            33: 20000,
            5: 20000,
            9: 20000,
        },

Since version 1.1.1, the ``cspy-csp`` command has been updated to allow for the user to specify the space group and number of 
valid structures inside the ``cspy.toml`` file. This is done by adding the following to the ``cspy.toml`` file:

.. code:: toml

   [sampling_settings]
   33 = 100
   14 = 500
   15 = 1000

Where the key is the space group number and the value is the number of valid structures that the user wants to generate.

Calculating PXRD during CSP
---------------------------

PXRD descriptors can be generated for each valid final structure during the
CSP by selecting a backend in ``cspy.toml``:

.. code:: toml

   [descriptors]
   pxrd = "pymatgen"

The supported values are ``"pymatgen"`` and ``"Platon"``.  Pymatgen uses the
CSPy 2-compatible unscaled Cu K-alpha pattern on the 0--20 degree grid with
0.02-degree spacing and Lorentzian broadening.  Descriptor errors are logged
but do not invalidate an otherwise successful crystal structure.  Setting
``pxrd = "none"`` disables generation.

Running ``cspy-csp`` on ARCHER2
-------------------------------

When running ``cspy-csp`` on ARCHER2, we recommend that the following environment variables are set within your job submission script

.. code:: bash

   # Prevents any threaded system libraries from automatically using threading.
   export OMP_NUM_THREADS=1

   # Prevents "STOP" message being printed from DMACRYS failures when compiled with Cray compilers
   export NO_STOP_MESSAGE=1


Employing dftb+ for optimisation steps in ``cspy-csp``
------------------------------------------------------

.. note::
   Using dftb+ for lattice optimisations can become very expensive as the number of valid structures and space groups are increased. 

Should the user wish to employ dftb+ for any minimization steps during a csp simulation, they will require a ``cspy.toml`` file in addition 
to the requisite ``.xyz``, ``.dma`` and ``.mols`` files. For example, using dftb+ for a final optimisation step after the default rigid-molecule
csp workflow would look like this:

.. code:: toml

   [[csp_minimization_step]]
   kind = "pmin"
   electrostatics = "charges"

   [[csp_minimization_step]]
   kind = "dmacrys"
   electrostatics = "charges"
   CONP = true
   PRES = "0.1 GPa"

   [[csp_minimization_step]]
   kind = "dmacrys"
   electrostatics = "multipoles"

   [[csp_minimization_step]]
   kind = "dftb"

   [dftb]
   single_point = false          # Bool to control whether we run a single-point energy calculation
   atomic_pos_opt = false        # Bool to control whether we run a geometry optimisation on the atomic positions
   lattice_and_atoms_opt = true  # Bool to control whether we run a geometry optimisation on the atomic positions and lattice parameters
   kpoint_spacing = 0.1
   timeout = 3600


Please note that there are a number of different keywords that can be specified under the ``[dftb]`` header. The defaults for these can be found in ``cspy/configuration.py``.

.. note::
   Sometimes, dftb+ may be used as a penultimate step to obtain a new geometry which is then passed to DMACRYS to perform a single-point energy calculation.
   In this case, the ``cspy-csp`` workflow will automatically re-calculate the multipoles for this new geometry and use these in the final step.
