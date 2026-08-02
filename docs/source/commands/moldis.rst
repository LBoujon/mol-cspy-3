.. _moldis-app:

cspy-moldis command
===================

The ``cspy-moldis`` command is used to distort molecular conformers to yield a database of molecular conformations around local conformational minima.

The conformation adopted by a molecule in the solid state may not be the same as the conformational minima in a vacuum. 
The conformations which are used should be considered within the context of their relative energy.
The energy penalty associated with the distortion from vacuum minima may be compensated for by gains in lattice energy, but the higher the molecular strain, the less likely the conformation can be found in a stable crystal structure.

.. sphinx_argparse_cli::
  :module: cspy.apps.mol_dis
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-moldis

.. note:: 
   ``cspy-moldis`` requires either ``Psi4`` or  ``Gaussian`` to be installed.

Overview
--------
``cspy-moldis`` will accept molecular conformers in either ``.xyz`` or ``.fchk`` format and are provided to the command line as the first term:

.. code:: bash

    cspy-moldis conf0.xyz conf1.xyz ...

These conformers are each distorted by rotating specified groups of atoms along the specified axis. A finite basis set DFT singlepoint is then run to calculate the energy, point charges, and multipoles of the distorted conformations.
The output will be a conformational database for each conformation containg the geometry, as well as the energy, point charges, multipoles, and molecular axes. This database can then be provided as input to ``cspy-flex``.


Scanning DOFs
-------------

``mol-CSPy`` does not make judgements on conformational flexibility by itself. It falls upon the user to decide which degrees of freedom should be sampled.
The greater the number of degrees of freedom, the more conformations, and the greater the cost of the CSP. The scanned DOFs should therefore be selected carefully.

For the purposes of defining DOFs, all atoms are indexed starting from 1. This provides better compatibility with the software that interfaces with ``mol-CSPy``'s flexibile workflow.
Furthermore, it is recommended when setting up DOFs, that the ``.xyz`` molecule is loaded in Mercury and atoms are labelled according to file ordering. This will allow the user to visualise the index of each atom in the molecule.

Defining a DOF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each dof must be defined as torsion, comprised of four atoms. 
We refer to the coordinates (``c``) of the torsion by the indices of each atom.
For a torsion between atoms 4, 3, 6, and 7, a torsion may be defined as:

.. code:: bash

    "{c:'4_3_6_7'}"

Sobol Scanning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user can quasi-randomly scan distortions around that torsion by providing the ``--scan_sobol`` with an integer that defines how many distortations to generate.
For multiple DOFs, this samples the full multidimensional range described by
each DOF without evaluating every Cartesian-product grid point.

Grid Scanning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The recommended approach to scanning DOFs is a grid scan. No additional flag is required, but the parameters of the scan must be defined in the DOF.
There are four terms that should be used to define a scan:

- ``n`` (number of steps)
- ``s`` (step size)
- ``o`` (offset)
- ``i`` (initial)

``n`` is the number of distorting steps that will be applied to the conformation. This will directly determine the number of the distorted conformers.
``s`` is the size of the step applied at each distortion. The sign determines the direction of the distortion. By default, radians are assumed, but the value may be provided in degrees if followed by ``.*D``.
``o`` is the offset of the sampling. By default, distortions will be applied in a single direction away from the input conformation. An offset can provided such that the range of distortions center around the input configuration. Again, radians are asummed but degrees are accepted.
``i`` is mutually exlusive with offset and serves the same purpose, but explicitly defines the starting value of the parameter. This may not be appropriate if multiple conformations are provided.

The below DOF defines a torsion of atoms 4, 3, 6, and 6, and instructs moldis to create 7 distortions (``n``) with a step size of 15 degrees (``s``) and an offset of -45 degrees (``o``).

.. code:: bash

    "{c:'4_3_6_7',n:7,s:15.*D,o:-45.*D}"


Joining Databases
-----------------

The output of ``cspy-moldis`` is one database per input conformation. If the user wishes for a single database comprising distorted conformations, they can run ``cspy-moldis`` again, but replace the input files with a list of databases and provide the ``-jointdb`` flag.
This will yield a single database that is suitable for ``cspy-flex``.


Examples
--------
The following bash will run ``cspy-moldis`` for two conformations of a single molecule that has 3 degrees of freedom.

.. code:: bash

    dof1="{c:'4_3_6_7',n:7,s:15.*D,o:-45.*D}"
    dof2="{c:'3_6_7_9',n:7,s:15.*D,o:-45.*D}"
    dof3="{c:'6_7_9_19',n:7,s:15.*D,o:-45.*D}"

    mpirun -np 4 cspy-moldis conf0.xyz conf1.xyz --scan_dofs $dof1 $dof2 $dof3

Each DOF must be passed as a separate argument, as above; do not wrap the three
definitions in an additional ``[...]`` string. With three DOFs and ``n:7``, a
grid scan evaluates :math:`7^3=343` geometries. A smaller quasi-random pilot can
sample the same three-dimensional angular ranges with, for example:

.. code:: bash

    mpirun -np 4 cspy-moldis conf0.xyz \
        --scan_dofs "$dof1" "$dof2" "$dof3" --scan_sobol 64

The following bash will run ``cspy-moldis`` for to combined two conformer databases into one.

.. code:: bash

    mpirun -np 4 cspy-moldis conf0.db conf1.db --jointdb
