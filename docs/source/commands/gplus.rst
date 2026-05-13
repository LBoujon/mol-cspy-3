.. _gplus-app:

cspy-gplus command
================

``cspy-gplus`` is a script for generating input files (``.xyz``, ``_rank0.dma``, ``.dma``, and ``.mols``) for running CSPy with DMACRYS on systems with G \>= 2 (i.e. Z'\>= 2 or co-crystals/salts).

Under normal circumstances, ``cspy-dma`` is ran to generate input files, but this requires running a DFT calculation on each unique molecule in the system, which is a relatively slow process.
If the user desires to do CSP on several related systems which vary in stoichiometry and/or Z' but share some or all molecules, ``cspy-gplus`` is a more efficient and practical approach than running ``cspy-dma`` for each system.

``cspy-gplus`` requires the user to run cspy-dma once for each molecule as if they were preparing to do a Z' 1 CSP. The resultant, ``.xyz``, ``_rank0.dma``, ``.dma``, and ``.mols`` can then be combined to create the input files for CSP of an arbitrary combination of the molecules.

.. sphinx_argparse_cli::
  :module: cspy.apps.gplus
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-gplus


Use with a Molecules Database
--------------------

Molecules databases store data pertaining to specific conformers of molecules. 
Each entry in the database contains:

- id
- xyz_coordinates (geometric description of the molecule in the ``.xyz`` format)
- total_charge on the molecule/ion
- chargess (poitns charges in the ``_rank0.dma`` format )
- mults (distributed multipoles in the ``.dma`` format)
- axes (molecular axes in the ``.mols`` format)

To build a new database or add molecules to an existing database, provide a list of .xyz files, the ``-d`` (``--database``) flag followed by the database name, and the ``-a`` (``add``) flag.
The user must be in a directory that contains the relevant ``.xyz``, ``_rank0.dma``, ``.dma``, and ``.mols`` files.

See below for an example:

.. code:: bash

    cspy-gplus moleculeA.xyz moleculeB.xyz -d molecules.db -a

To assemble the combined input files, drop the ``-a`` flag, and for any molecules/ions that are to be included in the asymmetric unit more than once, the ``.xyz`` is listed the same number of times.
See below for an example co-crystal with two of molecule A and one of molecule B in the asymmetric unit.:

.. code:: bash

    cspy-gplus moleculeA.xyz moleculeA.xyz moleculeB.xyz -d molecules.db


Use without a Molecules Database
--------------------

The user may wish to bypass building a database.
This is possible if the user is in a directory containing the ``.xyz``, ``_rank0.dma``, ``.dma``, and ``.mols`` files for each relevant molecule.
See below for an example co-crystal with two of molecule A and one of molecule B in the asymmetric unit.:

.. code:: bash

    cspy-gplus moleculeA.xyz moleculeA.xyz moleculeB.xyz


.. note::
   Simply concatenating input files is sometimes insufficient; particularly if more than one molecule is involved. It is far safer to use the ``cspy-gplus`` script.

.. note::
   ``cspy-gplus`` will ensure that all atoms in the system have a unique neighcrys label.

.. note::
   ``cspy-gplus`` will ensure that atoms are clustered by element and that the ordering of these elements is the same between molecules. This may mean that the ``.xyz`` file returned by ``cspy-gplus`` is different to the ``.xyz`` file provided.