.. _dma-app:

cspy-dma command
================

The ``cspy-dma`` command is used to carry out distributed multipole analysis.

.. sphinx_argparse_cli::
  :module: cspy.apps.dma
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-dma

.. note:: 
   ``cspy-dma`` requires the ``g09``, ``gdma`` and ``mulfit``executables 
   to be available in your ``$PATH``.

Neutral Molecules
-----------------

As an example, the distributed multipole analysis can be done as
follows:

.. code:: bash

   cspy-dma molecule.xyz

This will output the following files:

.. code:: bash

   molecule.mols               # molecular axis definition (NEIGHCRYS/DMACRYS) format
   molecule.dma                # molecular multipoles
   molecule_rank0.dma          # molecular charges in the same format (probably from MULFIT or similar)

Charged Molecules
-----------------

For example, for calculation of the distributed multipole analysis of
ammonium (charge = 1, multiplicity = 1), charge and multiplicity
(required for Gaussian calculations) can be entered as follows:

.. code:: bash

   cspy-dma ammonium.xyz --charges 1 --multiplicities 1

Z’ > 1 structure or a co-crystal
--------------------------------

Simply call all the program with multipole geometry files in the
stoichiometric ratio desired, e.g.:

.. code:: bash

   cspy-dma acetic.xyz acetic.xyz water.xyz

Which would result in a 2:1 hydrate of acetic acid. Try playing around
with cspy-dma

Salts
-----

Call the ``cspy-dma`` with geometry files of cation and anion followed
by charges and multiplicities. For example for ammonium benzoate with
ammonium as cation (charge = 1, multiplicity = 1) and benzoate as anion
(charge = -1, multiplicity = 1), the distributed multipole analysis can
be done as follows:

.. code:: bash

   cspy-dma ammonium.xyz benzoate.xyz --charges "1 -1" --multiplicities "1 1"

.. note::
   Use the same order in entering geometries, charges, and multiplicities.
