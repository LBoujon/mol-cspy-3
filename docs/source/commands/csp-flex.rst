.. _flex-app:

cspy-flex command
================

The ``cspy-flex`` command is used to perform crystal structure prediction calculations for flexible molecules.
Molecules are treated as rigid during geometry optimisation but molecules have different conformations at the point of random structure generation.

.. sphinx_argparse_cli::
  :module: cspy.apps.csp_flex
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-flex


To use this app, there is one prerequisite:

1. Generate a conformational database with ``cspy-moldis``. See :ref:`moldis-app`.


Workflow
--------

The workflow for ``cspy-flex`` is almost identical to ``cspy-csp`` (see :ref:`csp-app`) but differs slightly in use and in operation.
``cspy-csp`` generates random crystal structures comprised of molecules with a geometry defined by an input ``.xyz`` file.
In ``cspy-flex``, instead of sourcing molecular geometries from ``.xyz`` files, a conformation is selected randomly from a conformational database. 
The conformation's corresponding ``_rank0.dma``, ``.dma`` and ``.mols`` files for DMACRYS geometry optimisations are sourced automatically from the same database.


Command line usage of ``cspy-flex``
----------------------------------

Running a local ``cspy-flex`` calculation differs from ``cspy-csp`` in that instead of providing a list of ``.xyz`` files, a list of conformational database files should be provided instead.
Additionally, there is no need to specify charge (``-c``), multipole (``-m``), or axes (``-a``) files.
One new flag is introduced: ``--conf_energy_window``. This conformational energy window enforces a maximum intramolecular energy above the lowest energy conformation. 
Conformers with relative energies greater than this value will not be included at the structure generation step. A default value is set in ``cspy.configuration.py`` but is 22 kJ/mol at the time of writing.

An example usage of ``cspy-flex`` is as below:

.. code:: bash

   mpiexec -np 4 cspy-flex conformations.db --conf_energy_window 30 -g 33 -n 100