.. _templating-app:

cspy-templating command
================

The ``cspy-templating`` command is used to create a set of unoptimized analogues, albeit being comprised of a new molecule, of a given set of molecular crystal structures. The primary use of this command is in the generation of trial structures in templating CSP

.. sphinx_argparse_cli::
  :module: cspy.apps.templating
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-templating


.. note::

     Currently the cspy-templating command and corresponding templating CSP approach is only applicable to Z'=1, G=1 systems. That is, all template crystal structures must be single component and the structure files used must provide Z'=1 representations of the crystal structures. The ‘new molecule' for analogue formation must also be a single molecule. We hope that future versions of mol-cspy will expand upon this functionality

.. note::

     Use of this command requires access to the ccdc python module and, if beginning from xyz files, openbabel software.


Example Command Line Usage
--------

The analogue creation can be run using a single line of code:

.. code:: bash

    cspy-templating --template_structures  template_structures.zip --targ_mol  target_molecule_file --temp_mol template_molecule_file --end_db_name database_name

.. note::

     Currently, this command runs in serial and so if being run on worker nodes only a single core is needed.

This will produce a database named <database_name.db> containing unoptimized analogues of each template crystal structure – with those analogues being comprised of the target molecule. Each of these will be recorded as having an energy and density of 0. The value 0 here is serving as a stand in for n/a or unknown, within a database structure and code that requires these values to be numerical.


Continuing Templating CSP
--------

To complete the templating CSP process, the resulting database should be clustered using the ``cspy-db cluster`` command and then all unique structures should be lattice energy minimised using the ``cspy-reoptimize`` command.