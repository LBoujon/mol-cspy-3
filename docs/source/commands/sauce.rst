cspy-sauce command
===============

The ``cspy-sauce`` provides utilities for SAUCE. SAUCE with the AUT method can be enabled for ``cspy-csp`` with the ``--clg-aut`` flag, but this app is essential for UC2AU.

.. sphinx_argparse_cli::
  :module: cspy.apps.sauce
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-sauce

SAUCE
------------------------------------------------

Sensible Asymmetric Units for Crystal Exploration (SAUCE) is an algorithm to improve the efficiency of (quasi) random structure searching (QRSS) for crystals with more than one molecule per asymmetric unit.
In practice, we see advantages in SAUCE for crystals with 3 or more molecules.

Two methods for SAUCE are officially supported:
1. Asymmetric Unit Transplant (AUT)
2. Unit Cell to Asymmetric Unit (UC2AU)

UC2AU is typically more effective than AUT but is not always applicable and requires a little more work.

AUT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AUT is appropriate for any system where G (the number of molecules per asymmetric unit) is greater than 1.
With AUT, asymmetric units are sourced from the asymmetric units of geometry optimised crystals for the same system. These crytals must be of the same Z' and should be of the same space group.

UC2AU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UC2AU is only relevant for where Z' is greater than 1. A molecular salt with an arbitrary number of components, but only 1 formula unit per asymmetric unit, cannot be approached with UC2AU.
With UC2AU, clusters of molecules are extracted from the unit cell of geometry optimised crystals for the same system. These crytals must be of Z' = 1 and should be in a space group with a number of symmetry operations equal to the value of the target Z'.

E.g. If the user wishes to perform CSP on Z'=4 Artemisinin, the user must first perform a CSP on Z'=1 Artemisinin in space group 14 (which has 4 symmetry operators).


Automated AUT
------------------------------------------------

A fully automated workflow for AUT is integrated into ``cspy-csp`` and can be enabled with the ``--clg-aut`` flag. 
When running ``cspy-csp``, the first ``aut.num_asym`` (defaults to 1000 but may be changed in the ``cspy.toml`` file) structures per space group are generated with the traditional QRSS workflow. For each of these structures, an asymmetric unit is extracted, it's energy calculated, and it's inserted into a database.
For all subsequent structures generated, instead of inserting each molecule independently into the unit cell, an asymmetric unit from the database is inserted instead.


Building a database of asymmetric units
------------------------------------------------

As an alternative to the automated workflow, the user may provide their own database of asymmetric units.
Again, SAUCE may be enabled with ``--clg-aut`` but a database must be provided, and the number of asymmetric units in the database should be specified in the ``cspy.toml`` as ``aut.num_asym``.
CSPy will only use as many asymmetric units as defined in ``aut.num_asym``. If There are fewer asymmetric units in the database than ``aut.num_asym``, more will be generated and added to the database via the automated workflow defined above.

Databases are named with the scheme:
[system_name with  molecular multiplicity]-[spacegroup number]-AU.db
For a 1:2 cocrystal of benzoic acid to caffeine in space group 14, this would look like:
benzoic_acid.caffeinex2-14-AU.db


Extract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two scripts are provided for extracting asymmetric units from a CSP output database: ``cspy-sauce extract_aut`` and ``cspy-sauce extract_uc2au``.
These extract asymmetric units with the AUT and UC2AU methods, respectively, but are identical in their use. 
They differ only in the crystal structures in the input database. See above and the publication (https://chemrxiv.org/doi/10.26434/chemrxiv-2025-l4ftw).

The input to ``cspy-sauce extract_aut`` will ideally be a clustered database that is the output from a relevant CSP.
It is essential that structures with Buckingham catastrophes are not used to source asymmetric units. Several filters are therefore in place.
The simplest is an energy filter. This is set to -9999999 kJ/mol by default, but may be changed with ``--buckinghamtol``.
A second filter is geometry-informed but requires that the user provides G ``-G``.

After extraction of the asymmetric units, their energies are calculated. Consequently, it is required that the user provides the potential name (``-p``) and a ``_rank0.dma`` (``-d``). 
These should be consistent with the potential that will be used for the subsequent CSP with SAUCE.

Depending on the size of the databases, the process may be computationally demanding. We therefore offer parallelisation, but only across 1 node. The number of parallel process is controlled via ``-np``.

Finally, we introduce controls for the structures which are sampled by the script.
1. A maxmum energy tolerance (with respect to the global minimum crystal structure) may be enforced with ``-et``.
2. A maximum number of crystal structures may be enforced with ``-nc``.
3. The sampled crystal structures may be randomised with ``-r`` (only relevant when ``-nc`` is less than the size of the database.)

Example:

.. code:: bash

    cspy-sauce extract_aut artemisinin.db -G 4 -np 4 -p fit -d artemisininx4_rank0.dma


CLG
------------------------------------------------

We also provide the option to generate crystals without optimising them. This can be achieved via ``cspy-sauce clg_aut``.
After creation of an asymmetric unit database, AUT and UC2AU do not differ in operation and therefore only one script is provided.

The input ``cspy-sauce clg_aut`` is an asymmetric unit database, followed by several flags:

1. ``-sg`` the space group number of the generated crystals
2. ``-nc`` the number of crystal structures to generate
3. ``-np`` the number of parallel processes to use

Example:

.. code:: bash

    cspy-sauce clg_aut conformers.db -sg 14 -nc 100 -np 4

