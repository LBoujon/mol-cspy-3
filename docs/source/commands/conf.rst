.. _conf-app:

cspy-conf command
================

The ``cspy-conf`` command is used for searching for and clustering conformers of a given molecule.
Flexible molecules can adopt more than one stable conformation which we call conformers. These conformers are gas-phase minima in which there are no intermolecular interactions between the molecule and any other molecule. 
During CSP, we pack molecules as rigid-bodies into crystals to generate trial structures. Because of this rigid-body approach, for flexible moleucle CSP, it is important to have sufficiently sampled the intramolecular energy surface.
This is so that the set of conformations which we may pack into our crystals is diverse and sufficiently covers conformational space.
Therefore the first step in a flexible CSP is often to search for stable conformers using the ``cspy-conf`` command, often followed by use of the ``cspy-moldis`` command to generate conformations that sample the conformational space around those minima.

.. note::  
    Currently. the ``cspy-conf`` command is only applicable to neutral, singlet-state molecules.

.. sphinx_argparse_cli::
  :module: cspy.apps.conf
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-conf

.. note:: 
   ``cspy-conf`` requires ``xtb``, ``openbabel`` and ``CREST`` to be installed.

Generating conformations
--------

To generate conformers, you can call the ``--SEARCH`` flag and specify the conformer generation method. Initial clustering of the resulting conformers will also be performed automatically during this process.

.. code:: bash

    cspy-conf -xyz molecule.xyz --search CREST 

The above will perform a single CREST search using ``molecule.xyz`` as an initial starting position. Generated conformers can be found in the conformers folder, which will be created and populated automatically. The process will also generate a text file called ``uniques.txt``, which lists conformers within the conformers folder which are conformationally unique based on geometric overlay clustering (RMSD). The CREST program limits the final conformers included based upon an energy window relative to the global minimum conformer energy. Within the mol-cspy interface, this criterion is set to 9 kcal/mol (37.7 kJ/mol).
If you wish to cluster conformers based on torsions, this should be performed seperately after the search using the ``--cluster`` flag as explained below.

.. note::  
    The CREST search method does not parallelize across nodes. However, parallelization accross cores within a node is possible. The number of cores should be specified using the ``-np`` flag.

It has been previously shown that performing multiple CREST searches at differing starting postitions can yield a more complete conformational search. 
This can be achieved using the ``mCREST`` option as the search argument. This search option will genertae multiple starting positions, and run a CREST serach beginning from each of these. The number of starting positions can be specified using the ``-k`` flag. Alternatively, if not specified, the ideal number of starting positions will be determined automatically using a combination of kmeans clustering and principle component analysis.
It is also possible to perform a dryrun before performing a conformer search to identify correctly assigned clusters, by supplying the ``-dry`` flag.

.. code:: bash

    cspy-conf -xyz molecule.xyz --search mCREST -dry

Conformational clustering
--------

The ``cspy-conf`` command can also be used to cluster an existing set of conformers, whether they are generated via ``cspy-conf`` or generated elsewhere. To do this, use the ``--cluster`` flag. You should specify the clustering method you wish to use, i.e TORSIONS or RMSD. The molecule_torsions file is not required when clustering using RMSD.

.. code:: bash

    cspy-conf --cluster TORSIONS -e energies.txt -t molecule_torsions

The -e flag indicates a file name which holds the name of each conformer and its corresponding energy in eV. This is provided in order to determine which conformers, based upon an energy window ``-ewin`` , should be compared during clustering.
 
.. code:: code
    molecule_1.xyz -10.0
    molecule_2.xyz -10.1
    molecule_3.xyz -14.0
    molecule_4.xyz -16.0
    ...

The ``-t`` flag, which must be specified if choosing TORSIONS as the clustering method, indicates a torsion file. Each line of the file represents a torsion wherein each element is the atom number indexed from 1 and the final element is the rotational symmetry around that torsion.

.. code:: code
    
    1 2 3 4 1 #torsion made from atoms 1-4 with rotational symmetry 1.
    2 3 4 5 2 #torsion made from atoms 2-5 with rotational symmetry 2.

The TORSIONS clustering method will identify unique conformers based upon their similarity in the torsion angles specified in the torsions file. You can use the ``-smax`` and ``-rmax`` flags to specify the maximum rms angle between selected torsions.
maximum single torsion angle between selected torsions, for two conformers to be considerd duplicates. Deviation in torsion angles beyond these limits for a given conformer pair will lead to the pair being considered distinct from each other. 
All conformers in the set to be clustered using TORSIONS should share the same atom indexing. If you are unsure if this is the case, the indexing can be handled within ``cspy-conf`` by specifying the ``-o`` flag when running the clustering.
If clustering using RMSD, the tolerance for identifying conformers as duplicates can be altered by specifying the ``-rmsd`` flag.

.. note::  
    The torsions file should not contain comments at this stage.
Inside the generated ``conformers`` directory, you will find a list of unique conformers in ``uniques.txt``, a concatenated set of unique conformer geometries in ``uniques.xyz``

Optimisation
--------
It is reccommended that following conformer generation and clustering, the resulting conformers files are reoptimised at a higher level of theory than that used by CREST - such as DFT. Gaussian input files and optimisation job scripts can be prepared automatically for this purpose using the ``cspy-setup`` command.