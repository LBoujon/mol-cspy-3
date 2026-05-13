Installation
============

.. admonition:: Quick installation
    :class: tip

    If you already have a python 3.11 installation and want to get mol-CSPy with a quick command in the CLI:

    .. code:: bash

        pip install git+ssh://git@gitlab.com/mol-cspy/mol-cspy.git

    This will install mol-CSPy, but you will not be able to edit the source code.


To access the latest version of mol-CSPy, simply clone the repository with the following command:

.. code:: bash
  
  git clone https://gitlab.com/mol-cspy/mol-cspy.git


Setting up your python environment
--------------------------------------

Probably the most straightforward way to do this is to use conda (either via `anaconda <https://docs.anaconda.com/anaconda/install/>`_ or as many prefer `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. note::
  
  *conda* is often available on HPC systems, so your first port of call should be to check the modules there. *conda* will take care of installing/setting up python, so you do not need to load any python modules in your HPC profile before running conda.
    
.. note::

  It is common for available system versions of *conda* to automatically enter a *base* environment upon loading them.  This *base* env can cause problems when creating additional environments within it.  Best practice is to avoid the *base* environment entirely -- i.e. never create a new environment from within *base* (deactivate it first), and do not automatically activate *base* on login.

Once you have a working *conda* command, you can manually install the dependencies for mol-CSPy (see the ``project.dependencies`` list in ``pyproject.toml``) or let pip install them for you:

.. code:: bash

  conda create -c conda-forge -n cspy python=3.11  # This will create a python environment with the correct python version
  conda activate cspy                              # This will activate your cspy environment
  pip install -e ${CSPY_SRC}                       # This will install the package and its dependencies and allow you
                                                   # to edit the source in place, assuming ${CSPY_SRC} is the location
                                                   # of the mol-CSPy source directory

Optional dependencies may be installed installed by appending a list of optional dependency groups to the end of previous command. See [project.optional-dependencies] in pyproject.toml for available choices.
To enable pytest and mace, for example:

.. code:: bash
    pip install -e ${CSPY_SRC}[test,nn]

.. note::

    If you just want to install the mol-CSPy app but do not plan to make any changes to the source code, look at the 'Quick installation' tip at the top of the page.


If you are having issues with the *conda* environment, note that for *conda* versions older than 4.4, the command
should be: 

.. code:: bash

  source activate cspy # unix systems
  activate cspy # windows


Alternately, you may need to run ``conda init``, for the first time.

After running the ``pip install -e ${CSPY_SRC}`` command, you will notice that your python environment now has a series of commands that can be used on the command line.
These are explained in detail in the commands section. However, we will provide a brief summary of them below:

*  ``cspy-csp`` - A command to perform crystal structure prediction tasks
*  ``cspy-clg`` -  A command to perform crystal structure generation tasks
*  ``cspy-db`` - A command to process SQL databases that are output from ``cspy-csp`` commands
*  ``cspy-dma`` - A command to perform distributed multipole analysis
*  ``cspy-gplus`` - A command for creation of DMACRYS input files for multicomponent crystals
*  ``cspy-opt`` - A command to perform crystal structure minimization tasks
*  ``cspy-qrbh`` - A command to perform quasi-random basin hopping 
*  ``cspy-reoptimize`` - A command to perform crystal structure minimization tasks on SQL databases that are output from ``cspy-csp`` commands
*  ``cspy-templating`` - A command to produce unoptimised analogues of a given set of template crystal structures, albeit containing a new given molecule
*  ``cspy-threshold`` - A command to perform threshold monte carlo simulations
*  ``cspy-threshold-utils`` - A command to interact with threshold databases
*  ``cspy-sauce`` - A command to provide utilities for the SAUCE algorithm - this command is important for running the Unit Cell to Asymmetric Unit method
*  ``cspy-setup`` - A command to help set up multiple calculations
 
Using mol-CSPy on ARCHER2
------------------------------------------------

Note that, as always on ARCHER2, any software you wish to run on a compute node must be within the ``/work`` filesystem, as ``/home`` is not visible to compute nodes.

To install mol-CSPy follow the instructions from above. Installation without conda environments can cause issues, and so the (conda-free) virtual environemts approach often reccomended when installing software on ARCHER2 is not appropriate for mol-CSPy. 

Note, however, that *conda* is *not* available as a module on ARCHER2 and so you will need to install it yourself.


Required external program dependencies
--------------------------------------

In order to perform crystal minimisations, generate and fit distributed multipoles etc., mol-CSPy makes use of the following programs:

- Gaussian
- `GDMA <https://gitlab.com/anthonyjs/gdma>`_ 
- `MULFIT <https://gitlab.com/anthonyjs/gdma>`_
- `NEIGHCRYS <https://gitlab.com/mol-cspy/neighcrys>`_
- `DMACRYS <https://gitlab.com/mol-cspy/dmacrys>`_ 
- `PMIN <https://sourceforge.net/projects/molpak>`_ (This link takes you to the MOLPAK source code which includes the PMIN code)

All of these **must** be available in your ``$PATH`` if you wish to use mol-CSPy to its full capabilities. 

.. note::
  A few users in our group have reported problems with NEIGHCRYS 2.3.0
  
  Please see the following links for the `NEIGHCRYS <https://gitlab.com/mol-cspy/neighcrys/docs>`_ and `DMACRYS <https://gitlab.com/mol-cspy/dmacrys/docs>`_ manuals.


Optional external program dependencies
--------------------------------------

The following dependencies are optional extras:

- `PLATON <http://www.platonsoft.nl/platon/>`_
- `DFTB+ <https://github.com/dftbplus/dftbplus>`_

PLATON is mainly used for symmetry detection and PXRD simulation but is not available for non-commercial users without a license. For these users, we have added similar
functions using the pymatgen package which comes with mol-CSPy upon installation. DFTB+ is a software package used for tight-binding Density Funcitnal Theory calculations
which may be of some interest to our users. 


Compiling external programs on ARCHER2
--------------------------------------

Several of our group members have reported success in using binaries compiled using the GNU (i.e. ``gfortran``) compilers.  However, these are slower according to our benchmarks above.  The default, native Cray compilers on ARCHER2 (see the `ARCHER2 documentation <https://docs.archer2.ac.uk/user-guide/dev-environment/#compiler-environments>`__ can also be used to compile DMACRYS with minimal changes (see below).

NEIGHCRYS has not yet (Dec 2022) been compiled on ARCHER2 successfully using the Cray compilers, but the GNU binary seems to perform as expected.

We also do not have a license to run Gaussian on ARCHER2, so multipoles must currently be pre-computed with an alternative program (e.g. Psi-4 interface to mol-CSPy will be added in the near future) or run separately.

.. note::

  Only compile DMACRYS with the ``-O3`` flag (other flags intended for Intel compilers e.g. ``-ipo`` will not work with the Cray compiler).  If you have used the Cray compiler, you will need to set the runtime environment variable ``$NO_STOP_MESSAGE`` to prevent the unlabeled ``STOP`` calls in DMACRYS routines from printing to your output.  (e.g. add ``export $NO_STOP_MESSAGE=1`` to your submission scripts and/or ``~/.bashrc``).

  PMIN does not compile correctly with the Cray compilers, so you must use the GNU compilers on ARCHER2 by switching your compiler environment as described in the ARCHER2 documentation linked above (this only needs to be done when compiling PMIN, and can be switched back afterwards).


Testing your installation
-------------------------

.. admonition:: Read!
   :class: warning

   If you want to run the tests, you will have to install the optional ``test`` dependencies, otherwise the pytest command will not be available.
   

Now that you have everything, you can test your installation by running the following in the top directory:

.. code:: bash

  pytest --external-binaries


this will test the python code as well as the external binaries required to run mol-CSPy. This can take up to 30 mins to complete, if everything was installed successfully then you should not see any failures.

An example ``pytest`` summary (with external binary testing skipped) is shown below:

.. code:: text

   #!pytest

   pytest
   platform linux -- Python 3.8.18, pytest-8.0.1, pluggy-1.4.0
   rootdir: /mainfs/home/jag1c20/software/cspy
   configfile: pytest.ini
   testpaths: cspy
   collected 121 items / 1 skipped                                                                                                                                                                                      

   cspy/apps/dma.py .                                                                                                                                                                                             [  0%]
   cspy/chem/element.py ....                                                                                                                                                                                      [  4%]
   cspy/chem/multipole/distributed_multipoles.py .                                                                                                                                                                [  4%]
   cspy/chem/multipole/multipole.py ....                                                                                                                                                                          [  8%]
   cspy/chem/multipole/tests/test_multipoles.py .                                                                                                                                                                 [  9%]
   cspy/chem/test/test_molecule.py ............................                                                                                                                                                   [ 32%]
   cspy/configuration.py ...                                                                                                                                                                                      [ 34%]
   cspy/crystal/space_group.py .                                                                                                                                                                                  [ 35%]
   cspy/crystal/symmetry_operation.py ....                                                                                                                                                                        [ 38%]
   cspy/crystal/tests/test_crystal.py ...                                                                                                                                                                         [ 41%]
   cspy/crystal/tests/test_neighcrys_axis.py .                                                                                                                                                                    [ 42%]
   cspy/crystal/tests/test_spacegroup.py ..                                                                                                                                                                       [ 43%]
   cspy/crystal/tests/test_symmetry_operation.py .......                                                                                                                                                          [ 49%]
   cspy/db/key.py ..                                                                                                                                                                                              [ 51%]
   cspy/db/tests/test_datastore.py .                                                                                                                                                                              [ 52%]
   cspy/executable/tests/test_compack.py ss                                                                                                                                                                       [ 53%]
   cspy/executable/tests/test_dmacrys.py ssss                                                                                                                                                                     [ 57%]
   cspy/executable/tests/test_gaussian.py s                                                                                                                                                                       [ 57%]
   cspy/executable/tests/test_gdma.py s                                                                                                                                                                           [ 58%]
   cspy/executable/tests/test_neighcrys.py ss                                                                                                                                                                     [ 60%]
   cspy/executable/tests/test_suite.py ssssssssssssssssssssssss                                                                                                                                                   [ 80%]
   cspy/formats/gaussian.py .                                                                                                                                                                                     [ 80%]
   cspy/minimize/test_minimize.py s                                                                                                                                                                               [ 81%]
   cspy/ml/distance/test_cdtw.py .                                                                                                                                                                                [ 82%]
   cspy/potentials/potential_data.py .                                                                                                                                                                            [ 83%]
   cspy/sample/mc/test_mc.py ...........                                                                                                                                                                          [ 92%]
   cspy/sample/mc/test_threshold.py ...                                                                                                                                                                           [ 95%]
   cspy/sample/test_sobol_threads.py .                                                                                                                                                                            [ 95%]
   cspy/test_configuration.py ..                                                                                                                                                                                  [ 97%]
   cspy/util/__init__.py ...                                                                                                                                                                                      [100%]

   ========================================================================================== 86 passed, 36 skipped in 52.61s ===========================================================================================
