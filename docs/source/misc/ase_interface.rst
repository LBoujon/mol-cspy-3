.. _ase:

ASE Interface
======================

mol-CSPy offers an interface to `ASE <https://ase-lib.org>`__ for the purposes of singlepoint calculations and geometry optimisations.
This is the default means of using MACE in mol-CSPy (see tutorial) but also allows for the community to perform CSP with arbitrary energy models.

Configuring ASE
-------------------------------

ASE settings are fully configurable via the standard ``cspy.toml`` file by defining a ``csp_minimization_step`` and setting the kind of ``ase``.

ASE specific settings can then be further configured within the ``ase`` dictionary of that minimisation step.
Several keywords are supported at this stage:

* ``calculator_type`` The type of calculator. Set to ``mace`` for MACE, ``ase_builtin`` for ASE's TIP3P, TIP4P, Lennard-Jones, or Morse potential. A custom calculator name can also be provided. See below for details. Defaults to ``mace``.
* ``steps`` Number of geometry optimisation steps to perform. Set to 0 for a singlepoint. Defaults to 1000.
* ``fmax`` Maximum force tolerance after geometry optimization. Default is 0.05.
* ``optimizer`` Optimizer from ase.optimize for geometry optimization. Defaults to 'LBFGS'.
* ``timeout`` If set, the maximum time allowed for the minimization in seconds. Defaults to 600 seconds.
* ``symmetry`` If 'true', keep symmetry during optimization. If 'false', reduce cell to P1. Defaults to 'true'.

E.g:

.. code:: toml

   [[csp_minimization_step]]
   kind = "ase"

   [csp_minimization_step.ase]
    calculator_type = 'mace'
    symmetry = false
    optimizer = 'BFGS'
    fmax = 0.1
    steps = 100
    timeout = 300

Model/calculator-specific settings, can also be configured here by accessing the dictionary of the calculator type.
As the settings depend on the calculator, we cannot list them all here but will provide examples.

Example with MACE:

.. code:: toml

   [[csp_minimization_step]]
   kind = "ase"

   [csp_minimization_step.ase]
    calculator_type = 'mace'

   [csp_minimization_step.ase.mace]
    model = 'mace_mp'
    dispersion = false
    device = 'cuda'
    dispersion_cutoff = 22

Example with the N2P2 custom calculator bundeled with mol-CSPy:

.. code:: toml

   [[csp_minimization_step]]
   kind = "ase"

   [csp_minimization_step.ase]
    calculator_type = 'n2p2'

   [csp_minimization_step.ase.n2p2]
    n2p2_model = "Faraday2024"


Model units
-------------------------------
mol-CSPy has standard units which are inherited from DMACRYS. Lengths are expressed in Å and energies in kJ/mol per molecular formula units.
These units may differ from the units of the model/calculator that is being used.
Our ASE interface will automatically convert outputs to our standard units but the model/calculators units must be specified in a ``model_units`` dictionary either in the .toml file, or in a custom calculator file.

The standard ``model_units`` dictionary takes the form:

.. code:: python
   
   {'energy' : 'kJ/mol',
    'length' : 'Ang',
    'normalisation' : 'molecular_formula_unit',
    'energy_corr' : None}

Four energy units are supported
* 'kJ/mol'
* 'kJ'
* 'eV'
* 'kcal'

Lengths are always assumed to be Å ('ang').

Normalisation refers to the amount of the crystal that is represented by the energy.
Again, four options are provided:

* 'molecular_formula_unit': Energy per formula units of molecules, otherwise known as Z'.
* 'molecule': Energy per molecule
* 'atom': Energy per atom
* 'cell': Energy per unit cell

The energy correction ('energy_corr') is only relevant for delta learning.
With our N2P2 model, the energy is equal to the DMACRYS lattice energy plus the value predicted by the model.
If 'energy_corr' is set to 'lattice_energy', the value will be read from the database and added automatically.

Model units can be specified in the .toml file as such:

.. code:: toml

   [csp_minimization_step.ase.model_units]
    energy = 'eV'
    length = 'Ang'
    normalisation = 'cell'
    energy_corr = "none"

But when using a custom calculator, should be specified at the top of the file as:

.. code:: python

    calculator_units = {'energy' : 'eV',
                        'length' : 'Ang',
                        'normalisation' : 'cell',
                        'energy_corr' : 'none'}

If no model units can be found, we assume the dictionary in the example above as this is most natural to ASE.

Custom Calculators
-------------------------------
To provide maximum flexibility to the userbase, we support custom ASE calculators. 
Using a custom calculator does not require editing any existing mol-CSPy but it does require that the user adds a custom-written Python 3 file to ``cspy/ase/calculators``.
mol-CSPy will automatically find any calculators in that folder, and they can be called by setting ``calculator_type`` in the cspy.toml file to the file name of the calculator (excluding the .py extension).

A custom calculator file requires several key components.
Firstly, it requires an ASE calculator class. We do not provide full guidance here on writing ASE calculators and instead suggest the `ASE readthedocs <https://ase-lib.org>`__. 
This class should look something like:

.. code:: python

    from ase.calculators.calculator import Calculator

    class custom_calculator(Calculator):

        implemented_properties = ("energy")

        def __init__(self):
            self.results = {}

        def calculate(self, atoms, properties):
            return self.get_energy(atoms)

        def get_energy(atoms):
            return energy

We also require a function called ``setup_custom_calculator``, which performs setting configurations for the calculator, and then calls the calculator itself.
In its simplest form, it looks like:

.. code:: python

    def setup_custom_calculator(**kwargs):
        calculator = custom_calculator()
        return calculator

Finally, units should be provided for the calculator as a dictionary:

.. code:: python

    calculator_units = {'energy' : 'eV',
                        'length' : 'Ang',
                        'normalisation' : 'cell',
                        'energy_corr' : 'lattice_energy'}.

mol-CSPy is bundeled with two example custom calculators:

* dev-calculator: A minimalist calculator for development purposes with no physically meaningful energy
* n2p2: A calculator for using N2P2 delta-learning models