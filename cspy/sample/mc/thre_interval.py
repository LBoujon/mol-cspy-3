import numpy as np
import logging

LOG = logging.getLogger(__name__)

class Interval_Generic():
    """
    Generic Interval Tool for determining number of steps per energy lid in 
    threshold MC simulations
    
    Attributes
    ----------
    increase_energy : float
        The current energy threshold (in relative energy)
    iteration_tmp : int
        The MC step of the current energy lid
    initial_energy : float
        The energy of the initial structure
    """
    
    def __init__(self, para, initial_energy, increase_energy, restart):
        self._get_para(para)
        self._initiate(initial_energy, increase_energy, restart)

    def _get_para(self, para):
        """
        Parse cspy.toml interval_para settings for parameters
        """
        return

    def _initiate(self, initial_energy, increase_energy, restart):
        """
        Initialize the steps and other variables for determine the energy lid sampling
        """
        self.iteration_tmp = 0
        return

    def _reset(self):
        """
        Reset the variables tracking the progress in the energy lid
        """
        self.iteration_tmp = 0
        return

    def update_step(self, energy):
        """
        Process the current MC step and update energy lid progress.
        
        Parameters
        ----------
        energy : float
            The single point energy of the MC perturbed structure
        
        Returns
        -------
        valid : bool
            Whether the MC step is accepted or rejected based on the current energy lid
        """
        valid = False
        return valid

    def update_lid_energy(self, increase):
        self.increase_energy += increase
        return self.data

    @property
    def data(self):
        return [self.lid_energy]

    @property
    def lid_energy(self):
        return self.initial_energy + self.increase_energy

    @property
    def restart(self):
        _restart = {
            'increase_energy':self.increase_energy,
            'iteration_tmp':self.iteration_tmp,
        }
        return _restart

class Interval_Generic_Func(Interval_Generic):
    """
    Abstract class for implementing functions that change the number of steps
    per lid according to some function.
    """
    def __init__(self, para, initial_energy, increase_energy=None, restart=None):
        if increase_energy is None and restart is None:
            raise Exception("No initial increase_energy")
        super().__init__(para, initial_energy, increase_energy, restart)

    def _get_para(self, para):
        self.initial_interval = int(para[1])

    def _initiate(self, initial_energy, increase_energy=None, restart=None):
        self.initial_energy = initial_energy
        if restart is None:
            self.increase_energy = float(increase_energy)
            self.iteration_tmp = 0
            self.lid_num = 1
        else:
            self.increase_energy = float(restart['increase_energy'])
            self.iteration_tmp = int(restart['iteration_tmp'])
            self.lid_num = int(restart['iteration_num'])

    def _interval_steps_function(self):
        pass

    def update_step(self, energy):
        self.iteration_tmp += 1
        valid = False
        if energy < self.lid_energy:
            valid = True

        lift_lid = False
        if self.iteration_tmp >= self._interval_steps_function():
            LOG.info("lid {}: raising lid after {} steps".format(self.lid_num, self.iteration_tmp))
            self._reset()
            self.lid_num += 1
            lift_lid = True
        return valid, lift_lid

    @property
    def data(self):
        return [self.lid_energy, self.lid_num,]

    @property
    def restart(self):
        _restart = {
            'increase_energy':self.increase_energy,
            'iteration_tmp':self.iteration_tmp,
            'lid_num': self.lid_num,
        }
        return _restart
 
class Interval_Fixed(Interval_Generic):
    """
    Samples each energy lid with a fixed number of MC steps
    """
    def __init__(self, para, initial_energy, increase_energy=None, restart=None):
        if increase_energy is None and restart is None:
            raise Exception("No initial increase_energy")
        super().__init__(para, initial_energy, increase_energy, restart)

    def _get_para(self, para):
        self.interval = int(para[1])

    def _initiate(self, initial_energy, increase_energy=None, restart=None):
        self.initial_energy = initial_energy
        if restart is None:
            self.increase_energy = float(increase_energy)
            self.iteration_tmp = 0
        else:
            self.increase_energy = float(restart['increase_energy'])
            self.iteration_tmp = int(restart['iteration_tmp'])

    def update_step(self, energy):
        self.iteration_tmp += 1
        valid = False
        if energy < self.lid_energy:
            valid = True

        lift_lid = False
        if self.iteration_tmp >= self.interval:
            self._reset()
            lift_lid = True
        return valid, lift_lid

class Interval_Linear_Fixed(Interval_Generic_Func):
    """
    linearly increasing number of steps per lid

    Note:
        total number of steps = initial_steps * (num_lids*(num_lids + 1))/2
    """
    def _interval_steps_function(self):
        x = self.lid_num
        return self.initial_interval * x

class Interval_Exponential_Fixed(Interval_Generic_Func):
    """
    exponentially (2**x) increasing number of steps per lid

    Note:
        total number of steps = initial_steps * (base**(num_lids + 1) - 1)/base - 1
    """
    def _get_para(self, para):
        super()._get_para(para)
        self.exp_base = float(para[2])
        if self.exp_base is None:
            raise Exception("No base given for exponential")

    def _interval_steps_function(self):
        x = self.lid_num - 1
        return int(self.initial_interval * (self.exp_base ** x))
        
class Interval_Exp_Norm_Fixed(Interval_Generic_Func):
    """
    exponentially increasing number of steps per kJ/mol of the lid height
    A decent value for the base is 1.05

    Note:
        total number of steps = initial_steps * (base**(num_lids + 1) - 1)/base - 1
    """
    def _get_para(self, para):
        super()._get_para(para)
        self.exp_base = float(para[2])
        if self.exp_base is None:
            raise Exception("No base given for exponential")

    def _initiate(self, initial_energy, increase_energy=None, restart=None):
        self.initial_energy = initial_energy
        if restart is None:
            self.increase_energy = float(increase_energy)
            self.initial_lid_energy = float(increase_energy)
            self.iteration_tmp = 0
            self.lid_num = 1
        else:
            self.increase_energy = float(restart['increase_energy'])
            self.iteration_tmp = int(restart['iteration_tmp'])
            self.lid_num = int(restart['iteration_num'])

    def _interval_steps_function(self):
        x = min(self.increase_energy, 50.0)
        return int(self.initial_interval * (self.exp_base ** (x - self.initial_lid_energy)))

class Interval_Empirical_Exp_Fixed(Interval_Generic_Func):
    def _interval_steps_function(self):
        x = min(self.increase_energy, 50.0)
        #return int(self.initial_interval + (0.25015 * np.exp(0.14472 * x))) # fitting to only good data from one structure
        return int(self.initial_interval + (0.33139 * np.exp(0.17330 * x))) # fitting to all data from more structures


class Interval_Max_Ratio(Interval_Generic):
    """
    Samples each energy lid with an adaptive number of MC steps based on monitoring the 
    ratio of structures within and without an energy bin below the current energy 
    threshold (the 'max_ratio'). 
    
    NOTE: this was found to not be effective for monitoring convergence since under 
    this stratergy high energy lids could converge faster than lower energy lids. The 
    reason for this is that threshold MC simulations spend most of the simulation at 
    the energy lid (due to the number of states increasing exponentially with energy). 
    See Shiyue Yang's PhD thesis (2022) for more details.
    """
    def __init__(self, para, initial_energy, increase_energy=None, restart=None):
        if increase_energy is None and restart is None:
            raise Exception("No initial increase_energy")
        super().__init__(para, initial_energy, increase_energy, restart)

    def _get_para(self, para):
        if para[1] == "fix":
            self._calculate_energy_bin = self._fix_energy_bin
        elif para[1] == "linear":
            self._calculate_energy_bin = self._linear_energy_bin
        elif para[1] == "exp":
            self._calculate_energy_bin = self._exp_energy_bin
        elif para[1] == "sqrt":
            self._calculate_energy_bin = self._sqrt_energy_bin
        else:
            raise Exception("No energy bin calculation method", para[1])

        self.step_min = int(para[2])
        self.step_max = int(para[3])
        self.step_interval = int(para[4])
        self.bin_para = float(para[5])
        self.tolerance = float(para[6])

    def _initiate(self, initial_energy, increase_energy=None, restart=None):
        self.initial_energy = initial_energy
        if restart is None:
            self.increase_energy = float(increase_energy)
            self.iteration_tmp = 0
            self.num_max = 0
            self.max_ratio = []
        else:
            self.increase_energy = float(restart['increase_energy'])
            self.iteration_tmp = int(restart['iteration_tmp'])
            self.num_max = int(restart['num_max'])
            self.max_ratio = []
            for i in restart['max_ratio']:
                self.max_ratio.append(float(i))
        self._get_energy_bin()
        self._data = [
            self.lid_energy, self.energy_bin, self.iteration_tmp, self.num_max, 0.0
        ]

    def update_step(self, energy):
        self.iteration_tmp += 1
        valid = False
        if energy < self.lid_energy:
            valid = True
            if (self.lid_energy - energy) < self.energy_bin:
                self.num_max += 1

        self.max_ratio.append(float(self.num_max) / float(self.iteration_tmp))

        lift_lid = False
        if self.iteration_tmp >= self.step_min:
            variance = np.var(self.max_ratio[-self.step_interval:])
            if self.iteration_tmp >= self.step_max or variance < self.tolerance:
                self._data = [
                    self.lid_energy, 
                    self.energy_bin,
                    self.iteration_tmp, 
                    self.num_max, 
                    variance,
                ]
                self._reset()
                lift_lid = True
        return valid, lift_lid

    def _reset(self):
        self.iteration_tmp = 0
        self.num_max = 0
        self.max_ratio = []

    def _get_energy_bin(self):
        self._calculate_energy_bin()
        LOG.info("Increase_energy %s, energy_bin %s", self.increase_energy, self.energy_bin)
        
    def _fix_energy_bin(self):
        self.energy_bin = 1.0

    def _linear_energy_bin(self):
        self.energy_bin = self.increase_energy / self.bin_para / self.bin_para

    def _sqrt_energy_bin(self):
        self.energy_bin = np.sqrt(self.increase_energy) / self.bin_para

    def _exp_energy_bin(self):
        self.energy_bin = np.exp(self.increase_energy / self.bin_para / self.bin_para - 1)

    def update_lid_energy(self, increase):
        self.increase_energy += increase
        self._get_energy_bin()
        self._data[0] = self.lid_energy
        self._data[1] = self.energy_bin
        return self.data

    @property
    def data(self):
        return self._data

    @property
    def restart(self):
        _restart = {
            'increase_energy':self.increase_energy,
            'iteration_tmp':self.iteration_tmp,
            'num_max':self.num_max,
            'max_ratio':self.max_ratio[-self.step_interval:],
        }
        return _restart
    
class Interval_On_The_Fly_Clustering(Interval_Generic):
    """
    Determine intervals by convergence of number of minima found 
    by on-the-fly clustering.
    
    parameter list:
        [ 
            "cluster_s", 
            "<min steps per lid>", 
            "<max steps per lid>", 
            "<steps between check converged>",
            "<num converged checks to increase lid (overall not consecutive)>",
        ]
    """
    def __init__(self, para, unique_structures, initial_energy, increase_energy=None, restart=None):
        if increase_energy is None and restart is None:
            raise Exception("No initial increase_energy")
        if len(para) != 5:
            raise ValueError("missing parameters for interval too: {}".format(self.__doc__))
        self.unique_structures = unique_structures
        super().__init__(para, initial_energy, increase_energy, restart)
    
    def _get_para(self, para):
        self.min_steps = int(para[1])
        self.max_steps = int(para[2])
        self.check_interval = int(para[3])
        self.intervals_for_convergence = int(para[4])
        LOG.info('steps per lid by on-the-fly clustering')
        LOG.info('min steps = {}; max steps per lid = {}; check interval = {}; intervals for convergence {}'.format(
            self.min_steps,
            self.max_steps,
            self.check_interval,
            self.intervals_for_convergence,
        ))

    def _initiate(self, initial_energy, increase_energy=None, restart=None):
        self.initial_energy = initial_energy
        if restart is None: 
            self.increase_energy = float(increase_energy)
            self.iteration_tmp = 0
            self.converged_intervals = 0
            self.minima_total = 1
            self.unique_history = [ # something like this would needed to restart correctly
                (self.iteration_tmp, self.minima_total)
            ]
        else:
            self.increase_energy = float(restart['increase_energy'])
            self.iteration_tmp = int(restart['iteration_tmp'])
            self.minima_total = int(restart["num_minima"])
            self.converged_intervals = int(restart["current_interval"])

    def update_step(self, energy):
        self.iteration_tmp += 1
        valid = False

        if energy < self.lid_energy:
            valid = True

        lift_lid = False
        converged = False

        if self.iteration_tmp % self.check_interval == 0:
            current_minima_total = self.unique_structures.number_structures
            prev_minima_total = self.minima_total
            if prev_minima_total == current_minima_total:
                self.converged_intervals += 1
                converged = True if self.converged_intervals >= self.intervals_for_convergence else False

        if (
            self.iteration_tmp >= self.max_steps or 
            (self.iteration_tmp >= self.min_steps and converged)
        ):
            lift_lid = True
            self._reset()
    
        return valid, lift_lid
    
    def _reset(self):
        self.iteration_tmp = 0
        self.converged_intervals = 0

    @property
    def data(self):
        return [self.lid_energy]

    @property
    def restart(self):
        _restart = {
            'increase_energy':self.increase_energy,
            'iteration_tmp':self.iteration_tmp,
            'current_interval': self.converged_intervals,
            'num_minima': self.minima_total,
        }
        return _restart
    



