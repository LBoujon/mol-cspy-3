class Increase_Generic():
    """
    Generic Increase Tool for determining energy thresholds in threshold MC
    
    Attributes
    ----------
    Increase_energy : float
        The amount to increase the energy threshold by for the next energy lid
    """
    def __init__(self, para, restart):
        self._get_para(para)
        self._initiate(restart)

    def _get_para(self, para):
        return

    def _initiate(self, restart):
        return

    def _reset(self):
        return

    @property
    def increase_energy(self):
        return self._increase_energy

class Increase_Fixed(Increase_Generic):
    """
    Energy threshold is increased by a fix amount with each successive energy lid
    """
    def __init__(self, para, restart=None):
        super().__init__(para, restart)

    def _get_para(self, para):
        self._increase_energy = float(para[1])
        
class Increase_Specified(Increase_Generic):
    def __init__(self, para, restart=None):
        super().__init__(para, restart)
        
    def _get_para(self, para):
        self.increases = [ float(x) for x in para[1:] ]
        
    def _reset(self):
        self.lid_num = 1
        
    def _initiate(self, restart=None):
        if restart is None:
            self.lid_num = 1
        else:
            self.lid_num = int(restart['lid_num'])
            
    @property
    def increase_energy(self):
        if self.lid_num < len(self.increases):
            energy_increase = self.increases[self.lid_num - 1]
        else:
            energy_increase = self.increases[-1]
        
        self.lid_num += 1
        
        return energy_increase
