import numpy as np
from copy import deepcopy


class Min_Trial():
    """
    Recording energy stage for the connectivity graph; the min_energy is either
    set manually or the minimum energy of initial structures
    """
    def __init__(self, min_energy, increase=1):
        self.min_energy = min_energy
        self.increase = increase 

    def get_energy_from_state(self, state):
        return self.min_energy + self.increase * state

    def get_state_from_energy(self, energy):
        return int(np.ceil(float(energy - self.min_energy) / float(self.increase)))

    def get_ticks(self, state_max):
        ticks = []
        for i in range(state_max):
            ticks.append(self.min_energy + self.increase * i)
        return ticks

class Trial():
    """The trial information from table trial in databases"""
    def __init__(self, trajectory):
        self.traj = {}
        for items in trajectory:
            self.traj[items[0]] = items[1]

    def get_current_energy(self, step):
        for lid_step in sorted(self.traj, reverse=True):
            if step >= lid_step:
                return self.traj[lid_step]

    def get_state_from_mc_step(self, step):
        lid_steps = list(sorted(self.traj.keys(), reverse=False))
        for lid_step in sorted(lid_steps, reverse=True):
            if step >= lid_step:
                return lid_steps.index(lid_step)

class Trial_State():
    """Recoding states of trials and overlaps between trials"""
    def __init__(self, trial_numbers):
        self.trials = {}
        self.active_state = {}
        self.basin = {}
        for trial_number in trial_numbers:
            self.trials[trial_number] = trial_number
            self.basin[trial_number] = [trial_number]
            self.active_state[trial_number] = 0

    @property
    def active_trials(self):
        return list(self.active_state.keys()) # must be list. Can't pickle dict_keys/generators

    def get_overlap(self, trial_overlap):
        if len(trial_overlap) <= 1:
            return
        min_path = self.trials[min(trial_overlap, key=lambda x:self.trials[x])]
        for trial_i in sorted(trial_overlap):
            if self.trials[trial_i] <= min_path:
                min_path = self.trials[trial_i]
                continue
            self.trials[trial_i] = min_path
            self.trials[self.trials[trial_i]] = min_path
        self._cluster()
        self._update_basin()
        return

    def _update_basin(self):
        new_basin = {}
        for key, value in self.trials.items():
            if value in new_basin:
                new_basin[value].append(key)
            else:
                new_basin[value] = [key]
        self.basin = new_basin

    def _cluster(self):
        for trial_i in sorted(self.trials):
            self.trials[trial_i] = self.trials[self.trials[trial_i]]

    def update_active(self):
        new_active_state = {}
        for trial_number in self.trials.values():
            new_active_state[trial_number] = self.active_state[trial_number]
        self.active_state = new_active_state

    def update_state(self, trial_number, state):
        if trial_number not in self.active_state:
            LOG.error("Error: trial_number not active")
        self.active_state[trial_number] = state

class Range():
    """Generate x axis for connectivity graph"""
    def __init__(self, num_nodes):
        self.ranges = {0: (0, num_nodes)}
        self.old_ranges = {}
        self.new_ranges = {}
        self.overlap = {}
        self.new_overlap = {}
        pos_tmp = int(num_nodes / 2)
        self.thre_pos = {0: pos_tmp}

    def get_range(self, path, num):
        for old_path, block in self.overlap.items():
            if path in block:
                path_index = block.index(path)
                if path not in self.ranges:
                    #i = max(self.ranges.keys())
                    #i = min(self.ranges.keys(), key=lambda x: abs(x-path))
                    #start = self.ranges[i][0]
                    #start = self.ranges[path-1][0]
                    start = self.ranges[old_path][0]
                    #start = self.old_ranges[old_path][0]
                    #self.old_ranges[old_path] = (start + num, self.ranges[old_path][1])
                    self.ranges[old_path] = (start + num, self.ranges[old_path][1])
                    #if path_index == 0:
                    #    start = self.ranges[old_path][0]
                    #else:
                    #    start = self.ranges[old_path][1]
                    #if path_index == 0:
                    #    start = self.ranges[old_path][0]
                    #else:
                    #    prev_path = block[path_index - 1]
                    #    start = self.ranges[prev_path][1] + 1
                    #self.update_range(
                    #    old_path,
                    #    start + num,
                    #    self.ranges[old_path][1]
                    #)
                else:
                    start = self.ranges[path][0]
                    #self.old_ranges[path] = (start + num, self.ranges[old_path][1])

                #if path_index > 0:
                #    prev_index = block[path_index -1]
                #    print('{} start: {}'.format(path, start))
                #    try:
                #        print('prev end: {}'.format(self.ranges[prev_index][1]))
                #    except:
                #        print('no prev index for {}'.format(path))
                #        print('{}: {}'.format(old_path,block))
                #        raise ValueError

                end = start + num

                #if path_index < len(block) - 1:
                #    later_path = block[path_index + 1]
                #    if later_path not in self.ranges:
                #        self.ranges[later_path] = (end, 0)
                #    #elif covered and covered[0] not in self.ranges:
                #    #    self.ranges[covered[0]] = (end, 0)
                    
                #self.old_ranges[old_path] = (end, self.
                self.ranges[old_path] = (end, 0)
                self.new_ranges[path] = (start, end)
                #break
                return (start, end)

        return self.ranges[path]

    def get_thre_pos(self, path):
        if path not in self.thre_pos:
            num_sum = self.ranges[path][0] + self.ranges[path][1]
            self.thre_pos[path] = int(num_sum / 2)
        return self.thre_pos[path]

    def update_range(self, path, start, end):
        self.new_ranges[path] = (start, end)
        #self.ranges[path] = (start, end)

    def get_overlap(self, path, overlap):
        if path not in self.overlap or self.overlap[path] != overlap:
            self.thre_pos = {}
            self.new_overlap[path] = sorted(overlap)
                
    def update_ranges(self):
        #self.old_ranges = deepcopy(self.ranges)
        for path, interval in self.new_ranges.items():
            self.ranges[path] = interval

    def update_overlap(self):
        self.overlap = deepcopy(self.new_overlap)
        self.new_overlap = {}

