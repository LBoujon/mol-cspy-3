import time
from datetime import timedelta
import numpy as np


def strfdelta(delta, fmt):
    d = {"d": delta.days}
    d["h"], rem = divmod(delta.seconds, 3600)
    d["m"], d["s"] = divmod(rem, 60)
    d["ms"], d["us"] = divmod(delta.microseconds, 1000)
    return fmt.format(**d)


class TimeRemainingEstimator:
    def __init__(self, total, n=10, deviation=5.0, units="structures"):
        """

        Parameters
        ----------
        n : int
            Number of samples for moving average

        deviation : float
            Number of standard deviations away to reset
        """
        self.prev = 0
        self.timestamp = time.time()
        self.rates = []
        self.number_samples = n
        self.deviation_threshold = deviation
        self.std_dev = 1.0
        self.average = 0.0
        self.units = units
        self.total = total

    def reset(self):
        self.rates = []

    def _pop(self):
        self.rates.pop(0)

    def _calc_rate(self, progress):
        timestamp = time.time()
        return (progress - self.prev) / (timestamp - self.timestamp)

    def update(self, progress):
        rate = self._calc_rate(progress)

        if len(self.rates) >= self.number_samples and self.std_dev > 1e-6:
            self._pop()
            if abs(progress - self.average) / self.std_dev > self.deviation_threshold:
                self.reset()

        self.rates.append(rate)
        self.prev = progress
        self.timestamp = time.time()
        self.std_dev = np.std(self.rates)
        self.average = np.mean(self.rates)

    def estimated_time_remaining(self):
        if self.average < 1e-6:
            return float("inf")
        return (self.total - self.prev) / self.average

    def __str__(self, fmt=""):
        if self.prev >= self.total:
            seconds = 0
        else:
            seconds = self.estimated_time_remaining()
        try:
            if seconds == 0:
                return "Done"
            elif seconds < 60:
                return "< 1 min"
            elif seconds < 3600:
                return strfdelta(timedelta(seconds=seconds), "{m} mins")
            return strfdelta(timedelta(seconds=seconds), "{h} hrs {m}m")
        except OverflowError as e:
            return "Soon\u2122"
