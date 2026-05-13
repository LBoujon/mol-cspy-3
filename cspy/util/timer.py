import time


class Timer:
    def __init__(self):
        pass
        self.elapsed = 0.0

    def __enter__(self):
        self.t1 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.t1
