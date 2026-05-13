import signal
from types import FrameType


class TimeoutException(Exception):
    pass


def handler(signum: int, frame: FrameType | None) -> None:
    """
    Signal handler for timeout. It takes two parameters that are not used in this 
    function but are required by the signal module.
    When a signal is received, Python automatically calls the handler function 
    with two arguments:
        signum: The signal number (e.g., SIGALRM).
        frame: The current stack frame (where the signal was received).
    Even if the handler doesn't use these arguments, it must accept them to match 
    the expected signature. If they are omitted, a TypeError is raised when the 
    signal is triggered.

    Parameters
    ----------
    signum : int
        The signal number (e.g., SIGALRM).
    frame : FrameType | None
        The current stack frame.

    Raises
    ------
    TimeoutException
        Always raised to indicate the function call timed out.
    """
    raise TimeoutException("Function call timed out!")


def run_with_timeout(
    func: callable,
    args: tuple = (),
    kwargs: dict = None,
    timeout: int = 3600,
) -> any:
    """
    Run a function with a timeout. If the function does not complete within the specified time,
    a TimeoutException is raised.

    Parameters
    ----------
    func : callable
        The function to run.
    args : tuple, optional
        The positional arguments to pass to the function.
    kwargs : dict, optional
        The keyword arguments to pass to the function.
    timeout : int, optional
        The timeout duration in seconds. Default is 3600 seconds.

    Returns
    -------
    The return value of the function if it completes in time.

    Raises
    ------
    TimeoutException
        If the function call exceeds the specified timeout.

    Example
    -------
    def slow_function(a, b, delay=1):
        import time
        time.sleep(delay)
        return a + b

    try:
        print(run_with_timeout(slow_function, args=(2, 3), kwargs={'delay': 2}, timeout=5))
    except TimeoutException as e:
        print(e)
    """
    if kwargs is None:
        kwargs = {}
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result
