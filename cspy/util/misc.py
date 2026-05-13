from typing import List
import math

def group_close_numbers(numbers: List[float], threshold: float = 10) -> List[List[float]]:
    """
    Group (cluster) numbers that are within a certain threshold of each other.

    Parameters:
        numbers (list): A list of numbers to group.
        threshold (float): The maximum difference between numbers to be grouped together.
            Default is 10.

    Returns:
        list: A list of lists, where each sublist contains numbers that are close to each other.
    """
    if not numbers:
        return []

    numbers = sorted(numbers)
    groups = []
    current_group = [numbers[0]]

    for num in numbers[1:]:
        if num - current_group[-1] < threshold:
            current_group.append(num)
        else:
            groups.append(current_group)
            current_group = [num]

    groups.append(current_group)
    return groups


def mp_check_reservation(ind : int, Q : int, num_workers : int) -> bool:
    """
    This function facilitates distributing an iterable of tasks over a set of workers
    via multiprocessing. If the iterable cannot be split apriori (such as an intertools
    object) this function can help. The number of workers must be known (num_workers)
    and each worker is assigned a unique ID (Q) from 0 to [num_workers]. Have each worker
    iterate over the entire iterable and keep track of the the index through the iterable.
    The function herein reserves each index to a unique Q value to ensure that no two
    workers do the same task.

    Parameters:
        ind (int): A numerical index associated with a task (start count from 0)
        Q (int): A unique numerical ID for the worker
        num_workers (int): The number of workers assigned to this task

    Returns:
        bool: True if the index is reserved for this worker. False if not.

    Example:
        for ind, task in enumerate(all_tasks):
            if check_reservation(ind_ind, Q, num_workers):
                task.do()
            else:
                pass
    """
    if ind == 0:
        ind_decimal = 0
        if Q == 0:
            Q_decimal = 0
        else:
            Q_decimal = 1
    else:
        frac = ind / num_workers
        ind_decimal = frac - math.floor(frac)
        Q_decimal = Q / num_workers

    return round(ind_decimal, 5) == round(Q_decimal, 5)