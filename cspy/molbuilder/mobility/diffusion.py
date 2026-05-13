import random
import math
import numpy as np


def kmc_diffusion(hopping_rates, number_of_traj, number_of_steps):
    """Calculate the average diffusion over a number of kinetic
     monte carlo trajectories.

    :param hopping_rates: A dictionary containing the hopping
        rates and hopping information.
    :param number_of_traj: Number of kinetic monte carlo simulation
        to be run.
    :param number_of_steps: Number of steps (iterations) that the
        simulation should be run for.
    """
    diffusion_matrix = np.zeros((3, 3))
    for i in range(number_of_traj):
        diffusion_matrix += kmc_single_traj(hopping_rates, number_of_steps)
    return diffusion_matrix / number_of_traj


def kmc_single_traj(hopping_rates, number_of_steps):
    """Run a single kinetic monte carlo simulation for a number of
    iterations and calculate the diffusion for this trajectory.

    :param hopping_rates: A dictionary containing the hopping
        rates and hopping information.
    :param number_of_steps: Number of steps (iterations) that the
        simulation should be run for.
    """
    unique_sites = [i for i in range(len(hopping_rates.keys()))]
    hopping_rates = hopping_rates

    rate_summations = {}
    for site, rates in hopping_rates.items():
        sum_rate = 0
        sum_rates = []
        for rate in rates:
            sum_rate += rate[1]
            sum_rates.append(sum_rate)
        rate_summations[site] = sum_rates

    time = 0
    current_x = 0
    current_y = 0
    current_z = 0
    site = random.choice(unique_sites)

    for i in range(number_of_steps):

        site_rates = hopping_rates[site]
        sum_rates = rate_summations[site]
        total_rate = sum_rates[-1]
        rate = total_rate * random.random()

        for j, sum_rate in enumerate(sum_rates):
            if rate <= sum_rate:
                site, rate, x, y, z = site_rates[j]
                current_x += x
                current_y += y
                current_z += z
                break

        time -= (math.log(random.random()) / total_rate)

    diffusion_matrix = np.array([
        [current_x**2, current_x*current_y, current_x*current_z],
        [current_y*current_x, current_y**2, current_y*current_z],
        [current_z*current_x, current_z*current_y, current_z**2]
    ]) / (2 * time)

    return diffusion_matrix


def kubo_diffusion(hopping_rates):
    """Calculate the diffusion matrix using an expression derived
    from the kubo formula for the spin-boson model with the
    U(t) = 1 approximation.

    :param hopping_rates: A dictionary containing the hopping
        rates and hopping information.
    """
    unique_sites = [i for i in range(len(hopping_rates.keys()))]
    hopping_rates = hopping_rates

    diffusion_matrix = np.zeros((3, 3))
    for site in unique_sites:
        for other_site, rate, x, y, z in hopping_rates[site]:
            diffusion_matrix += np.array([
                [x**2, x*y, x*z],
                [y*x, y**2, y*z],
                [z*x, z*y, z**2]
            ]) * (rate / 2)
    return diffusion_matrix / len(unique_sites)
