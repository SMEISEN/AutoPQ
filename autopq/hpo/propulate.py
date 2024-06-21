import numpy as np
from mpi4py import MPI
from random import Random
from typing import Callable, List, Union
from propulate import Islands, Propulator, PolliPropulator
from propulate.population import Individual
from propulate.utils import get_default_propagator

from autopq.sub_pipelines.utils import save_modules


def log_sigma(active_pop: List[Individual]) -> float:
    """
    Estimate the standard deviation of the log-normal distribution of the optimal hyperparameters in the active population.
    :param active_pop:
        Active population of individuals.
    :type active_pop: List[Individual]
    :return:
        Standard deviation of the log-normal distribution
    :rtype: float
    """

    if len(active_pop) < MPI.COMM_WORLD.size:
        return np.nan
    x = [ind.sampling_std for ind in active_pop if ind.loss is not None]
    mu = sum(np.log(x)) / len(x)
    return float(np.exp(np.sqrt(sum([(np.log(x_i) - mu) ** 2 for x_i in x]) / (len(x) - 1))))


def log_mu(active_pop: List[Individual]) -> float:
    """
    Estimate the mean of the log-normal distribution of the optimal hyperparameters in the active population.
    :param active_pop:
        Active population of individuals.
    :type active_pop: List[Individual]
    :return:
        Mean of the log-normal distribution
    :rtype: float
    """

    if len(active_pop) < MPI.COMM_WORLD.size:
        return np.nan
    x = [ind.sampling_std for ind in active_pop if ind.loss is not None]
    return float(np.exp(sum(np.log(x)) / len(x)))


def keep_best_pipeline(self: Union[Propulator, PolliPropulator], ind: Individual) -> None:
    """
    Save the modules if the assessed pipeline is better than the population.
    :param self:
        Instance of the Populate propulator.
    :type self: Union[Propulator, PolliPropulator]
    :param ind:
        Assessed individual.
    :type ind: Individual
    :return: None
    :rtype: None
    """

    # Check if loss is better than population
    if len(self.population) > 0 and ind.loss < sorted(self.population, key=lambda obj: obj.loss)[0].loss:
        # Save fitted modules
        save_modules(pipeline_modules=ind.pipeline_modules, dry=self.checkpoint_path)
    del ind.pipeline_modules


def propulate_search(
        rng: Random, pop_size: int, limits: dict, mate_prob: float, mut_prob: float, random_prob: float,
        migration_probability: float, loss_fn: Callable, generations: int, num_isles: int, checkpoint_path: str
) -> Individual:
    """
    Hyperparameter optimization using Propulate.
    :param rng:
        Random number generator.
    :type rng: Random
    :param pop_size:
        Population size.
    :type pop_size: int
    :param limits:
        Hyperparameter configuration space.
    :type limits: dict
    :param mate_prob:
        Crossover probability.
    :type mate_prob: float
    :param mut_prob:
        Point mutation probability.
    :type mut_prob: float
    :param random_prob:
        Random-initialization.
    :type random_prob: float
    :param migration_probability:
        Migration probability.
    :type migration_probability: float
    :param loss_fn:
        Function to be minimized.
    :type loss_fn: Callable
    :param generations:
        Number of generations.
    :type generations: int
    :param num_isles:
        Numer of isles.
    :type num_isles: int
    :param checkpoint_path:
        Path where checkpoints and the modules of the best-performing pipeline are stored.
    :type checkpoint_path: str
    :return:
        Best-performing individual.
    :rtype: Individual
    """

    propagator = get_default_propagator(
        pop_size=pop_size,
        limits=limits,
        mate_prob=mate_prob,
        mut_prob=mut_prob,
        random_prob=random_prob,
        rng=rng)
    islands = Islands(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        generations=generations,
        num_isles=num_isles,
        migration_probability=migration_probability,
        pollination=True,
        checkpoint_path=checkpoint_path,
        pop_stat={"sampling_std__mu": log_mu, "sampling_std__sigma": log_sigma},
        custom_logger=keep_best_pipeline
    )
    result = islands.evolve(top_n=1, logging_interval=1, DEBUG=2)
    result = [item for isle in result for item in isle]
    return min(result, key=lambda d: d.loss)
