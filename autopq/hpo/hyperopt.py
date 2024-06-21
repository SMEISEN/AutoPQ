import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from typing import Callable, Union
from matplotlib import pyplot as plt
from functools import partial
from hyperopt import hp, Trials, fmin
from hyperopt.early_stop import no_progress_loss
from hyperopt.rand import suggest as rand_suggest
from hyperopt.tpe import suggest as tpe_suggest
from hyperopt.atpe import suggest as atpe_suggest
from pywatts.core.pipeline import Pipeline

from autopq.config import HyperoptAlgo, ValMetric
from autopq.hpo.utils import assess_configuration


def plateau_loss(std: float = 0.001, top: int = 10, patience: int = 0) -> Callable:
    """
    Stop function that will stop if loss plateaus across trials, i.e., the standard deviation of the best-performing trials
    is smaller than the threshold with patience.
    :param std:
        Standard deviation threshold.
    :type std: float
    :param top:
        Number of best-performing trials to be considered.
    :type top: int
    :param patience:
        Number of iterations to be patient when the plateau condition is fulfilled.
    :type patience: int
    :return:
        Early stopping function.
    :rtype: Callable
    """
    def stop_fn(trials: Trials, best_loss: float = None, iteration_patience: int = 0, top_values: list = None) -> bool:
        """
        Early stopping function.
        :param trials:
            Storage for completed, ongoing, and scheduled evaluation points.
        :type trials: Trials
        :param best_loss:
            Best loss observe so far.
        :type best_loss: float
        :param iteration_patience:
            Number of iterations to be patient when the plateau condition is fulfilled.
        :type iteration_patience: int
        :param top_values:
            List of the best-performing scores.
        :type top_values: list
        :return:
            Weather to continue or stop the optimization.
        :rtype: bool
        """
        if top_values is None:
            top_values = []
        new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
        top_values.append(new_loss)
        if best_loss is None:
            return False, [new_loss, iteration_patience, top_values]
        top_values = sorted(top_values)[:top]

        # If the current iteration has to stop
        if len(top_values) == top and np.std(top_values) <= std:
            iteration_patience += 1
        else:
            iteration_patience = 0

        return iteration_patience >= patience, [new_loss, iteration_patience, top_values]

    return stop_fn


def hyperopt_search(estimator: Pipeline, data: pd.DataFrame, val_metric: ValMetric, algo: HyperoptAlgo, max_evals: int,
                    distribution: str, a: float, b: float, log: bool, early_stopping: Union[str, None],
                    checkpoint_path: str) -> (float, float, int, float):
    """
    Hyperparameter optimization using Hyperopt.
    :param estimator:
        Forecasting pipeline to be assessed.
    :type estimator: Pipeline
    :param data:
        Data used for assessing the estimator's performance.
    :type data: pd.DataFrame
    :param val_metric:
        Metric to assess the estimator's performance.
    :type val_metric: ValMetric
    :param algo:
        Trial scheduling algorithm.
    :type algo: HyperoptAlgo
    :param max_evals:
        Maximal number of evaluations.
    :type max_evals: int
    :param distribution:
        Prior distribution of the configuration space. Can be 'normal' or 'uniform'.
    :type distribution: str
    :param a:
        Lower bound of the sampling standard deviation.
    :type a: float
    :param b:
        Upper bound of the sampling standard deviation.
    :type b: float
    :param log:
        Weather the prior distribution is logarithmic or not.
    :type log: bool
    :param early_stopping:
        Weather to use an early stopping function ('progress' or 'plateau') or not (None)
    :type early_stopping: Union[str, None]
    :param checkpoint_path:
        Path for storing the optimization result.
    :type checkpoint_path: str
    :return:
        float(y[idx[0]]), the best score
        float(x[idx[0]]), the best sampling standard deviation
        len(idx), the number of evaluations
        end-start, computing time in seconds
    :rtype: (float, float, int, float)
    """

    # Define trial sampling algorithm
    if algo == HyperoptAlgo.rand:
        sampler = rand_suggest
    elif algo == HyperoptAlgo.TPE:
        sampler = tpe_suggest
    elif algo == HyperoptAlgo.aTPE:
        sampler = atpe_suggest
    else:
        raise NotImplementedError(f"Search algorithm {algo} is not implemented!")

    # Define configuration space
    if distribution == "normal":
        if log:
            search_space = {"sampling_std": hp.lognormal("sampling_std", np.log(a), np.log(b))}
            dist = stats.lognorm(s=np.log(b), scale=a)
        else:
            search_space = {"sampling_std": hp.normal("sampling_std", a, b)}
            dist = stats.norm(s=b, scale=a)
    elif distribution == "uniform":
        if log:
            search_space = {"sampling_std": hp.loguniform("sampling_std", np.log(a), np.log(b))}
            dist = stats.loguniform(np.log(a), np.log(b))
        else:
            search_space = {"sampling_std": hp.uniform("sampling_std", a, b)}
            dist = stats.uniform(a, b)
    else:
        raise NotImplementedError(f"Distribution {distribution} is not implemented!")

    # Define early stopping function
    if early_stopping == "progress":
        early_stop_fn = no_progress_loss(iteration_stop_count=10, percent_increase=0.05)
    elif early_stopping == "plateau":
        early_stop_fn = plateau_loss(std=0.0005, top=5, patience=5)
    elif early_stopping is None:
        early_stop_fn = None
    else:
        raise NotImplementedError(f"Early stopping {early_stopping} is not implemented!")

    # Define assessment function
    fn = partial(assess_configuration,
                 estimator=estimator, data=data, val_metric=val_metric)

    # Start hyperparameter optimization
    trials = Trials()
    start = time.time()
    fmin(
        fn=fn,
        space=search_space,
        algo=sampler,
        max_evals=max_evals,
        early_stop_fn=early_stop_fn,
        trials=trials
    )
    end = time.time()

    # Plot results
    x = np.array([trial["misc"]["vals"]["sampling_std"][0] for trial in trials.trials])
    y = np.array([trial["result"]["loss"] for trial in trials.trials])

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    fig, ax1 = plt.subplots()

    ax1.scatter(x=x, y=y, color="blue")  # Samples
    plt.ylabel(val_metric)

    ax2 = ax1.twinx()  # Distributions
    x_ = np.linspace(0, 3, 600)
    ax2.fill_between(x_, dist.pdf(x_), alpha=0.33, color="orange")

    ax1.set_xlabel("sampling std")
    ax1.set_ylabel(val_metric)
    ax2.set_ylabel("PDF")
    plt.xlim([0, 3])

    filename = f"{checkpoint_path}/{algo}_{distribution}_{early_stopping}_" \
               f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.png"

    plt.savefig(filename)
    plt.close()

    # Sort trials by score
    idx = np.argsort(y)

    return float(y[idx[0]]), float(x[idx[0]]), len(idx), end-start
