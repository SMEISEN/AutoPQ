import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from copy import deepcopy
from random import Random
from typing import List, Tuple, Union
from propulate.population import Individual
from pywatts.core.pipeline import Pipeline
from pywatts.core.summary_formatter import SummaryJSON

from autopq.config import PointForecaster, ComputingResources, ConfigSpace, HyperoptAlgo, ValMetric
from autopq.hpo.hyperopt import hyperopt_search
from autopq.hpo.propulate import propulate_search
from autopq.hpo.utils import map_forecaster_p_params, evaluate_trial
from autopq.sub_pipelines.forecaster_p import add_forecaster_p
from autopq.sub_pipelines.forecaster_q import add_forecaster_q
from autopq.sub_pipelines.preprocessing import add_preprocessing
from autopq.sub_pipelines.utils import create_modules, load_modules, save_modules


class AutoPQ:
    """
    AutoPQ: Template for automated point forecast-based quantile forecasts
    The underlying idea of AutoPQ is to generate a probabilistic forecast based on an arbitrary point forecast using a
    conditional Invertible Neural Network (cINN) and to make corresponding design decisions automatically, aiming to increase
    the probabilistic performance.
    """

    def __init__(self, target_name: str, feature_names: List[str], forecaster_p_name: PointForecaster,
                 computing_resources: ComputingResources = ComputingResources.Default,
                 horizon: int = 24, forecast_limits: Tuple[Union[float, None], Union[float, None]] = None,
                 quantiles: list = None, forecaster_p_params: dict = None, forecaster_q_params: dict = None,
                 val_metric: ValMetric = ValMetric.CRPS,
                 hp_checkpoint_path: str = f"{os.getcwd()}/results/Hyperopt", hp_algo: HyperoptAlgo = HyperoptAlgo.TPE,
                 hp_max_evals: int = 100,
                 ppl_mate_prob: float = 0.7, ppl_mut_prob: float = 0.4, ppl_random_prob: float = 0.2,
                 ppl_migration_prob: float = 0.7, ppl_generations: int = 100, ppl_num_isles: int = 2,
                 ppl_checkpoint_path: str = f"{os.getcwd()}/results/Propulate"
                 ):
        """
        :param target_name:
            Column name of the target in the dataframe given in train and predict.
        :type target_name: str
        :param feature_names:
            Column names of the features in the dataframe given in train and predict.
        :type feature_names: List[str]
        :param forecaster_p_name:
            Name of the point forecasting method.
        :type forecaster_p_name: PointForecaster
        :param computing_resources:
            Available computing resources, determines complexity of hyperparameter optimization and achievable probabilistic
            forecasting accuracy (optional, default=ComputingResources.Default).
        :type computing_resources: ComputingResources
        :param horizon:
            Forecasting horizon in number of time stamps (optional, default=24).
        :type horizon: int
        :param forecast_limits:
            Physical limitations (min and max) of the target time series (optional, default=None).
        :type forecast_limits: Tuple[Union[float, None], Union[float, None]]
        :param quantiles:
            Generated quantiles (optional, default=[50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]).
        :type quantiles: list
        :param forecaster_p_params:
            Hyperparameters of the point forecasting method (optional, default=None).
        :type forecaster_p_params: dict
        :param forecaster_q_params:
            Hyperparameters of the quantile forecasting method (optional, default=None).
        :type forecaster_q_params: dict
        :param val_metric:
            Validation metric for the hyperparameter optimization. (optional, default=ValMetric.CRPS).
        :type val_metric: ValMetric
        :param hp_checkpoint_path:
            Hyperopt checkpoint path (optional, default=f"{os.getcwd()}/results/Hyperopt").
        :type hp_checkpoint_path: str
        :param hp_algo:
            Hyperopt search algorithm (optional, default=HyperoptAlgo.TPE).
        :type hp_algo: HyperoptAlgo.TPE
        :param hp_max_evals:
            Hyperopt number of iterations (optional, default=100).
        :type hp_max_evals: int
        :param ppl_mate_prob:
            Propulate crossover probability (optional, default=0.7).
        :type ppl_mate_prob: float
        :param ppl_mut_prob:
            Propulate point mutation probability (optional, default=0.4).
        :type ppl_mut_prob: float
        :param ppl_random_prob:
            Propulate random-initialization (optional, default=0.2).
        :type ppl_random_prob: float
        :param ppl_migration_prob:
            Propulate migration probability (optional, default=0.7).
        :type ppl_migration_prob: float
        :param ppl_generations:
            Propulate number of generations (optional, default=100).
        :type ppl_generations: int
        :param ppl_num_isles:
            Propulate numer of isles (optional, default=2).
        :type ppl_num_isles: int
        :param ppl_checkpoint_path:
            Propulate checkpoint path (optional, default=f"{os.getcwd()}/results/Propulate").
        :type ppl_checkpoint_path: str
        """

        if quantiles is None:
            quantiles = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]
        if forecast_limits is None:
            forecast_limits = (None, None)
        if forecaster_p_params is None:
            forecaster_p_params = {}
        if forecaster_q_params is None:
            forecaster_q_params = {"quantiles": quantiles}

        self.target_name = target_name
        self.feature_names = feature_names
        self.forecaster_p_name = forecaster_p_name
        self.computing_resources = computing_resources
        self.horizon = horizon
        self.quantiles = quantiles
        self.forecast_limits = forecast_limits
        self.forecaster_p_params = forecaster_p_params
        self.forecaster_q_params = forecaster_q_params
        self.val_metric = val_metric

        # Initialize Hyperopt params
        self.hp_checkpoint_path = hp_checkpoint_path
        self.hp_algo = hp_algo
        self.hp_max_evals = hp_max_evals

        # Initialize Propulate params
        self.ppl_mate_prob = ppl_mate_prob
        self.ppl_mut_prob = ppl_mut_prob
        self.ppl_random_prob = ppl_random_prob
        self.ppl_migration_prob = ppl_migration_prob
        self.ppl_generations = ppl_generations
        self.ppl_num_isles = ppl_num_isles
        self.ppl_checkpoint_path = ppl_checkpoint_path

        # Initialize internal params
        self._modules = None
        self._train = None
        self._val = None
        self._test = None
        self._identification_times = []

        # Initialize attributes
        self.val_score_ = None

    def _create_pipeline(self, modules: dict) -> Pipeline:
        """
        Creates the pyWATTS pipeline, consisting of pre-processing, and the creation of the point forecasting model, and the
        quantile forecasting model.
        :param modules:
            Initialized modules of the pyWATTS pipeline
        :type modules: dict
        :return:
            The pyWATTS pipeline
        :rtype: Pipeline
        """
        pipeline = Pipeline(path=f"{os.getcwd()}/results/pipeline", name="AutoPQ")
        pipeline = add_preprocessing(pipeline=pipeline, modules=modules, target_name=self.target_name,
                                     feature_names=self.feature_names, horizon=self.horizon)
        pipeline = add_forecaster_p(pipeline=pipeline, modules=modules, forecast_limits=self.forecast_limits)
        pipeline = add_forecaster_q(pipeline=pipeline, modules=modules, forecast_limits=self.forecast_limits)

        return pipeline

    def _assess(self, configuration: Individual) -> float:
        """
        Assesses a trial configuration by training the pipeline and evaluating the performance on the validation data.
        :param configuration:
            Trial configuration to be evaluated in the hyperparameter optimization.
        :type configuration: Individual
        :return:
            Score on the validation data set.
        :rtype: float
        """

        # Map params
        params = deepcopy(dict(configuration))
        forecaster_p_params = map_forecaster_p_params(forecaster_name=self.forecaster_p_name, params=params)

        # Create modules
        self._modules = create_modules(modules=self._modules, target_name=self.target_name, feature_names=self.feature_names,
                                       horizon=self.horizon, forecaster_p_name=self.forecaster_p_name,
                                       forecaster_p_params=forecaster_p_params,
                                       forecaster_q_params=self.forecaster_q_params)

        # Create pipeline
        start = time.time()
        pipeline = self._create_pipeline(modules=self._modules)
        pipeline.train(data=self._train, summary=False, summary_formatter=None)
        training_time = time.time() - start

        if self.computing_resources == ComputingResources.Default or (
                self.computing_resources == ComputingResources.Advanced and
                any([np.isnan(configuration.population_statistics["sampling_std__mu"]),
                     np.isnan(configuration.population_statistics["sampling_std__sigma"])])):
            # Define uniform prior distribution
            distribution = "uniform"
            log = False
            a = ConfigSpace["cINN"]["sampling_std"][0]
            b = ConfigSpace["cINN"]["sampling_std"][1]
        else:
            # Define prior distribution using population statistics
            distribution = "normal"
            log = True
            a = configuration.population_statistics["sampling_std__mu"]
            b = configuration.population_statistics["sampling_std__sigma"]

        # Skip inner loop if training time is shorter than identification time (Algorithm 3)
        # i.e., sampling hyperparameter is optimized using Propulate
        if len(self._identification_times) <= 5 and training_time < np.mean(self._identification_times[-5:-1]):
            result = pipeline.test(data=self._val, summary=False, summary_formatter=None)
            val_score = float(evaluate_trial(val_metric=self.val_metric,
                                             y=result["target_scale"], y_hat=result["forecast_q_scale"]))
            sampling_std = configuration.sampling_std

        # Hyperparameter optimization of the sampling hyperparameter (Algorithm 1)
        # i.e. sampling hyperparameter is optimized using Hyperopt
        else:
            start = time.time()
            search_kwargs = {"estimator": pipeline, "data": self._val, "val_metric": self.val_metric,
                             "algo": self.hp_algo, "max_evals": self.hp_max_evals, "early_stopping": "plateau",
                             "distribution": distribution, "log": log, "a": a, "b": b,
                             "checkpoint_path": self.hp_checkpoint_path}
            val_score, sampling_std, _, _ = hyperopt_search(**search_kwargs)
            self._identification_times.append(time.time() - start)

        configuration.sampling_std = sampling_std
        configuration.pipeline_modules = self._modules

        return val_score

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        """
        Fit AutoWP to the data.
        :param train:
            Data set for training the forecasting pipeline.
        :type train: pd.DataFrame
        :param val:
            Data set for validating the forecasting pipeline in the hyperparameter optimization.
        :type train: pd.DataFrame
        :return:
            Predictions on the training data. Consists of the measurement (measurement, measurement_scale), the point
            forecast (forecast_p, forecast_p_scale), and the quantile forecast (forecast_q, forecast_q_scale). Each is
            provided in the original and normalized scale (_scale).
        :rtype: dict
        """

        self._train = train
        self._val = val

        if self.computing_resources == ComputingResources.Default:
            # Fit pipeline and optimize sampling hyperparameter
            configuration = Individual()
            configuration.update({**self.forecaster_p_params})
            self.val_score_ = self._assess(configuration)
            save_modules(pipeline_modules=configuration.pipeline_modules, dry=self.ppl_checkpoint_path)
        elif self.computing_resources == ComputingResources.Advanced:
            # Combined hyperparameter optimization of the point forecaster and the sampling hyperparameter
            search_kwargs = {
                "rng": Random(MPI.COMM_WORLD.rank), "pop_size": 2 * MPI.COMM_WORLD.size,
                "limits": {**ConfigSpace[self.forecaster_p_name]}, "loss_fn": self._assess, "generations": self.ppl_generations,
                "mate_prob": self.ppl_mate_prob, "mut_prob": self.ppl_mut_prob, "random_prob": self.ppl_random_prob,
                "migration_probability": self.ppl_migration_prob, "num_isles": self.ppl_num_isles,
                "checkpoint_path": self.ppl_checkpoint_path
            }
            configuration = propulate_search(**search_kwargs)

        # Get configurations and fitted modules
        self.forecaster_q_params["sampling_std"] = configuration.sampling_std

        # Create pipeline with best configuration
        modules = load_modules(pipeline_modules=self._modules, dry=self.ppl_checkpoint_path)
        pipeline = self._create_pipeline(modules=modules)

        # Predict with pipeline
        res = pipeline.test(data=self._train, summary=False, summary_formatter=SummaryJSON())

        return res

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Predict using the AutoWP model.
        :param data:
            Data set for making predictions with the forecasting pipeline.
        :type data: pd.DataFrame
        :return:
            Predictions on the test data. Consists of the measurement (measurement, measurement_scale), the point
            forecast (forecast_p, forecast_p_scale), and the quantile forecast (forecast_q, forecast_q_scale). Each is
            provided in the original and normalized scale (_scale).
        :rtype: dict
        """

        # Load modules of pipeline
        modules = create_modules(modules=self._modules, target_name=self.target_name, feature_names=self.feature_names,
                                 horizon=self.horizon, forecaster_p_name=self.forecaster_p_name,
                                 forecaster_p_params=self.forecaster_p_params,
                                 forecaster_q_params=self.forecaster_q_params)
        modules = load_modules(pipeline_modules=modules, dry=self.ppl_checkpoint_path)
        pipeline = self._create_pipeline(modules=modules)

        # Predict with pipeline
        result = pipeline.test(data=data, summary=False, summary_formatter=SummaryJSON())

        return result
