from abc import ABC
from typing import Dict

import numpy as np
import torch
import xarray as xr
from pywatts.core.base import BaseEstimator
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class GeneratorBase(BaseEstimator, ABC):
    """
    Base class for the generative models trained with pyTorch.
    """

    def __init__(self, epochs: int = 100, val_train_split: float = 0.2, name: str = "GeneratorBase"):
        super().__init__(name)
        """
        :param epochs:
            Number of training epochs (optional, default=100).
        :type epochs: int
        :param val_train_split:
            Percentage of randomly chosen samples used for validation (optional, default=0.2).
        :type val_train_split: float
        :param name:
            Step name in the pyWATTS pipeline (optional, default='GeneratorBase').
        :type name: str
        """
        self.epochs = epochs
        self.val_train_split = val_train_split

        self.is_fitted = False
        self.has_inverse_transform = True
        self.generator = None

    @staticmethod
    def _get_splits(kwargs: Dict[str, xr.DataArray], rdx: np.array, x: np.array) -> (np.array, np.array, np.array, np.array):
        """
        Split data into training and validation sub-sets.
        :param kwargs:
            Covariate time series used as conditional information.
        :type kwargs: Dict[str, xr.DataArray]
        :param rdx:
            Indexes of samples to be hold-out in the training (validation samples).
        :type rdx: np.array
        :param x:
            Target time series.
        :type x: np.array
        :return:
            conds_train, conditional information for the cINN (training sub-set).
            conds_val, conditional information for the cINN (validation sub-set).
            x_train, target time series (training sub-set).
            x_val, target time series (validation sub-set).
        :rtype: (np.array, np.array, np.array, np.array)
        """
        x = x.reshape((len(x), -1))
        x_train = np.delete(x, rdx, axis=0)
        x_val = x[rdx]
        conds_train = []
        conds_val = []
        for key, value in kwargs.items():
            value = value.values.reshape((len(value), -1))
            conds_train.append(np.delete(value, rdx, axis=0))
            conds_val.append(value[rdx])

        return np.concatenate(conds_train, axis=1), np.concatenate(conds_val, axis=1), x_train, x_val

    def _apply_backprop(self, nll: torch.Tensor) -> None:
        """
        Apply backpropagation on loss.
        :param nll:
            Loss.
        :type nll: torch.Tensor
        :return: None
        :rtype: None
        """
        nll.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.trainable_parameters, 1.)
        self.generator.optimizer.step()
        self.generator.optimizer.zero_grad()

    @staticmethod
    def _get_conditions(kwargs: Dict[str, xr.DataArray]) -> np.array:
        """
        Pre-process covariate time series used as conditional information for training the cINN.
        :param kwargs:
            Covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            Numpy array of concatenated covariate time series.
        :rtype: np.array
        """
        conditions = []
        for key, value in kwargs.items():
            value = value.values.reshape((len(value), -1))
            conditions.append(value)
        return np.concatenate(conditions, axis=1)

    @staticmethod
    def _create_dataloader(x_train: np.array, cond_train: np.array, batch_size: int = 512) -> DataLoader:
        """
        Creates the pyTorch data loader.
        :param x_train:
            Target time series (training sub-set).
        :type x_train: np.array
        :param cond_train:
            Conditional information for the cINN (training sub-set).
        :type cond_train: np.array
        :param batch_size:
            Batch size for the training algorithm.
        :type batch_size: int
        :return:
            Initialized pyTorch data loader.
        :rtype: DataLoader
        """
        dataset = TensorDataset(torch.from_numpy(x_train.astype("float32")),
                                torch.from_numpy(cond_train.astype("float32")))

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @classmethod
    def get_generator(cls, **kwargs: dict) -> nn.Module:
        """
        Initialize the generative model.
        :param kwargs:
            Hyperparameters of the generative model.
        :type kwargs: dict
        :return:
            Initialized generative model object.
        :rtype: nn.Module
        """
        pass

    @classmethod
    def _run_epoch(cls, **kwargs: dict) -> None:
        """
        Runs a training epoch.
        :param kwargs:
            Training kwargs.
        :type kwargs: dict
        :return: None
        :rtype: None
        """
        pass

    def fit(self, target: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> None:
        """
        Fit the generative model.
        :param target:
            Target time series.
        :type target: xr.DataArray
        :param kwargs:
            Covariate time series used as conditional information.
        :type kwargs: Dict[str, xr.DataArray]
        :return: None
        :rtype: None
        """

        # Set the target time series as training target and remove the point forecast from the input data.
        x = target.values
        kwargs.pop("forecast")

        # Split conditions and targets into training and validation sub-sets
        rdx = np.random.choice(np.arange(0, len(x)), int(len(target) * self.val_train_split), replace=False)
        cond_train, cond_val, x_train, x_val = self._get_splits(kwargs, rdx, x)

        # Initialize generative model
        self.generator = self.get_generator(x_train.shape[-1], cond_train.shape[-1])

        # Create data loader
        data_loader = self._create_dataloader(x_train, cond_train)

        # Train generative model
        for epoch in range(self.epochs):
            self._run_epoch(data_loader, epoch, cond_val, x_val)

        self.is_fitted = True
