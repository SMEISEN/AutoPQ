import torch
import pickle
import numpy as np
import xarray as xr
from abc import ABC
from typing import Dict
from scipy import stats

from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indexes

from pywatts_modules.generator_base import GeneratorBase
from pywatts_modules.inn_base_functions import INN
from torch.utils.data import DataLoader


class INNWrapperBase(GeneratorBase, ABC):
    """
    Base class for the generative conditional Invertible Neural Network (cINN).
    """

    def __init__(self, quantiles: list = None, sample_size=100, std=0.1, name: str = "INN", **kwargs):
        """
        :param quantiles:
            Quantiles to be computed in percentage.
        :type quantiles: list
        :param sample_size:
            Number of samples randomly drawn from the cINN's latent space around the given point forecast (optional,
            default=100).
        :type sample_size: int
        :param std:
            Standard deviation for randomly sampling in the cINN's latent space around the given point forecast (optional,
            default=0.1).
        :type std: float
        :param name:
            Step name in the pyWATTS pipeline (optional, default='INN').
        :type name: str
        """
        super().__init__(name=name, **kwargs)
        if quantiles is None:
            quantiles = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]
        self.quantiles = quantiles
        self.std = std
        self.sample_size = sample_size

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the INN object.
        :return: Parameters.
        :rtype: Dict[str, object]
        """
        return {
            "epochs": self.epochs,
            "horizon": self.horizon,
            "cond_features": self.cond_features,
        }

    def set_params(self, epochs: int = None, horizon: int = None, cond_features=None) -> None:
        """
        Set or change parameters of the INNWrapperBase object.
        :param epochs:
            Number of training epochs.
        :type epochs: int
        :param horizon:
            Forecasting horizon.
        :type horizon: int
        :param cond_features:
            Number of conditional features.
        :type cond_features: int
        :return: None
        :rtype: None
        """
        if epochs is not None:
            self.epochs = epochs
        if horizon is not None:
            self.horizon = horizon
        if cond_features is not None:
            self.cond_features = cond_features

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.
        :param fm:
            Filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return:
            A dictionary containing the information needed for restoring the module.
        :rtype:Dict
        """
        json_module = super().save(fm)
        path = fm.get_path(f"module_{self.name}.pickle")
        with open(path, 'wb') as outfile:
            pickle.dump(self.generator, outfile)
        json_module["module"] = path
        return json_module

    def _transform(self, forecast: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Generate quantile forecasts using the given (trained) cINN based on the given point forecast.
        :param forecast:
            Point forecast.
        :type forecast: xr.DataArray
        :param kwargs:
            Conditional information.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            Quantile forecast.
        :rtype: xr.DataArray
        """

        # Pre-process inputs for the cINN
        x = forecast.values.reshape((len(forecast), -1))
        conds = self._get_conditions(kwargs)

        # Transform point forecast into latent space representation
        z = self.generator.forward(torch.Tensor(x), torch.Tensor(conds), rev=False)[0]

        # Analyze neighborhood of the point forecast's latent space representation by generating random samples
        noise = torch.Tensor(self.sample_size * len(x), forecast.shape[-1]).normal_(
            mean=1, std=self.std) * z.repeat(self.sample_size, 1)  # random noise around point z

        # Inversely mapping the samples from the latent space into the realization space
        samples = self.generator.forward(noise, torch.Tensor(conds).repeat(self.sample_size, 1), rev=True)[
            0].detach().numpy()
        samples = samples.reshape(self.sample_size, len(x), -1)

        # Calculate quantiles from samples
        quantiles = {}
        for k in self.quantiles:
            quantiles[k] = stats.scoreatpercentile(samples, k, axis=0)
        arr = np.array(list(quantiles.values()))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(2, 1)
        da = xr.DataArray(arr, dims=[_get_time_indexes(forecast)[0], "horizon", "quantiles"],
                          coords={"quantiles": list(quantiles.keys()),
                                  _get_time_indexes(forecast)[0]: forecast.indexes[_get_time_indexes(forecast)[0]]})
        return da

    def get_generator(self, x_features: int, cond_features: int) -> INN:
        """
        Initialize the generative cINN.
        :param x_features: int
        :type x_features:
            Number of outputs.
        :param cond_features:
            Number of conditional features.
        :type cond_features: int
        :return:
            Initialized cINN object.
        :rtype: INN
        """
        return INN(5e-4, horizon=x_features, cond_features=cond_features, n_layers_cond=10)


class INNWrapper(INNWrapperBase):

    def loss_function(self, z: torch.Tensor, log_j: torch.Tensor) -> torch.Tensor:
        """
        Maximum likelihood loss function.
        :param z:
            Latent tensor.
        :type z: torch.Tensor
        :param log_j:
            Jacobian.
        :type log_j: torch.Tensor
        :return:
            Loss
        :rtype: torch.Tensor
        """
        loss = torch.mean(z ** 2) / 2 - torch.mean(log_j) / z.shape[-1]
        return loss

    def _run_epoch(self, data_loader: DataLoader, epoch: int, conds_val: np.array, x_val: np.array) -> None:
        """
        Runs a training epoch.
        :param data_loader:
            Initialized pyTorch data loader.
        :type data_loader: DataLoader
        :param epoch:
            Training epoch number.
        :type epoch: int
        :param conds_val:
            Conditional information for the cINN (validation sub-set).
        :type conds_val: np.array
        :param x_val:
            Target time series (validation sub-set).
        :type x_val: np.array
        :return: None
        :rtype: None
        """

        # Set generative model into training mode
        self.generator.train()

        # Iterate over batches in the data loader
        for batch_idx, (data, conds) in enumerate(data_loader):

            # Calculate latent tensor and Jacobian
            z, log_j = self.generator(data, conds)

            # Calculate loss
            loss = self.loss_function(z, log_j)

            # Apply backpropagation
            self._apply_backprop(loss)

            if not batch_idx % 50:
                with torch.no_grad():
                    z, log_j = self.generator(torch.from_numpy(x_val.astype("float32")),
                                              torch.from_numpy(conds_val.astype("float32")))
                    val_loss = self.loss_function(z, log_j)
                    print(f"{epoch}, {batch_idx}, {len(data_loader.dataset)}, {loss.item()}, {val_loss.item()}")

    def transform(self, input_data: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Predict with the INN.
        :param input_data:
            Target time series.
        :type input_data: xr.DataArray
        :param kwargs:
            Covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            Prediction of INN.
        :rtype: xr.DataArray
        """
        kwargs.pop("target")
        return self._transform(input_data=input_data, **kwargs)
