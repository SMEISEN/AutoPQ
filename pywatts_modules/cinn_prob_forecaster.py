import torch
import numpy as np
import xarray as xr
from typing import Dict
from scipy import stats

from pywatts.utils._xarray_time_series_utils import _get_time_indexes
from pywatts_modules.inn import INNWrapper


class ProbForecastCINN(INNWrapper):
    """
    Generates quantiles based on a given point forecast by sampling in the latent space of a conditional Invertible Neural
    Network (cINN), which is trained to learn the underlying conditional probability distribution of the target time series.
    """

    def __init__(self, quantiles: list,
                 sample_size: int = 100, sampling_std: float = 0.1, name: str = "ProbForecastCINN"):
        super().__init__(name=name)
        """
        :param quantiles:
            Quantiles to be computed in percentage.
        :type quantiles: list
        :param sample_size:
            Number of samples randomly drawn from the cINN's latent space around the given point forecast (optional,
            default=100).
        :type sample_size: int 
        :param sampling_std:
            Standard deviation for randomly sampling in the cINN's latent space around the given point forecast (optional,
            default=0.1).
        :type sampling_std: float 
        :param name:
            Step name in the pyWATTS pipeline (optional, default='ProbForecastCINN').
        :type name: str
        """
        self.quantiles = quantiles
        self.sample_size = sample_size
        self.sampling_std = sampling_std

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the ProbForecastCINN object.
        :return: Parameters.
        :rtype: Dict[str, object]
        """
        return {
            "quantiles": self.quantiles,
            "sample_size": self.sample_size,
            "sampling_std": self.sampling_std
        }

    def set_params(self, quantiles: list = None, sample_size: int = None, sampling_std: float = None) -> None:
        """
        Set or change parameters of the ProbForecastCINN object.
        :param quantiles:
            Quantiles to be computed in percentage.
        :type quantiles: list
        :param sample_size:
            Number of samples randomly drawn from the cINN's latent space around the given point forecast.
        :type sample_size: int
        :param sampling_std:
            Standard deviation for randomly sampling in the cINN's latent space around the given point forecast.
        :type sampling_std: float
        :return: None
        :rtype: None
        """
        if quantiles is not None:
            self.quantiles = quantiles
        if sample_size is not None:
            self.sample_size = sample_size
        if sampling_std is not None:
            self.sampling_std = sampling_std

    def transform(self, forecast: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
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
            mean=1, std=self.sampling_std) * z.repeat(self.sample_size, 1)  # random noise around point z

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
                                  _get_time_indexes(forecast)[0]: forecast.indexes[
                                      _get_time_indexes(forecast)[0]]})

        return da
