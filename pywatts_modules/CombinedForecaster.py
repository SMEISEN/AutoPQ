import xarray as xr
from typing import Dict, Union
from sktime.forecasting.ets import AutoETS
from pywatts.modules import SKLearnWrapper
from pywatts.modules.wrappers.base_wrapper import BaseWrapper
from pywatts.utils._xarray_time_series_utils import _get_time_indexes

from pywatts_modules.sktime_wrapper import SKTimeWrapper
from pywatts_modules.pytorch_forecasting_determ_wrapper import PyTorchForecastingDeterministicWrapper


class CombinedForecaster(BaseWrapper):
    """
    Estimator that unifies the interface for SKTime, SKlearn or PyTorch-Forecasting estimators to simplify the pipeline.
    """

    def __init__(self, forecaster: Union[SKLearnWrapper, PyTorchForecastingDeterministicWrapper, SKTimeWrapper],
                 kwargs: dict = None, use_cache: bool = True, name: str = "CombinedForecaster"):
        """
        Initialize the step.
        :param forecaster:
            Initialized point forecasting model, which is a SKTime, SKlearn or PyTorch-Forecasting estimators
        :type forecaster: Union[SKTimeWrapper, SKLearnWrapper, PyTorchForecastingDeterministicWrapper]
        :param kwargs:
            Hyperparameters to be passed to the point forecasting model (optional, default=None).
        :type kwargs: dict
        :param use_cache:
            Whether to cache point forecasts to reduce inference time in hyperparameter optimization (optional,
            default=True).
        :type use_cache: bool
        :param name:
            Step name in the pyWATTS pipeline (optional, default='CombinedForecaster')
        :type name: str
        """
        super().__init__(name)

        if kwargs is None:
            kwargs = {}

        self.forecaster = forecaster
        self.kwargs = kwargs
        self.use_cache = use_cache

        self._set_params(kwargs)
        self._transform_cache = None
        self._last_ts = None

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the Ensemble object.
        :return: Parameters.
        :rtype: Dict[str, object]
        """

        return {
            "forecaster": self.forecaster,
            "kwargs": self.kwargs,
            "use_cache": self.use_cache
        }

    def set_params(self, forecaster: Union[SKTimeWrapper, SKLearnWrapper, PyTorchForecastingDeterministicWrapper] = None,
                   kwargs: dict = None, use_cache: bool = None) -> None:
        """
        Set or change parameters of the CombinedForecaster object.
        :param forecaster:
            Initialized point forecasting model, which is a SKTime, SKlearn or PyTorch-Forecasting estimators
        :type forecaster: Union[SKTimeWrapper, SKLearnWrapper, PyTorchForecastingDeterministicWrapper]
        :param kwargs:
            Hyperparameters to be passed to the point forecasting model.
        :type kwargs: dict
        :param use_cache:
            Whether to cache point forecasts to reduce inference time in hyperparameter optimization.
        :type use_cache: bool
        :return: None
        :rtype: None
        """

        if forecaster is not None:
            self.forecaster = forecaster
        if kwargs is not None:
            self.kwargs = kwargs
            self._set_params(kwargs)
        if use_cache is not None:
            self.use_cache = use_cache

    def fit(self, **kwargs: Dict[str, xr.DataArray]) -> None:
        """
        Fit the point forecasting model.
        :param kwargs:
            Input data including the target time series and covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return: None
        :rtype: None
        """

        kwargs = self._filter_kwargs(kwargs, remove_target=False)

        if type(self.forecaster) == SKTimeWrapper and type(self.forecaster.module) == AutoETS:
            # ETS requires strictly positive time series for multiplicative composition.
            kwargs = self._scale_kwargs(kwargs=kwargs, fit=True)

        # Fit point forecaster
        self.forecaster.fit(**kwargs)
        self.is_fitted = True

    def transform(self, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Make forecasts with the trained point forecasting model.
        :param kwargs:
            Input data including the target time series and covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            Point forecast.
        :rtype: xr.DataArray
        """

        kwargs = self._filter_kwargs(kwargs, remove_target=True)

        if self.use_cache:
            # Check whether a point forecast has already been cached and whether to update the cache
            ts_data = list(kwargs.values())[0][_get_time_indexes(kwargs)[0]].values
            if self._transform_cache is not None and ts_data[-1] == self._last_ts:
                return self._transform_cache
            self._last_ts = ts_data[-1]

        if type(self.forecaster) == SKTimeWrapper and type(self.forecaster.module) == AutoETS:
            # ETS requires strictly positive time series for multiplicative composition.
            kwargs = self._scale_kwargs(kwargs=kwargs, fit=False)

        # Compute point forecast
        self._transform_cache = self.forecaster.transform(**kwargs)

        if type(self.forecaster) == SKTimeWrapper and type(self.forecaster.module) == AutoETS:
            # Rescale the ETS point forecast to original scale.
            self._transform_cache = self._rescale_res(self._transform_cache)

        return self._transform_cache

    def _filter_kwargs(self, kwargs: Dict[str, xr.DataArray], remove_target: bool) -> Dict[str, xr.DataArray]:
        """
        Filter input data based on the forecasting family (statistical modeling, machine learning, or deep learning).
        :param kwargs:
            Input data including the target time series and the covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :param remove_target:
            Whether to remove the target (transform) or not (fit).
        :type remove_target: bool
        :return:
            Filtered input data.
        :rtype: Dict[str, xr.DataArray]
        """

        if type(self.forecaster) == SKTimeWrapper:
            # Statistical Modeling (SM)
            kwargs = {key: kwargs[key] for key in kwargs if "_SM_" in key}  # Drop inputs for ML and DL
            target_name = [k for k in kwargs.keys() if k.startswith("t_SM")][0]
            update_name = [k for k in kwargs.keys() if k.startswith("u_SM")][0]
            # Rename target and update
            kwargs["target"] = kwargs.pop(target_name)
            kwargs["update"] = kwargs.pop(update_name)  # copy of the target time series for updating the past horizon
            return kwargs
        elif type(self.forecaster) == SKLearnWrapper:
            # Machine Learning (ML)
            kwargs = {key: kwargs[key] for key in kwargs if "_ML_" in key}
            target_name = [k for k in kwargs.keys() if k.startswith("t_ML")][0]  # Drop inputs for SM and DL
            if remove_target:
                kwargs.pop(target_name)
            else:
                kwargs["target"] = kwargs.pop(target_name)
            return kwargs
        elif type(self.forecaster) == PyTorchForecastingDeterministicWrapper:
            # Deep Learning (DL)
            kwargs = {key: kwargs[key] for key in kwargs if "_DL_" in key}  # Drop inputs for SM and ML
            target_name = [k for k in kwargs.keys() if k.startswith("t_DL")][0]
            kwargs["x"] = kwargs.pop(target_name)
            return kwargs
        else:
            raise NotImplementedError("Currently only SKTime, SKlearn or PyTorch-Forecasting estimators are implemented")

    def _set_params(self, kwargs: dict) -> None:
        """
        Set or change parameters of the CombinedForecaster object.
        :param kwargs:
            Hyperparameters to be passed to the point forecasting model.
        :type kwargs: dict
        :return: None
        :rtype: None
        """

        if type(self.forecaster) == SKTimeWrapper:
            # Statistical Modeling (SM)
            self.forecaster.set_params(**kwargs)
        elif type(self.forecaster) == SKLearnWrapper:
            # Machine Learning (ML)
            self.forecaster.set_params(**kwargs)
        elif type(self.forecaster) == PyTorchForecastingDeterministicWrapper:
            # Deep Learning (DL)
            if "batch_size" in kwargs.keys():
                batch_size = kwargs.pop("batch_size")
                self.forecaster.batch_size = batch_size
            self.forecaster.model_kwargs = kwargs
        else:
            raise NotImplementedError("Currently only SKTime, SKlearn or PyTorch-Forecasting estimators are implemented")

    def _scale_kwargs(self, kwargs: Dict[str, xr.DataArray], fit: bool) -> Dict[str, xr.DataArray]:
        """
        Scale input data by adding intercept to ensure that the time series is positive.
        :param kwargs:
            Input data including the target time series and the time series for updating the past horizon (i.e., a copy of
            the target time series).
        :type kwargs: Dict[str, xr.DataArray]
        :param fit:
            Whether to determine intercept based on the training data (fit) or not (transform).
        :type fit: bool
        :return:
            Scaled input data.
        :rtype: Dict[str, xr.DataArray]
        """

        if fit:
            # Determine intercept based on minium realization in the training data, multipy by safety factor
            self.intercept = abs(min(kwargs["target"].data)) * 6

        kwargs["target"] = kwargs["target"] + self.intercept
        kwargs["update"] = kwargs["update"] + self.intercept

        return kwargs

    def _rescale_res(self, res: xr.DataArray) -> xr.DataArray:
        """
        Rescale result to the original scale by subtracting the intercept.
        :param res:
            Scaled result.
        :type res: xr.DataArray
        :return:
            Rescaled result.
        :rtype: xr.DataArray
        """

        return res - self.intercept
