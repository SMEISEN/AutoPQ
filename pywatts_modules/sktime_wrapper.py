import pickle
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from typing import Dict, Union
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import KindOfTransformDoesNotExistException, \
    KindOfTransform
from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indexes, numpy_to_xarray
from pywatts.modules.wrappers.base_wrapper import BaseWrapper
from pywatts.modules import Sampler
from sktime.transformations.base import BaseTransformer

warnings.filterwarnings('ignore', message='.*NotImplementedWarning.*', )


class SKTimeWrapper(BaseWrapper):
    """
    A wrappers class for SKTime modules.
    """

    def __init__(self, module: Union[BaseForecaster, BaseTransformer],
                 fit_kwargs: dict = None, predict_kwargs: dict = None,
                 rolling_update: bool = False, update_params: bool = False,
                 fh: int = 1, name: str = "SKTimeWrapper"):
        """
        Initialize the step.
        :param module:
            Initialized SKTime module to wrap.
        :type module: Union[BaseForecaster, BaseTransformer]
        :param fit_kwargs:
            Additional kwargs to be passed to module.fit() (optional, default=None).
        :type fit_kwargs: dict
        :param predict_kwargs:
            Additional kwargs to be passed to module.predict() (optional, default=None).
        :type predict_kwargs: dict
        :param rolling_update:
            Whether to compute rolling origin update forecasts (i.e. rolling update of the cutoff) or not (optional,
            default=False).
        :type rolling_update: bool
        :param update_params:
            Whether to update fitted parameters or not (optional, default=False).
        :type update_params: bool
        :param fh:
            The forecasting horizon in number of samples (optional, default=1)
        :type fh: int
        :param name:
            Step name in the pyWATTS pipeline (optional, default='SKTimeWrapper')
        :type name: str
        """
        if name is None:
            name = module.__class__.__name__
        super().__init__(name)
        self.module = module
        if fit_kwargs is None:
            fit_kwargs = {}
        if predict_kwargs is None:
            predict_kwargs = {}
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs
        self.update_params = update_params
        self.rolling_update = rolling_update
        self.fh = fh

        self.targets = []

        if hasattr(self.module, 'inverse_transform'):
            self.has_inverse_transform = True

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameters of the sktime wrapper.
        :return: A dict containing the module keyword arguments, the fit keyword arguments, the predict keyword
        arguments and the fitted model parameters
        :rtype: Dict
        """
        return {
            "fit_kwargs": self.fit_kwargs,
            "predict_kwargs": self.predict_kwargs,
            "update_params": self.update_params,
            "rolling_update": self.rolling_update,
            "fh": self.fh
        }

    def set_params(self, module: Union[BaseForecaster, BaseTransformer] = None, fit_kwargs: dict = None,
                   predict_kwargs: dict = None, rolling_update: bool = None, update_params: bool = None, fh: int = None):
        """
        :param module:
            Initialized SKTime module to wrap.
        :type module: Union[BaseForecaster, BaseTransformer]
        :param fit_kwargs:
            Additional kwargs to be passed to module.fit() (optional, default=None).
        :type fit_kwargs: dict
        :param predict_kwargs:
            Additional kwargs to be passed to module.predict() (optional, default=None).
        :type predict_kwargs: dict
        :param rolling_update:
            Whether to compute rolling origin update forecasts (i.e. rolling update of the cutoff) or not (optional,
            default=False).
        :type rolling_update: bool
        :param update_params:
            Whether to update fitted parameters or not (optional, default=False).
        :type update_params: bool
        :param fh:
            The forecasting horizon in number of samples (optional, default=1)
        :type fh: int
        """

        if module is not None:
            self.module = module
        if fit_kwargs is not None:
            self.fit_kwargs = fit_kwargs
        if predict_kwargs is not None:
            self.predict_kwargs = predict_kwargs
        if rolling_update is not None:
            self.rolling_update = rolling_update
        if update_params is not None:
            self.update_params = update_params
        if fh is not None:
            self.fh = fh

    def fit(self, **kwargs: Dict[str, xr.DataArray]) -> None:
        """
        Fit the SKTime module.
        :param kwargs:
            Target and covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return: None
        :rtype: None
        """

        # Split covariates, targets and update time series
        inputs, targets, _ = self.split_kwargs_inputs_targets_update(kwargs)
        target = self._dataset_to_sktime_input(targets)
        if not inputs:
            # No external input (covariate time series)
            self.module.fit(y=target, **self.fit_kwargs)
        else:
            # External input (covariate time series)
            x = self._dataset_to_sktime_input(inputs)
            self.module.fit(y=target, X=x, **self.fit_kwargs)

        self.is_fitted = True

    @staticmethod
    def _dataset_to_sktime_input(x: Dict[str, xr.DataArray]) -> pd.DataFrame:
        """
        Pre-process data set to align with the interface of SKTime modules.
        :param x:
            Time series.
        :type x: Dict[str, xr.DataArray]
        :return:
            Concatenated time series.
        :rtype: pd.DataFrame
        """

        if x is None:
            return None
        result = None
        i = 0
        for data_array in x.values():
            result_temp = pd.DataFrame(data_array.to_pandas())
            result_temp.columns = [i + j for j in range(len(result_temp.columns))]
            if result is not None:
                result = result.join(result_temp)
            else:
                result = result_temp
            i += len(result_temp.columns)
        result.index.freq = pd.infer_freq(result.index)  # Index freq is required by sktime
        return result

    @staticmethod
    def _batch_to_sktime_input(t: xr.DataArray, x: xr.DataArray, y: xr.DataArray, freq: str
                               ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Pre-process data batch to align with the interface of SKTime modules.
        :param t:
            Time index batch.
        :type t: xr.DataArray
        :param x:
            External input batch.
        :type x: xr.DataArray
        :param y:
            Target batch.
        :type y: xr.DataArray
        :param freq:
            Frequency of the time index.
        :type freq: str
        :return:
            t, time index batch
            x, external input batch
            y, target batch
        :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        """

        t = t.to_pandas()
        t.drop(t[t == pd.Timestamp(0)].index, inplace=True)  # Drop "zero" timestamps, i.e., 1970-01-01 00:00:00
        t.sort_values(inplace=True)  # Make sure the order is correct

        if len(x) == len(y):
            # External input
            x = x.to_pandas().loc[t.index].set_index(t.values)  # Drop accordingly
            x.columns = [i for i in range(len(x.columns))]  # Reset column names
            x.index.freq = freq

        y = y.to_pandas().loc[t.index].set_index(t.values)  # Drop accordingly
        y.columns = [i for i in range(len(y.columns))]  # Reset column names
        y.index.freq = freq

        return t, x, y  # Now all batches are pd

    def _sktime_output_to_dataset(self, kwargs: xr.DataArray, prediction: list) -> xr.DataArray:
        """
        Post-process SKTime module output to align with pyWATTS,
        :param kwargs:
            Input data used as reference to create xr.DataArray.
        :type kwargs: xr.DataArray
        :param prediction:
            The SKTime module's outputs.
        :type prediction: list
        :return:
            Aligned SKtime module output.
        :rtype: xr.DataArray
        """
        reference = kwargs[list(kwargs)[0]][self.fh:-self.fh]

        if isinstance(prediction, list):  # fh != 1
            coords = (
                # First dimension is number of batches. We assume that this is the time.
                ("time", list(reference.coords.values())[0].to_dataframe().index.array),
                *[(f"dim_{j}", list(range(size))) for j, size in enumerate(prediction[-1].shape[:1])])

            prediction = np.array(  # List of pd.DataFrames to np.array
                [list(pred.values.ravel()) + [np.nan] * (max(map(len, prediction)) - len(pred)) for pred in prediction])
            prediction = np.nan_to_num(prediction)  # Fill nan with zero
        else:  # fh == 1
            coords = (
                # First dimension is number of batches. We assume that this is the time.
                ("time", list(reference.coords.values())[0].to_dataframe().index.array),
                *[(f"dim_{j}", list(range(size))) for j, size in enumerate(prediction.shape[1:])])

        return xr.DataArray(prediction, coords=coords)

    def transform(self, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Transforms a dataset or predicts the result with the wrapped SKTime module.
        :param kwargs:
            The input dataset.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            The transformed output of the wrapped SKTime module.
        :rtype: xr.DataArray
        """

        time_data = list(kwargs.values())[0][_get_time_indexes(kwargs)[0]]
        freq = pd.infer_freq(time_data.data)
        inputs, _, updates = self.split_kwargs_inputs_targets_update(kwargs)
        updates = xr.concat([array.rename(name) for name, array in updates.items()], dim='dim_0')
        if len(inputs) > 0:
            # External input
            single_feature_inputs = [array.rename(name) for name, array in inputs.items() if "features" not in array.dims]
            multi_feature_inputs = [array.rename(name) for name, array in inputs.items() if "features" in array.dims]
            inputs_1, inputs_2 = None, None
            if len(single_feature_inputs) > 0:
                inputs_1 = xr.concat(single_feature_inputs, dim='dim_0')
                inputs_1 = self._reshape(inputs_1, time_data)
            if len(multi_feature_inputs) > 0:
                inputs_2 = xr.concat(multi_feature_inputs, dim='features')
                inputs_2 = inputs_2.rename({"features": "dim_0"})
                inputs_2 = self._reshape(inputs_2, time_data)

            if inputs_1 is not None and inputs_2 is not None:
                inputs = xr.concat([inputs_1, inputs_2], dim="dim_0")
            elif inputs_1 is not None:
                inputs = inputs_1
            elif inputs_2 is not None:
                inputs = inputs_2

        # Check if cutoff needs to be updated, update with one shot for transformation with rolling_update=False
        if pd.Timestamp(time_data[-1].values) > self.module.cutoff and not self.rolling_update:
            y = updates.to_pandas().set_index(time_data.values)
            y.columns = [i for i in range(len(y.columns))]
            y.index.freq = freq
            y = self._reshape(y, time_data)
            if len(inputs) > 0:
                # External input
                x = inputs.to_pandas().set_index(time_data.values)
                x.columns = [i for i in range(len(x.columns))]
                x.index.freq = freq
                x = self._reshape(x, time_data)
                self.module.update(y=y, X=x, update_params=self.update_params)
            else:
                # No external input
                self.module.update(y=y, update_params=self.update_params)

        # Training or transformation with rolling_update=False (validation)
        if pd.Timestamp(time_data[-1].values) <= self.module.cutoff and not self.rolling_update:
            fh = ForecastingHorizon(pd.DatetimeIndex(time_data.values, freq=freq), is_relative=False)
            if len(inputs) > 0:
                # External input
                x = inputs.to_pandas().set_index(time_data.values)
                x.columns = [i for i in range(len(x.columns))]
                x.index.freq = freq
            if isinstance(self.module, BaseTransformer):
                prediction = numpy_to_xarray(self.module.predict(fh=fh, X=x).values, updates)
            elif "predict" in dir(self.module):
                if len(inputs) > 0:
                    # External input
                    prediction = numpy_to_xarray(self.module.predict(fh=fh, X=x, **self.predict_kwargs).values, updates)
                else:
                    # No external input
                    prediction = numpy_to_xarray(self.module.predict(fh=fh, **self.predict_kwargs).values, updates)
            else:
                raise KindOfTransformDoesNotExistException(
                    f"The sktime-module in {self.name} does not have a predict or transform method",
                    KindOfTransform.PREDICT_TRANSFORM)
            sampler = Sampler(self.fh)
            return sampler.transform(prediction)[2 * self.fh:-2 * self.fh, :]
        else:
            # Sample data for rolling origin update forecast and cut away zeros
            sampler = Sampler(self.fh)
            time_data = sampler.transform(time_data)[self.fh - 1:, :]
            updates = sampler.transform(updates)[self.fh - 1:, :, :]
            if len(inputs) > 0:
                inputs = sampler.transform(inputs)[self.fh - 1:, :, :]
            else:
                inputs = [[] for _ in updates]  # No external input, create empty dummy to iterate over

            prediction = []
            for time_batch, x_batch, y_batch in zip(time_data, inputs, updates):  # Iterate over xr
                time_batch, x_batch, y_batch = self._batch_to_sktime_input(time_batch, x_batch, y_batch, freq)
                fh = ForecastingHorizon(pd.DatetimeIndex(time_batch.values, freq=freq), is_relative=False)

                if isinstance(self.module, BaseTransformer):
                    prediction.append(pd.DataFrame(self.module.transform(X=x_batch)))
                elif "predict" in dir(self.module):
                    if len(x_batch) == len(y_batch):
                        # External input
                        prediction.append(pd.DataFrame(self.module.predict(fh=fh, X=x_batch, **self.predict_kwargs)))
                        if pd.Timestamp(fh.to_numpy()[0]) > self.module.cutoff:
                            # Add new ground truth to the model if in pipeline.test
                            self.module.update(y=y_batch.iloc[[0]], X=x_batch.iloc[[0]], update_params=self.update_params)
                    else:
                        # No external input
                        prediction.append(pd.DataFrame(self.module.predict(fh=fh)))
                        if pd.Timestamp(fh.to_numpy()[0]) > self.module.cutoff:
                            # Add new ground truth to the model if in pipeline.test
                            self.module.update(y=y_batch.iloc[[0]], update_params=self.update_params)
                else:
                    raise KindOfTransformDoesNotExistException(
                        f"The sktime-module in {self.name} does not have a predict or transform method",
                        KindOfTransform.PREDICT_TRANSFORM)

            # Add remaining values
            for i in range(1, self.fh):
                if pd.Timestamp(fh.to_numpy()[i]) > self.module.cutoff:
                    if len(x_batch) == len(y_batch):
                        # External input
                        self.module.update(y=y_batch.iloc[[i]], X=x_batch.iloc[[i]], update_params=self.update_params)
                    else:
                        # No external input
                        self.module.update(y=y_batch.iloc[[i]], update_params=self.update_params)
            return self._sktime_output_to_dataset(kwargs, prediction[1:-self.fh])[self.fh:-self.fh, :]

    def inverse_transform(self, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Performs the inverse transform of a dataset with the wrapped SKTime module
        :param kwargs:
            The input dataset.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            The inverse-transformed output of the wrapped SKTime module.
        :rtype: xr.DataArray
        """

        x_pd = self._dataset_to_sktime_input(kwargs)

        if self.has_inverse_transform:
            prediction = self.module.inverse_transform(x_pd)
        else:
            raise KindOfTransformDoesNotExistException(
                f"The sktime-module in {self.name} does not have a inverse transform method",
                KindOfTransform.INVERSE_TRANSFORM)

        return self._sktime_output_to_dataset(kwargs, prediction)

    def predict_proba(self, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Performs the probabilistic transform of a dataset with the wrapped SKTime module
        :param kwargs:
            The input dataset.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            The transformed output of the wrapped SKTime module.
        :rtype: xr.DataArray
        """

        x_pd = self._dataset_to_sktime_input(kwargs)
        time_data = list(kwargs.values())[0][_get_time_indexes(kwargs)[0]]
        fh = ForecastingHorizon(pd.DatetimeIndex(time_data.values, freq='infer'), is_relative=False)

        if self.return_pred_int:
            if self.use_exog:
                prediction = self.module.predict_quantiles(fh=fh, X=x_pd, alpha=self.alpha)
            else:
                prediction = self.module.predict_quantiles(fh=fh, alpha=self.alpha)
        else:
            raise KindOfTransformDoesNotExistException(
                f"The sktime-module in {self.name} does not have a return_pred_int method",
                KindOfTransform.PROBABILISTIC_TRANSFORM)

        return self._sktime_output_to_dataset(kwargs, prediction)

    def save(self, fm: FileManager) -> dict:
        """
        Save module.
        :param fm:
            The pyWATTS FileManager.
        :type fm: FileManager
        :return:
            JSON including the file path of the pickled SKTime module.
        :rtype: dict
        """
        json = super().save(fm)
        file_path = fm.get_path(f'{self.name}.pickle')
        with open(file_path, 'wb') as outfile:
            pickle.dump(obj=self.module, file=outfile)
        json.update({"sktime_module": file_path})
        return json

    @classmethod
    def load(cls, load_information: dict) -> 'SKTimeWrapper':
        """
        Load module.
        :param load_information:
            Information for loading the SKTimeWrapper.
        :type load_information: dict
        :return:
            The loaded SKTimeWrapper.
        :rtype: SKTimeWrapper
        """

        name = load_information["name"]
        with open(load_information["sktime_module"], 'rb') as pickle_file:
            module = pickle.load(pickle_file)
        module = cls(module=module, name=name)
        module.is_fitted = load_information["is_fitted"]

        return module

    @staticmethod
    def split_kwargs_inputs_targets_update(kwargs: Dict[str, xr.DataArray]
                                           ) -> (Dict[str, xr.DataArray], Dict[str, xr.DataArray], Dict[str, xr.DataArray]):
        """
        Split the input time series into external inputs (covariates), target and update time series.
        :param kwargs:
            Input time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            inputs, the external inputs (covariate time series)
            target, the target time series
            update, the time series to update the past horizon (i.e. a copy of the target time series).
        :rtype: (Dict[str, xr.DataArray], Dict[str, xr.DataArray], Dict[str, xr.DataArray])
        """
        inputs = dict()
        targets = dict()
        updates = dict()
        for key, value in kwargs.items():
            if key.startswith("target"):
                targets[key] = value
            elif key.startswith("update"):
                updates[key] = value
            else:
                inputs[key] = value
        return inputs, targets, updates

    @staticmethod
    def _reshape(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape the data to align with the reference.
        :param data:
            Data to be aligned.
        :type data: pd.DataFrame
        :param reference:
            Reference for alignment.
        :type reference: pd.DataFrame
        :return:
            Reshaped data.
        :rtype: pd.DataFrame
        """
        if len(data) != len(reference):
            data = data.T
        return data
