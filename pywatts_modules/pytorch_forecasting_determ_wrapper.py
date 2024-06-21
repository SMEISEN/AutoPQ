import multiprocessing
import numpy as np
import xarray as xr
import pandas as pd
import pytorch_lightning as pl
from typing import Dict, Callable
from pytorch_forecasting import TimeSeriesDataSet, BaseModelWithCovariates
from pytorch_lightning.callbacks import EarlyStopping
from pywatts.utils._xarray_time_series_utils import _get_time_indexes
from pywatts.modules.wrappers.dl_wrapper import DlWrapper


class PyTorchForecastingDeterministicWrapper(DlWrapper):
    """
    Wrapper class for PyTorch forecasting models.
    """

    def __init__(self, model: BaseModelWithCovariates, max_encoder_length: int, max_prediction_length: int,
                 n_gpu: int = 0, batch_size: int = 64,
                 fit_kwargs: dict = None, model_kwargs: dict = None,
                 trainer_kwargs: dict = None, process_results_fn: Callable = None,
                 training_dataset_kwargs: dict = None, validation_dataset_kwargs: dict = None,
                 name: str = "PyTorchForecastingWrapper",):
        """
        Initialize the step.
        :param model:
            Initialized PyTorch Forecasting model to wrap.
        :type model: BaseModelWithCovariates
        :param max_encoder_length:
             Maximum length to encode. This is the maximum history length used by the time series dataset.
        :type max_encoder_length: int
        :param max_prediction_length:
            Maximum prediction/decoder length.
        :type max_prediction_length: int
        :param n_gpu:
            Number of GPUs available.
        :type n_gpu: int
        :param batch_size:
            Batch size for training model (optional, default=64).
        :type batch_size: int
        :param fit_kwargs:
            Additional kwargs to be passed to trainer.fit() (optional, default=None).
        :type fit_kwargs: dict
        :param model_kwargs:
            Additional kwargs to be passed to model.from_dataset() (optional, default=None).
        :type model_kwargs: dict
        :param trainer_kwargs:
            Additional kwargs to be passed to pl.Trainer() (optional, default=None).
        :type trainer_kwargs: dict
        :param process_results_fn:
            Function to process the point forecast (optional default=None).
        :type process_results_fn: Callable
        :param training_dataset_kwargs:
            Additional kwargs to be passed to the training TimeSeriesDataSet() (optional, default=None).
        :type training_dataset_kwargs: dict
        :param validation_dataset_kwargs:
            Additional kwargs to be passed to the validation TimeSeriesDataSet() (optional, default=None).
        :type validation_dataset_kwargs: dict
        :param name:
            Step name in the pyWATTS pipeline (optional, default='PyTorchForecastingWrapper')
        :type name: str
        """
        super().__init__(model, name, fit_kwargs)

        if model_kwargs is None:
            model_kwargs = {}
        if training_dataset_kwargs is None:
            training_dataset_kwargs = {}
        if validation_dataset_kwargs is None:
            validation_dataset_kwargs = {}
        if trainer_kwargs is None:
            trainer_kwargs = {}

        self.model_kwargs = model_kwargs
        self.training_dataset_kwargs = training_dataset_kwargs
        self.validation_dataset_kwargs = validation_dataset_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.n_gpu = n_gpu
        self.process_results_fn = process_results_fn

        self.targets = []
        self.ts_index = None
        self.last = None
        self.freq = None
        self.trained_model = None
        self.is_fitted = False

    def fit(self, x: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> None:
        """
        Fit the point forecasting model.
        :param x:
            Target time series.
        :type x: xr.DataArray
        :param kwargs:
            Covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return: None
        :rtype: None
        """

        # Concatenate covariate time series
        externals = self._concat_externals(kwargs)
        data = pd.DataFrame(externals, columns=[str(i) for i in range(externals.shape[-1])])
        data["ts_index"] = range(len(data))
        data["value"] = x.values
        data["series"] = 0

        # Determine time index
        time_index = _get_time_indexes(x)[0]

        # Create known reals
        known_reals = set(data.columns)
        known_reals.remove("value")
        known_reals.remove("ts_index")
        known_reals.remove("series")

        # Update last observed time index
        self.ts_index = len(x)
        self.last = x.indexes[time_index][-1]
        self.freq = x.indexes[time_index][-1] - x.indexes[time_index][-2]

        # Create data loaders
        training = TimeSeriesDataSet(
            data,
            time_idx="ts_index",
            target="value",
            time_varying_unknown_reals=["value"],
            time_varying_known_reals=sorted(list(known_reals)),
            group_ids=["series"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            **self.training_dataset_kwargs
        )
        validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=int(len(x) * 0.8),
                                                    **self.validation_dataset_kwargs)

        train_dataloader = training.to_dataloader(
            train=True, shuffle=True, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count() // 2,
            persistent_workers=True
        )
        val_dataloader = validation.to_dataloader(
            train=False, shuffle=False, batch_size=len(validation), num_workers=0
        )

        # Initialize model with dataloader
        model = self.model.from_dataset(training, **self.model_kwargs)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=1000,
            accelerator="gpu" if self.n_gpu > 0 else "cpu",
            devices=[i for i in range(self.n_gpu)] if self.n_gpu > 0 else None,
            enable_model_summary=True,
            callbacks=[early_stop_callback],
            limit_train_batches=0.33,
            enable_checkpointing=True,
            gradient_clip_val=0.1,
            **self.trainer_kwargs
        )

        # Fit model and restore best model after early stopping
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            **self.fit_kwargs
        )
        best_model_path = trainer.checkpoint_callback.best_model_path
        self.trained_model = self.model.load_from_checkpoint(best_model_path)

        self.is_fitted = True

    def transform(self, x: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Make forecasts with the trained point forecasting model.
        :param x:
            Target time series.
        :type x: xr.DataArray
        :param kwargs:
            Covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            Point forecast.
        :rtype: xr.DataArray
        """

        # Concatenate covariate time series
        externals = self._concat_externals(kwargs)
        data = pd.DataFrame(externals, columns=[str(i) for i in range(externals.shape[-1])])
        data["value"] = x.values
        data["series"] = 0

        # Determine time index
        time_index = _get_time_indexes(x)[0]
        new_last = x.indexes[time_index][-1]

        # Create known reals
        nums = int((new_last - self.last) / self.freq)
        data["ts_index"] = range(self.ts_index + nums - len(data), self.ts_index + nums)
        known_reals = set(data.columns)
        known_reals.remove("value")
        known_reals.remove("ts_index")
        known_reals.remove("series")

        if new_last != self.last:
            # Update last observed time index
            self.last = new_last
            self.ts_index += nums

        # Create data loaders
        test = TimeSeriesDataSet(
            data,
            time_idx="ts_index",
            target="value",
            time_varying_unknown_reals=["value"],
            time_varying_known_reals=self.trained_model.decoder_variables,
            group_ids=["series"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            **self.training_dataset_kwargs
        )
        test_dataloader = test.to_dataloader(
            train=False, shuffle=False, batch_size=len(test), num_workers=0
        )

        # Make point forecast
        if self.process_results_fn is None:
            prediction = self.trained_model.predict(test_dataloader, mode="prediction")

            return xr.DataArray(
                prediction.numpy(),
                dims=[time_index, "horizon"],
                coords={
                    time_index: x[time_index][self.max_encoder_length - 1:-self.max_prediction_length],
                    "horizon": range(prediction.shape[1])})

        prediction = self.trained_model.predict(test_dataloader, mode="raw")

        return self.process_results_fn(
            prediction=prediction,
            dims=[time_index, "horizon"],
            coords={
                time_index: x[time_index][self.max_encoder_length - 1:-self.max_prediction_length],
                "horizon": range(prediction["prediction"].shape[1])},
            model=self.trained_model)

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters for the PyTorchForecastingWrapper object.
        :return: Parameters.
        :rtype: Dict[str, object]
        """
        return {
            "fit_kwargs": self.fit_kwargs,
            "model_kwargs": self.model_kwargs,
            "training_dataset_kwargs": self.training_dataset_kwargs,
            "validation_dataset_kwargs": self.validation_dataset_kwargs,
            "trainer_kwargs": self.trainer_kwargs
        }

    def set_params(self, model: BaseModelWithCovariates = None,
                   max_encoder_length: int = None, max_prediction_length: int = None,
                   n_gpu: int = None, batch_size: int = None,
                   fit_kwargs: dict = None, model_kwargs: dict = None,
                   process_results_fn: Callable = None,
                   training_dataset_kwargs: dict = None, validation_dataset_kwargs: dict = None,
                   trainer_kwargs: dict = None) -> None:
        """
        Set or change parameters of the PyTorchForecastingWrapper object.
        :param model:
            Initialized PyTorch Forecasting model.
        :type model: BaseModelWithCovariates
        :param max_encoder_length:
             Maximum length to encode. This is the maximum history length used by the time series dataset.
        :type max_encoder_length: int
        :param max_prediction_length:
            Maximum prediction/decoder length.
        :type max_prediction_length: int
        :param n_gpu:
            Number of GPUs available.
        :type n_gpu: int
        :param batch_size:
            Batch size for training model (optional, default=64).
        :type batch_size: int
        :param fit_kwargs:
            Additional kwargs to be passed to trainer.fit() (optional, default=None).
        :type fit_kwargs: dict
        :param model_kwargs:
            Additional kwargs to be passed to model.from_dataset() (optional, default=None).
        :type model_kwargs: dict
        :param trainer_kwargs:
            Additional kwargs to be passed to pl.Trainer() (optional, default=None).
        :type trainer_kwargs: dict
        :param process_results_fn:
            Function to process the point forecast (optional default=None).
        :type process_results_fn: Callable
        :param training_dataset_kwargs:
            Additional kwargs to be passed to the training TimeSeriesDataSet() (optional, default=None).
        :type training_dataset_kwargs: dict
        :param validation_dataset_kwargs:
            Additional kwargs to be passed to the validation TimeSeriesDataSet() (optional, default=None).
        :type validation_dataset_kwargs: dict
        :return: None
        :rtype: None
        """

        if model is not None:
            self.model = model
        if max_encoder_length is not None:
            self.max_encoder_length = max_encoder_length
        if max_prediction_length is not None:
            self.max_prediction_length = max_prediction_length
        if n_gpu is not None:
            self.n_gpu = n_gpu
        if batch_size is not None:
            self.batch_size = batch_size
        if fit_kwargs is not None:
            self.fit_kwargs = fit_kwargs
        if model_kwargs is not None:
            self.model_kwargs = model_kwargs
        if process_results_fn is not None:
            self.process_results_fn = process_results_fn
        if training_dataset_kwargs is not None:
            self.training_dataset_kwargs = training_dataset_kwargs
        if validation_dataset_kwargs is not None:
            self.validation_dataset_kwargs = validation_dataset_kwargs
        if trainer_kwargs is not None:
            self.trainer_kwargs = trainer_kwargs

    @staticmethod
    def _concat_externals(kwargs: Dict[str, xr.DataArray]) -> np.array:
        """
        Concatenate covariate time series into one numpy array.
        :param kwargs:
            Covariate time series.
        :type kwargs: Dict[str, xr.DataArray]
        :return:
            Concatenated covariate time series.
        :rtype: np.array
        """
        externals = []
        for key, value in kwargs.items():
            externals.append(value.values.reshape((len(value), -1)))
        return np.concatenate(externals, axis=-1)
