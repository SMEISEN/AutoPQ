import os

import cloudpickle
import xarray as xr
from typing import List, Union
from mpi4py import MPI
from pywatts.modules import CalendarExtraction, CalendarFeature, SKLearnWrapper
from pywatts.core.step_information import StepInformation
from pywatts.summaries import RMSE, MAE
from torch.cuda import device_count
from multiprocessing import cpu_count
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sklearn.neural_network import MLPRegressor
from msvr.msvr import MSVR
from xgboost import XGBRegressor
from pytorch_forecasting.models import NHiTS, DeepAR, TemporalFusionTransformer

from autopq.config import PointForecaster

from pywatts_modules.cinn_prob_forecaster import ProbForecastCINN
from pywatts_modules.pytorch_forecasting_determ_wrapper import PyTorchForecastingDeterministicWrapper
from pywatts_modules.sktime_wrapper import SKTimeWrapper
from pywatts_modules.CombinedForecaster import CombinedForecaster

FORECASTER_NAMES = [

]


def create_modules(modules: Union[dict, None], target_name: str, feature_names: List[str], horizon: int,
                   forecaster_p_name: PointForecaster, forecaster_p_params: dict, forecaster_q_params: dict) -> dict:
    """
    Creates the modules for the pyWATTS pipeline.
    :param modules:
        Column name of the target in the dataframe given in train and predict.
    :type modules: Union[dict, None]
    :param target_name:
        Column name of the target in the dataframe given in train and predict.
    :type target_name: str
    :param feature_names:
        Column names of the features in the dataframe given in train and predict.
    :type feature_names: List[str]
    :param forecaster_p_name:
        Name of the point forecasting method.
    :type forecaster_p_name: PointForecaster
    :param horizon:
        Forecasting horizon in number of time stamps.
    :type horizon: int
    :param forecaster_p_params:
        Hyperparameters of the point forecasting method.
    :type forecaster_p_params: dict
    :param forecaster_q_params:
        Hyperparameters of the quantile forecasting method.
    :type forecaster_q_params: dict
    :return:
        Initialized pyWATTS modules.
    :rtype: dict
    """

    # Initialize quantile forecaster
    if modules is None:
        modules = {"forecaster_q": ProbForecastCINN(**forecaster_q_params)}

    # Initialize scalers and feature extraction
    modules["target_scaler"] = SKLearnWrapper(module=StandardScaler(), name=f"target_SM_{target_name}")
    modules["calendar_extractor"] = CalendarExtraction(continent="Europe", country="Germany",
                                                       name=f"calendar_SM_{target_name}",
                                                       features=[CalendarFeature.workday, CalendarFeature.hour_cos,
                                                                 CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                                                 CalendarFeature.month_cos])
    for feature_name in feature_names:
        modules[f"feature_scaler_{feature_name}"] = SKLearnWrapper(module=StandardScaler(),
                                                                   name=f"feature_SM_{feature_name}")

    # Initialize point forecaster
    n_cpu = (cpu_count() - 1) // 1
    n_gpu = device_count()
    dl_kwargs = {
        "max_prediction_length": horizon,
        "max_encoder_length": horizon + 1,
        "mpi": MPI,
        "n_gpu": n_gpu,
        "name": "forecaster",
        "batch_size": 64  # default in pytorch forecasting
    }

    if forecaster_p_name == PointForecaster.ETS:
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=SKTimeWrapper(AutoETS(sp=24, auto=False), fh=horizon),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.TBATS and not \
            forecaster_p_params.get("trigonometric_seasonality", True):
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=SKTimeWrapper(BATS(
                use_box_cox=False, use_trend=False, use_damped_trend=False, use_arma_errors=False, sp=24, n_jobs=1
            ), fh=horizon),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.TBATS:
        if forecaster_p_params.get("trigonometric_seasonality", False):
            forecaster_p_params.pop("trigonometric_seasonality")
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=SKTimeWrapper(TBATS(
                use_box_cox=False, use_trend=False, use_damped_trend=False, use_arma_errors=False, sp=24, n_jobs=1
            ), fh=horizon),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.sARIMAX:
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=SKTimeWrapper(SARIMAX(),  # sp is set in params
                                     fh=horizon),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.MLP:
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=SKLearnWrapper(module=MLPRegressor(
                # same config like pytorch-forecasting
                batch_size=64, max_iter=350, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, tol=1e-4
            )),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.SVR:
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=SKLearnWrapper(module=MSVR(n_jobs=n_cpu)),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.XGB:
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=SKLearnWrapper(module=XGBRegressor(
                n_jobs=n_cpu, n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1., sampling_method="uniform"
            )),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == "DeepAR":
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=PyTorchForecastingDeterministicWrapper(model=DeepAR, **dl_kwargs),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.NHiTS:
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=PyTorchForecastingDeterministicWrapper(model=NHiTS, **dl_kwargs),
            kwargs=forecaster_p_params)
    elif forecaster_p_name == PointForecaster.TFT:
        modules["forecaster_p"] = CombinedForecaster(
            forecaster=PyTorchForecastingDeterministicWrapper(model=TemporalFusionTransformer, **dl_kwargs),
            kwargs=forecaster_p_params)
    else:
        raise NotImplementedError(f"Point forecasting method {forecaster_p_name} is not implemented!"
                                  f"Please load one of {list(PointForecaster)}")

    return modules


def save_modules(pipeline_modules: dict, dry: str) -> None:
    """
    Save pyWATTS modules as pickle files.
    :param pipeline_modules:
        Initialized and fitted pyWATTS modules.
    :type pipeline_modules: dict
    :param dry:
        Directory of saved AutoPQ model.
    :type dry: str
    :return: None
    :rtype: None
    """
    if not os.path.exists(dry):
        os.makedirs(dry)
    for module_name, module_object in pipeline_modules.items():
        cloudpickle.dump(module_object, open(f"{dry}/{module_name}.pickle", 'wb'))


def load_modules(pipeline_modules: dict, dry: str) -> dict:
    """
    Load pyWATTS modules from pickle files.
    :param pipeline_modules:
        Initialized and fitted pyWATTS modules.
    :type pipeline_modules: dict
    :param dry:
        Directory of saved AutoPQ model.
    :type dry: str
    :return:
        Initialized and fitted pyWATTS modules.
    :rtype: dict
    """
    for module_name, module_object in pipeline_modules.items():
        with open(f"{dry}/{module_name}.pickle", 'rb') as file:
            pipeline_modules[module_name] = cloudpickle.load(file)
    return pipeline_modules


def add_metrics(y_hat: StepInformation, y: StepInformation, suffix: str) -> None:
    """
    Add metrics to the pyWATTS pipeline.
    :param y_hat:
        Prediction.
    :type y_hat: StepInformation
    :param y:
        Target.
    :type y: StepInformation
    :param suffix:
        Individual suffix for the metric name.
    :type suffix: str
    :return: None
    :rtype: None
    """

    RMSE(name=f"RMSE_{suffix}")(y_hat=y_hat, y=y)
    MAE(name=f"MAE_{suffix}")(y_hat=y_hat, y=y)


def clip_values(x: xr.DataArray, min_value: Union[float, None], max_value: Union[float, None]) -> xr.DataArray:
    """
    Clips values between min_value and max_value. If both are None, the unchanged DataArray is returned.
    :param x:
        DataArray to be clipped.
    :type x: xr.DataArray
    :param min_value:
        Lower limit of the DataArray.
    :type min_value: Union[float, None]
    :param max_value:
        Upper limit of the DataArray
    :type max_value: Union[float, None]
    :return:
        The clipped DataArray.
    :rtype: xr.DataArray
    """
    if min_value is None and max_value is None:
        return x
    return x.clip(min=min_value, max=max_value)
