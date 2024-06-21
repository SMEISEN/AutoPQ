from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
import properscoring
from pywatts.core.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss, mean_absolute_error, mean_squared_error

from autopq.config import ValMetric, PointForecaster


def map_forecaster_p_params(forecaster_name: PointForecaster, params: dict) -> dict:
    """
    Map hyperparameter configuration of the point forecasting method. Ensures that the parameter names match the
    implementation of the point forecasting method.
    :param forecaster_name:
        Name of the point forecasting method.
    :type forecaster_name: PointForecaster
    :param params:
        Hyperparameters of the point forecasting method.
    :type params: dict
    :return:
        Mapped hyperparameter configuration of the point forecasting method.
    :rtype: dict
    """

    # Default point forecaster hyperparameters
    if len(params) == 0:
        return params

    # Map choices (str) to str, int, or float
    for key, value in params.items():
        if isinstance(value, str):
            try:
                params[key] = eval(value)
            except NameError:
                pass

    if forecaster_name == PointForecaster.MLP:
        # Map hidden layer sizes, deactivates a layer if n_neurons < 10
        hidden_layer_sizes = [params.pop("n_neurons_1"), params.pop("n_neurons_2"), params.pop("n_neurons_3")]
        hidden_layer_sizes = [n_neurons for n_neurons in hidden_layer_sizes if n_neurons >= 10]
        if len(hidden_layer_sizes) == 0:
            # Minimal number of parameters, one hidden layer with 10 neurons
            hidden_layer_sizes = 10
        params["hidden_layer_sizes"] = hidden_layer_sizes
    elif forecaster_name == PointForecaster.sARIMAX:
        # Map order and seasonal order
        params["order"] = (params.pop("p"), params.pop("d"), params.pop("q"))
        params["seasonal_order"] = (params.pop("P"), params.pop("D"), params.pop("Q"), 24)
    elif forecaster_name == PointForecaster.ETS:
        params["auto"] = False  # Deactivate internal automated search
        if params["trend"] is None:  # Damped trend is only possible if "trend" is not None
            params["damped_trend"] = False

    return params


def _calculate_abs_empirical_quantile_diff(quant: int, y: xr.DataArray, y_hat: xr.DataArray) -> float:
    """
    Calculates the absolute empirical quantile difference.
    :param quant:
        Empirical quantile.
    :type quant: int
    :param y:
        Ground truth.
    :type y: xr.DataArray
    :param y_hat:
        Prediction.
    :return: absolute empirical quantile difference
    :rtype: float
    """
    half = np.less_equal(np.squeeze(y), y_hat.sel(quantiles=quant))
    full = (np.mean(half.sum(axis=1) / half.shape[-1])).values.mean()
    qq = np.round(quant / 100, 2)
    return float(np.abs(full - qq))


def CRPS(y: xr.DataArray, y_hat: xr.DataArray) -> float:
    """
    Calculates the Continuous Ranked Probability Score (CRPS).
    :param y:
        Ground truth.
    :type y: xr.DataArray
    :param y_hat:
        Prediction.
    :type y_hat: xr.DataArray
    :return: CRPS
    :rtype: float
    """
    try:
        crps = np.mean(properscoring.crps_ensemble(np.squeeze(y), y_hat))
    except Exception:
        crps = np.inf
    return float(crps)


def mPL(y: xr.DataArray, y_hat: xr.DataArray) -> float:
    """
    Calculates the mean Pinball Loss (mPL)
    :param y:
        Ground truth.
    :type y: xr.DataArray
    :param y_hat:
        Prediction.
    :type y_hat: xr.DataArray
    :return: mPL
    :rtype: float
    """
    try:
        pl = []
        for quant in y_hat.quantiles:
            pl.append(mean_pinball_loss(np.squeeze(y), y_hat.loc[:, :, quant.values], alpha=quant.values / 100))
        pl = np.mean(pl)
    except Exception:
        pl = np.inf
    return float(pl)


def MAE(y: xr.DataArray, y_hat: xr.DataArray) -> float:
    """
    Calculates the Mean Absolute Error (MAE).
    :param y:
        Ground truth.
    :type y: xr.DataArray
    :param y_hat:
        Prediction.
    :type y_hat: xr.DataArray
    :return: MAE
    :rtype: float
    """
    if hasattr(y_hat, "quantiles"):
        y_hat = y_hat.sel(quantiles=[50])[:, :, 0]
    try:
        mae = mean_absolute_error(y_true=np.squeeze(y), y_pred=np.squeeze(y_hat))
    except Exception:
        mae = np.inf
    return mae


def MSE(y: xr.DataArray, y_hat: xr.DataArray) -> float:
    """
    Calculates the Mean Squared Error (MSE).
    :param y:
        Ground truth.
    :type y: xr.DataArray
    :param y_hat:
        Prediction.
    :type y_hat: xr.DataArray
    :return: MSE
    :rtype: float
    """
    if hasattr(y_hat, "quantiles"):
        y_hat = y_hat.sel(quantiles=[50])[:, :, 0]
    try:
        mse = mean_squared_error(y_true=np.squeeze(y), y_pred=np.squeeze(y_hat))
    except Exception:
        mse = np.inf
    return float(mse)


def MAQD(y: xr.DataArray, y_hat: xr.DataArray) -> float:
    """
    Calculates the Mean Absolute Quantile Deviation (MAQD).
    :param y:
        Ground truth.
    :type y: xr.DataArray
    :param y_hat:
        Prediction.
    :type y_hat: xr.DataArray
    :return: MAQD
    :rtype: float
    """
    required_quants = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]
    q_list = y_hat.quantiles
    q_list = q_list.values
    if all(quant in q_list for quant in required_quants):
        try:
            maqd = np.mean([_calculate_abs_empirical_quantile_diff(50, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(1, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(99, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(5, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(95, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(15, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(85, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(25, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(75, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(10, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(90, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(20, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(80, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(30, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(70, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(40, y, y_hat),
                            _calculate_abs_empirical_quantile_diff(60, y, y_hat)])
        except Exception:
            maqd = np.inf
    else:
        raise NotImplementedError("The quantiles required to calculate the MAQD are not included.")
    return float(maqd)


def evaluate_trial(val_metric: ValMetric, y: xr.DataArray, y_hat: xr.DataArray) -> float:
    """
    Evaluate the performance.
    :param val_metric:
        Metric used to assess the estimator's performance.
    :type val_metric: ValMetric
    :param y:
        Ground truth.
    :type y: xr.DataArray
    :param y_hat:
        Prediction.
    :type y_hat: xr.DataArray
    :return:
        Validation score of the assessed estimator.
    :rtype: float
    """
    if val_metric == ValMetric.CRPS:
        return CRPS(y=y, y_hat=y_hat)
    elif val_metric == ValMetric.mPL:
        return mPL(y=y, y_hat=y_hat)
    elif val_metric == ValMetric.MAQD:
        return MAQD(y=y, y_hat=y_hat)
    elif val_metric == ValMetric.MAE:
        return MAE(y=y, y_hat=y_hat)
    elif val_metric == ValMetric.MSE:
        return MSE(y=y, y_hat=y_hat)
    else:
        raise NotImplementedError(f"Validation metric {val_metric} is not implemented!")


def assess_configuration(trial, estimator: Pipeline, data: pd.DataFrame, val_metric: ValMetric,
                         cv: Union[int, None] = None) -> float:
    """
    Assess the sampling standard deviation on a fitted estimator on the given data using the given validation metric.
    :param trial:
        Hyperparameter configuration to be assessed.
    :type trial: dict
    :param estimator:
        Trained pyWATTS pipeline.
    :type estimator: Pipeline
    :param data:
        Data used to assess the estimator's performance.
    :type data: pd.DataFrame
    :param val_metric:
        Metric used to assess the estimator's performance.
    :type val_metric: ValMetric
    :param cv:
        Numer of cross-validation fold (optional, default=None).
    :type cv: Union[int, None]
    :return:
        Validation score of the assessed estimator.
    :rtype: float
    """
    for obj in reversed(estimator.id_to_step.values()):
        if obj.name == "ProbForecastCINN":
            obj.module.sampling_std = trial["sampling_std"]
            break
    if cv is not None:
        loss = []
        for _, val_index in KFold(n_splits=cv).split(data):
            result = estimator.test(data=data.iloc[val_index], summary=False, summary_formatter=None)
            loss.append(evaluate_trial(val_metric=val_metric, y=result["target_scale"], y_hat=result["forecast_q_scale"]))
        return np.mean(loss)
    result = estimator.test(data=data, summary=False, summary_formatter=None)
    return float(evaluate_trial(val_metric=val_metric, y=result["target_scale"], y_hat=result["forecast_q_scale"]))
