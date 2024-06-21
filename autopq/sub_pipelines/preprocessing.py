from typing import List
from pywatts.core.pipeline import Pipeline
from pywatts.modules import Sampler, Slicer, ClockShift


def add_preprocessing(pipeline: Pipeline, modules: dict,
                      target_name: str, feature_names: List[str], horizon: int) -> Pipeline:
    """
    Adds pre-processing steps to the pipeline.
    :param pipeline:
        Initialized pyWATTS pipeline object.
    :type pipeline: Pipeline
    :param modules:
        Initialized pyWATTS modules.
    :type modules: dict
    :param target_name:
        Column name of the target in the dataframe given in train and predict.
    :type target_name: str
    :param feature_names:
        Column names of the features in the dataframe given in train and predict.
    :type feature_names: List[str]
    :param horizon:
        Forecasting horizon in number of time stamps.
    :type horizon: int
    :return:
        Modified pyWATTS pipeline object.
    :rtype: Pipeline
    """

    # Transform target to normalized scale
    target = modules["target_scaler"](x=pipeline[target_name])

    # Transform features to normalized scale
    features = {}
    for feature_name in feature_names:
        features[f"feature_SM_{feature_name}"] = modules[f"feature_scaler_{feature_name}"](
            x=pipeline[feature_name])

    # Extract (already normalized) calendar features
    calendar = modules["calendar_extractor"](x=pipeline[target_name])

    # Sample and slice target
    target_ML = Sampler(horizon)(x=target)
    Slicer(start=2 * horizon, end=-2 * horizon, name=f"target_ML_{target_name}")(x=target_ML)
    target_DL = ClockShift(lag=horizon)(x=target)
    Slicer(start=horizon, end=-horizon, name=f"target_DL_{target_name}")(x=target_DL)

    # Sample and slice features
    for feature_name in feature_names:
        feature_ML = Sampler(horizon)(
            x=features[f"feature_SM_{feature_name}"])
        Slicer(start=2 * horizon, end=-2 * horizon, name=f"feature_ML_{feature_name}")(x=feature_ML)
        feature_DL = ClockShift(lag=horizon)(x=features[f"feature_SM_{feature_name}"])
        Slicer(start=horizon, end=-horizon, name=f"feature_DL_{feature_name}")(x=feature_DL)

    # Sample and slice calendar values
    calendar_ML = Sampler(horizon)(x=calendar)
    Slicer(start=2 * horizon, end=-2 * horizon, name=f"calendar_ML_{target_name}")(x=calendar_ML)
    calendar_DL = ClockShift(lag=horizon)(x=calendar)
    Slicer(start=horizon, end=-horizon, name=f"calendar_DL_{target_name}")(x=calendar_DL)

    # Create historical values, sample and slice
    history_ML = Sampler(horizon)(x=ClockShift(lag=horizon + 1)(x=target))
    Slicer(start=2 * horizon, end=-2 * horizon, name=f"history_ML_{target_name}")(x=history_ML)

    # Make outputs available in the results
    Slicer(start=2 * horizon, end=-2 * horizon, name=f"target")(x=Sampler(horizon)(x=pipeline[target_name]))
    Slicer(start=2 * horizon, end=-2 * horizon, name="target_scale")(x=Sampler(horizon)(x=target))

    return pipeline
