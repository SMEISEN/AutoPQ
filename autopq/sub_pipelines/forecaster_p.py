from functools import partial
from typing import Tuple, Union
from pywatts.core.pipeline import Pipeline
from pywatts.core.step_information import StepInformation
from pywatts.modules import FunctionModule

from autopq.sub_pipelines.utils import clip_values


def add_forecaster_p(pipeline: Pipeline, modules: dict,
                     forecast_limits: Tuple[Union[float, None], Union[float, None]]) -> Pipeline:
    """
    Adds the point forecaster to the pipeline and clips physical limits.
    :param pipeline:
        Initialized pyWATTS pipeline object.
    :type pipeline: Pipeline
    :param modules:
        Initialized pyWATTS modules.
    :type modules: dict
    :param forecast_limits:
        Physical limitations of the target time series.
    :type forecast_limits: Tuple[Union[float, None], Union[float, None]]
    :return:
        Modified pyWATTS pipeline object.
    :rtype: Pipeline
    """

    # Get steps uses as input in the following
    sm_inputs = {}
    ml_inputs = {}
    dl_inputs = {}
    for step in pipeline.id_to_step.values():
        if "_SM_" in step.name:
            sm_inputs.update({step.name.replace("target", "t"): StepInformation(step, pipeline)})
            if "target" in step.name:
                sm_inputs.update({step.name.replace("target", "u"): StepInformation(step, pipeline)})
        elif "_ML_" in step.name:
            ml_inputs.update({step.name.replace("target", "t"): StepInformation(step, pipeline)})
        elif "_DL_" in step.name:
            dl_inputs.update({step.name.replace("target", "t"): StepInformation(step, pipeline)})

    # Add point forecaster to the pipeline
    forecast_p_scale = modules["forecaster_p"](**sm_inputs,  # SM inputs
                                               **ml_inputs,  # ML inputs
                                               **dl_inputs  # DL inputs
                                               )

    # Inverse transform to initial scale
    forecast_p = FunctionModule(
        lambda x: (x * modules["target_scaler"].module.scale_) + modules["target_scaler"].module.mean_)(x=forecast_p_scale)

    # Clip physical limits
    forecast_p = FunctionModule(partial(clip_values, min_value=forecast_limits[0], max_value=forecast_limits[1]),
                                name="forecast_p")(x=forecast_p)

    # Transform to normalized scale
    FunctionModule(
        lambda x: (x - modules["target_scaler"].module.mean_) / modules["target_scaler"].module.scale_,
        name="forecast_p_scale")(x=forecast_p)

    # Make outputs available in the results
    forecast_p.step.last = True

    return pipeline
