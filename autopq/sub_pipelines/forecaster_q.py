from functools import partial
from typing import Union, Tuple

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.core.step_information import StepInformation
from pywatts.modules import FunctionModule

from autopq.sub_pipelines.utils import clip_values


def add_forecaster_q(pipeline: Pipeline, modules: dict,
                     forecast_limits: Tuple[Union[float, None], Union[float, None]]) -> Pipeline:
    """
    Adds the quantile forecaster to the pipeline and clips physical limits.
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
    inputs = {}
    for step in pipeline.id_to_step.values():
        if "_ML_" in step.name:
            if "target" in step.name:
                # Add target
                inputs.update({"target": StepInformation(step, pipeline)})
            else:
                # Add context
                inputs.update({step.name: StepInformation(step, pipeline)})
        elif step.name == "forecast_p_scale":
            # Add point forecast
            inputs.update({"forecast": StepInformation(step, pipeline)})

    # Add quantile forecaster to the pipeline
    computation_mode = ComputationMode.Transform if modules["forecaster_q"].is_fitted else ComputationMode.Default
    forecast_q_scale = modules["forecaster_q"](**inputs, computation_mode=computation_mode)

    # Inverse transform to initial scale
    forecast_q = FunctionModule(
        lambda x: (x * modules["target_scaler"].module.scale_) + modules["target_scaler"].module.mean_)(x=forecast_q_scale)

    # Clip physical limits
    forecast_q = FunctionModule(partial(clip_values, min_value=forecast_limits[0], max_value=forecast_limits[1]),
                                name="forecast_q")(x=forecast_q)

    # Transform to normalized scale
    FunctionModule(
        lambda x: (x - modules["target_scaler"].module.mean_) / modules["target_scaler"].module.scale_,
        name="forecast_q_scale")(x=forecast_q)

    # Make outputs available in the results
    forecast_q.step.last = True
    inputs["forecast"].step.last = True

    return pipeline
