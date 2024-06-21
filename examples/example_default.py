from autopq.config import PointForecaster, ComputingResources
from data.utils import load_example_data
from autopq.autopq import AutoPQ


if __name__ == "__main__":
    # AutoPQ-default is not based on mpi4py, run via terminal:
    # PYTHONPATH=./ python examples/example_default.py

    # Load the example data.
    (_, train, val, test), target_name, feature_names = load_example_data(data_name="Mobility")

    # Initialize AutoPQ-default.
    autopq = AutoPQ(target_name=target_name, feature_names=feature_names, forecaster_p_name=PointForecaster.MLP,
                    computing_resources=ComputingResources.Default, forecast_limits=(0., None))

    # Train the AutoPQ-default model.
    result_fit = autopq.fit(train=train, val=val)

    # Predict with the AutoPQ-default model.
    result_predict = autopq.predict(data=test)
