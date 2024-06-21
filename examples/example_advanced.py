from autopq.config import PointForecaster, ComputingResources
from data.utils import load_example_data
from autopq.autopq import AutoPQ

if __name__ == "__main__":
    # AutoPQ-advanced is based on mpi4py using <n_parallel> workers, run via terminal:
    # PYTHONPATH=./ mpirun -n <n_parallel> python examples/example_advanced.py

    # Load the example data.
    (_, train, val, test), target_name, feature_names = load_example_data(data_name="Mobility")

    # Initialize AutoPQ-advanced.
    autopq = AutoPQ(target_name=target_name, feature_names=feature_names, forecaster_p_name=PointForecaster.MLP,
                    computing_resources=ComputingResources.Advanced, forecast_limits=(0., None))

    # Train the AutoPQ-advanced model.
    result_fit = autopq.fit(train=train, val=val)

    # Predict with the AutoPQ-advanced model.
    result_predict = autopq.predict(data=test)
