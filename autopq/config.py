from enum import Enum


class PointForecaster(str, Enum):
    """
    Enum which contains the point forecasting methods for AutoPQ.
    Statistical Modeling (SM)
        ETS: Error Trend Seasonality
        TBATS: Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, Seasonal components
        sARIMAX: seasonal AutoRegressive Integrated Moving Average with eXternal input
    Machine Learning (ML)
        MLP: MultiLayer Perceptron
        SVR: Support Vector Regression
        XGB: eXtreme Gradient Boosting
    Deep Learning (DL)
        DeepAR: Deep AutoRegression
        NHiTS: Neural Hierarchical Interpolation for Time Series
        TFT: Temporal Fusion Transformer
    """
    ETS = "ETS"
    TBATS = "TBATS"
    sARIMAX = "sARIMAX"
    MLP = "MLP"
    SVR = "MLP"
    XGB = "XGB"
    DeepAR = "DeepAR"
    NHiTS = "NHiTS"
    TFT = "TFT"


class ComputingResources(str, Enum):
    """
    Enum which contains the computing resources for AutoPQ.
    Default: Standard computing systems for achieving competitive probabilistic forecasting accuracy
    Advanced: High Performance Computing (HPC) systems to further improve probabilistic forecasting accuracy, e.g., for smart
        grid applications with high decision costs
    """
    Default = "Default"
    Advanced = "Advanced"


class ValMetric(str, Enum):
    """
    Enum which contains the validation metric for the hyperparameter optimization.
    CRPS: Continuous Ranked Probability Score
    mPL: mean Pinball Loss
    MAQD: Mean Absolute Quantile Deviation
    MAE : Mean Absolute Error
    MSE: Mean Squared Error
    """
    CRPS = "CRPS"
    mPL = "mPL"
    MAQD = "MAQD"
    MAE = "MAE"
    MSE = "MSE"


class HyperoptAlgo(str, Enum):
    """
    Enum which contains the hyperopt search algorithm.
    TPE: Tree Parzen Estimator
    aTPE: adaptive Tree Parzen Estimator
    rand: random search
    """
    TPE = "TPE"
    aTPE = "aTPE"
    rand = "rand"


ConfigSpace = {
    "SVR": {
        # https://github.com/Analytics-for-Forecasting/msvr
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        "kernel": ("'rbf'", "'laplacian'", "'sigmoid'"),  # str, choice - default: 'rbf'
        "C": (1e-2, 1e2),  # float, range between - default: 1.
        "epsilon": (1e-3, 1.)  # float, range between - default: 0.1
    },
    "XGB": {
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
        # https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
        # https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
        "n_estimators": (10, 300),  # int, range between - default: 100
        "learning_rate": (1e-2, 1e0),  # float, range between - default: 0.3
        "max_depth": (1, 18),  # int, range between - default: 6
        "subsample": (0.5, 1.)  # float, range between - default: 1.
    },
    "MLP": {
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        "batch_size": ("32", "64", "128"),
        "activation": ("'logistic'", "'tanh'", "'relu'"),  # str, choice - default: 'relu'
        # random probabilities: 1 hidden layer = 0.25, 2 hidden layers = 0.5, 3 hidden layers = 0.25
        "n_neurons_1": (10, 100),  # int, range between - default: 100, one hidden layer
        "n_neurons_2": (-80, 100),  # int, range between  -- neurons < 10 => no hidden layer 2
        "n_neurons_3": (-80, 100),  # int, range between  -- neurons < 10 => no hidden layer 3
    },
    "NHiTS": {
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nhits.NHiTS.html
        "batch_size": ("32", "64", "128"),
        "hidden_size": (8, 1024),  # int, range between - default: 512
        "shared_weights": ("True", "False"),  # bool, choice - default: True
        "n_layers": (1, 3),  # int, range between - default: 2
        "dropout": (0.0, 0.2),  # float, range between - default: 0.0
    },
    "DeepAR": {
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.deepar.DeepAR.html
        # https://docs.aws.amazon.com/sagemaker/latest/dg/deepar-tuning.html
        "batch_size": ("32", "64", "128"),
        "cell_type": ("'LSTM'", "'GRU'"),  # str, choice - default: 'LSTM'
        "hidden_size": (10, 100),  # int, range between - default: 10
        "rnn_layers": (1, 3),  # int, range between - default: 2
        "dropout": (0.0, 0.2),  # float, range between - default: 0.1
    },
    "TFT": {
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters.html#pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters
        "batch_size": ("32", "64", "128"),
        "hidden_size": (8, 256),  # int, range between - default: 16
        "lstm_layers": (1, 3),  # int, range between - default: 1
        "attention_head_size": (1, 4),  # int, range between - default: 4
        "dropout": (0.0, 0.2),  # float, range between - default: 0.1
    },
    "SARIMAX": {
        # https://www.sktime.net/en/v0.20.0/api_reference/auto_generated/sktime.forecasting.arima.AutoARIMA.html
        "p": (0, 5),  # int, range between
        "d": (0, 2),  # int, range between
        "q": (0, 5),  # int, range between
        "P": (0, 2),  # int, range between
        "D": (0, 1),  # int, range between
        "Q": (0, 2),  # int, range between
    },
    "ETS": {  # 30 possible combinations
        # https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.ets.AutoETS.html
        "error": ("'add'", "'mul'"),  # str, choice
        "trend": ("'add'", "'mul'", "None"),  # str, choice
        "seasonal": ("'add'", "'mul'", "None"),  # str, choice
        "damped_trend": ("True", "False"),  # bool, choice
    },
    "TBATS": {  # 32 possible combinations
        # https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.bats.BATS.html
        # https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.tbats.TBATS.html
        "trigonometric_seasonality": ("True", "False"),  # bool, choice
        "use_box_cox": ("True", "False"),  # bool, choice
        "use_trend": ("True", "False"),  # bool, choice
        "use_damped_trend": ("True", "False"),  # bool, choice
        "use_arma_errors": ("True", "False"),  # bool, choice
    },
    "cINN": {
        "sampling_std": (0.01, 3.0),  # float, range between
    }
}
