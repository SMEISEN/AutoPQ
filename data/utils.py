import os
import pandas as pd
from typing import List

VAL_PERCENT = 0.3
TEST_PERCENT = 0.2
DATA_NAMES = ["Load-BW", "Load-GCP", "Mobility", "Price", "PV", "WP"]


def _split_data(data, val_percent: float, test_percent: float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split the (not shuffled) data into a training and a test data sub-set with a fixed
    :param data:
        Data to be splitted.
    :type data: pd.DataFrame
    :param val_percent:
        Percentage of samples from the training data sub-set considered for the validation data sub-set.
    :type val_percent: float
    :param test_percent:
        Percentage of samples considered for the test data sub-set.
    :type test_percent: float
    :return:
        data, the complete data set
        train, the training data sub-set
        val, the validation data sub-set
        test, the test data sub-set
    :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    split_index = int(len(data) - (len(data) * test_percent))
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    split_index = int(len(train) - (len(train) * val_percent))
    val = train[split_index:]
    train = train[:split_index]

    return data, train, val, test


def _prepare_load_bw() -> ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str]):
    """
    Prepare the Load-Baden-Württemberg (BW) data set.
    :return:
        data_splits, the complete data set, as well as the training, validation, and test data sub-sets
        target_name, column name of the target time series
        feature_names, column names of the feature time series
    :rtype: ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str])
    """
    data = pd.read_csv(f"{os.getcwd()}/data/Load-BW.csv",
                       index_col="time",
                       parse_dates=["time"],
                       infer_datetime_format=True,
                       sep=",")
    data.index.name = 'time'
    data = data.interpolate()

    data_splits = _split_data(data=data, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT)
    target_name = "load_power_statistics"
    feature_names = []

    return data_splits, target_name, feature_names


def _prepare_load_gcp() -> ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str]):
    """
    Prepare the Load-Grid Connection Point (GCP) data set.
    :return:
        data_splits, the complete data set, as well as the training, validation, and test data sub-sets
        target_name, column name of the target time series
        feature_names, column names of the feature time series
    :rtype: ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str])
    """
    data = pd.read_csv(f"{os.getcwd()}/data/Load-GCP.csv",
                       index_col="time",
                       parse_dates=True)
    data.index.name = 'time'

    data_splits = _split_data(data=data[:26280], val_percent=VAL_PERCENT, test_percent=TEST_PERCENT)
    target_name = "MT_158"
    feature_names = []

    return data_splits, target_name, feature_names


def _prepare_mobility() -> ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str]):
    """
    Prepare the Mobility data set.
    :return:
        data_splits, the complete data set, as well as the training, validation, and test data sub-sets
        target_name, column name of the target time series
        feature_names, column names of the feature time series
    :rtype: ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str])
    """
    data = pd.read_csv(f"{os.getcwd()}/data/Mobility.csv",
                       index_col="time",
                       parse_dates=True)
    data.index.name = "time"

    data_splits = _split_data(data=data, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT)
    target_name = "cnt"
    feature_names = ["temp", "hum", "windspeed", "weathersit"]

    return data_splits, target_name, feature_names


def _prepare_price() -> ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str]):
    """
    Prepare the Price data set.
    :return:
        data_splits, the complete data set, as well as the training, validation, and test data sub-sets
        target_name, column name of the target time series
        feature_names, column names of the feature time series
    :rtype: ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str])
    """
    data = pd.read_csv(f"{os.getcwd()}/data/Price.csv")
    data.index = pd.date_range(start="1/1/2011 0:00:00", freq="1H", periods=len(data))
    data.index.name = 'time'
    data = data.drop(columns=["Unnamed: 0", "ZONEID", "timestamp"])

    data_splits = _split_data(data=data, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT)
    target_name = "Zonal Price"
    feature_names = ["Forecasted Total Load", "Forecasted Zonal Load"]

    return data_splits, target_name, feature_names


def _prepare_pv() -> ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str]):
    """
    Prepare the PhotovVoltaic (PV) data set.
    :return:
        data_splits, the complete data set, as well as the training, validation, and test data sub-sets
        target_name, column name of the target time series
        feature_names, column names of the feature time series
    :rtype: ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str])
    """
    data = pd.read_csv(f"{os.getcwd()}/data/PV.csv")
    data.index = pd.date_range(start="4/1/2012 01:00:00", freq="1H", periods=len(data))
    data.index.name = 'time'
    data = data.drop(columns=["Unnamed: 0"])

    data_splits = _split_data(data=data, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT)
    target_name = "POWER"
    feature_names = ["SSRD", "TCC"]

    return data_splits, target_name, feature_names


def _prepare_wp() -> ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str]):
    """
    Prepare the Wind Power (WP) data set.
    :return:
        data_splits, the complete data set, as well as the training, validation, and test data sub-sets
        target_name, column name of the target time series
        feature_names, column names of the feature time series
    :rtype: ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str])
    """
    data = pd.read_csv(f"{os.getcwd()}/data/WP.csv")
    data = data.dropna()
    data.index = pd.date_range(start="1/1/2012 01:00:00", freq="1H", periods=len(data))
    data.index.name = 'time'
    data = data.drop(columns=["Unnamed: 0"])

    data_splits = _split_data(data=data, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT)
    target_name = "TARGETVAR"
    feature_names = ["U100", "V100", "Speed100"]

    return data_splits, target_name, feature_names


def load_example_data(data_name: str) -> ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str]):
    """
    Load the example data.
    :param data_name:
        Name of the example data set.
    :type data_name: str
    :return:
        data_splits, the complete data set, as well as the training, validation, and test data sub-sets
        target_name, column name of the target time series
        feature_names, column names of the feature time series
    :rtype: ((pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame), str, List[str])
    """

    if data_name == "Load-BW":
        return _prepare_load_bw()
    elif data_name == "Load-GCP":
        return _prepare_load_gcp()
    elif data_name == "Mobility":
        return _prepare_mobility()
    elif data_name == "Price":
        return _prepare_price()
    elif data_name == "PV":
        return _prepare_pv()
    elif data_name == "WP":
        return _prepare_wp()
    else:
        raise NotImplementedError(f"Data {data_name} is not implemented!"
                                  f"Please load one of {DATA_NAMES}")
