# Loading the package

import pandas as pd
from hyperts.utils import metrics
import time
import os
import yaml
import sys

f = open("config.yaml", 'r', encoding='utf-8')
config = yaml.load(f.read(), Loader=yaml.FullLoader)
if len(sys.argv) > 1:
    result_file_path = config['report_path'] + os.sep + sys.argv[1] + '.csv'
else:
    result_file_path = config['report_path'] + os.sep + 'benchmark_auts' + time.strftime("%Y%m%d_%H%M%S",
                                                                                         time.localtime()) + '.csv'
base_path = config['data_path']
mode = config['env']
max_trials = config['max_trials']
f.close()

trained_data_names = []
if os.path.exists(result_file_path):
    trained_data_names = pd.read_csv(result_file_path)['dataset'].values

types = ['univariate-forecast', 'multivariate-forecast']
data_sizes = ['small', 'medium', 'large']


def autotstest(train_df, test_df, Date_Col_Name, metric):
    forecast_length = test_df.shape[0]
    time1_start = time.time()
    from autots import AutoTS
    train_df = train_df.copy(deep=True)
    train_df.index = pd.DatetimeIndex(train_df[Date_Col_Name])
    metric_weighting = {'rmse_weighting': 1}
    model = AutoTS(forecast_length=forecast_length,
                   drop_data_older_than_periods=366, metric_weighting=metric_weighting,
                   max_generations=1,
                   min_allowed_train_percent=0.01)  # TODO max_generations 去掉  drop_data_older_than_periods 这个值给多少？

    train_df.drop(Date_Col_Name, 1, inplace=True)
    model = model.fit(train_df)

    prediction = model.predict()
    forecast = prediction.forecast
    time1_end = time.time()
    return forecast, (time1_end - time1_start), "{}"


def hpyertstest(train_df, test_df, Date_Col_Name, format, task, covariables, metric):
    """

    Parameters
    ----------
    train_df: pandas DataFrame
        Feature data for training with target column.
    test_df: pandas DataFrame
        Feature data for testing with target column, should be None or have the same python type with 'train_data'.
    Date_Col_Name: str
        Forecast task Date_Col_Name cannot be None, (default=None).
    format: str
        The date format of timestamp col for forecast task, (default='%Y-%m-%d %H:%M:%S').
    task: str
        Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass', etc.
    Returns
    -------
    (y_pred, time_cost)

    y_pred: panda.DataFrame.
        The test_df with the predict result.
    time_cost: int.
        The time train cost.
    """
    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts
    train_df = train_df.copy(deep=True)

    exp = make_experiment(train_df,
                          timestamp=Date_Col_Name,
                          task=task,
                          reward_metric=metric,
                          timestamp_format=format,
                          covariables=covariables,
                          max_trials=max_trials
                          )

    model = exp.run()
    y_pred = model.predict(test_df)
    time2_end = time.time()
    return y_pred, (time2_end - time2_start), exp.run_kwargs


def score_calc(y_true, y_pred1, metric):
    return metrics.calc_score(y_true, y_pred1, metrics=[metric])[metric]


def trail(TRAIN_APTH, TEST_PATH, Date_Col_Name, Series_Col_name, forecast_length, format, task, metric, covariables):
    """
    Parameters
    ----------
    TRAIN_APTH: str.
        The path of Feature data for training.
    TEST_PATH: str.
        The path of Feature data for evaluationing.
    Date_Col_Name: str.
        Forecast task Date_Col_Name cannot be None, (default=None).
    Series_Col_name: str. (like='Close,Open').
        The Series columns name.
    forecast_length: int.
        The length of data in TEST_PATH.
    format: str, the date format of timestamp col for forecast task, (default='%Y-%m-%d %H:%M:%S').
    task: str,
        Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass', etc.
        See consts.py for details.
    metric: str.
        The metric to focus.
    Returns
    -------
    (metric, time_cost)

    metric: float.
        The value of evaluation metric, like rmse.
    time_cost: int.
        The time train cost.
    """
    # load data
    df_train = pd.read_csv(TRAIN_APTH)
    df_test = pd.read_csv(TEST_PATH)
    df_train[Date_Col_Name] = pd.to_datetime(df_train[Date_Col_Name], format=format)
    df_test[Date_Col_Name] = pd.to_datetime(df_test[Date_Col_Name], format=format)

    if Series_Col_name != None:
        df_train[Series_Col_name] = df_train[Series_Col_name].astype(float)
        df_test[Series_Col_name] = df_test[Series_Col_name].astype(float)

        Series_Col_name.append(Date_Col_Name)
        # split data
        df_train = df_train[Series_Col_name]
        df_test = df_test[Series_Col_name]

    else:
        for col in df_train.columns:
            if col == Date_Col_Name:
                continue
            df_train[col] = df_train[col].astype(float)
            df_test[col] = df_test[col].astype(float)
        # split data

    # TEST FOR HyperTS
    y_pred, time_cost, run_kwargs = autotstest(df_train, df_test, Date_Col_Name, metric)

    # Metrics
    return score_calc(df_test.drop(Date_Col_Name, 1), y_pred, metric), time_cost, run_kwargs


def save_metrics(dataset, metric, hypertsmetric, hypertscost, shape, type, task, forecast_len, run_kwargs):
    import pandas as pd
    try:
        metrics_df = pd.read_csv(result_file_path)
    except:
        print(result_file_path + ' is null')
        metrics_df = pd.DataFrame(
            columns=['dataset', 'shape', 'type', 'task', 'forecast_len', 'metric', 'AutoTS_SCORE', 'AutoTS_duration[s]',
                     'run_kwargs'])

    data = {'dataset': dataset, 'shape': shape, 'task': type, 'forecast_len': forecast_len, 'metric': metric,
            'AutoTS_SCORE': round(hypertsmetric, 6),
            'AutoTS_duration[s]': round(hypertscost, 1),
            'run_kwargs': run_kwargs}
    metrics_df = metrics_df.append(data, ignore_index=True)
    metrics_df.to_csv(result_file_path, index=False)


time_start = time.time()
print("start", time.strftime("%Y-%m-%d %H:%M:%S"))

for type in types:
    for data_size in data_sizes:
        path = base_path + os.sep + type + os.sep + data_size
        if os.path.exists(path):
            list = os.listdir(path)
            for dir in list:
                if (dir == '__init__.py'):
                    continue
                train_file_path = path + os.sep + dir + os.sep + 'train.csv'
                if mode == 'dev':
                    train_file_path = path + os.sep + dir + os.sep + 'train_dev.csv'
                test_file_path = path + os.sep + dir + os.sep + 'test.csv'
                metadata_path = path + os.sep + dir + os.sep + 'metadata.yaml'

                if os.path.exists(train_file_path) and os.path.getsize(train_file_path) > 0:
                    print("train_file_path: ", train_file_path)
                    print("test_file_path: ", test_file_path)
                    print("metadata_path: ", metadata_path)
                    f = open(metadata_path, 'r', encoding='utf-8')
                    config = yaml.load(f.read(), Loader=yaml.FullLoader)
                    forecast_len = config['forecast_len']
                    type = config['type']
                    dtformat = config['dtformat']
                    data_name = config['name']
                    metric = config['metric']
                    shape = config['shape']
                    task = config['type']
                    forecast_len = config['forecast_len']
                    series_col_name = config['series_col_name'].split(",") if 'series_col_name' in config else None
                    covariables = config['covariables_col_name'].split(",") if 'covariables_col_name' in config else None
                    f.close()

                    if data_name in trained_data_names:
                        print('==skipped== already trained ', data_name)
                        continue

                    # try:
                    hypertsmetric, hypertscost, run_kwargs = trail(train_file_path, test_file_path,
                                                                   config['date_col_name'],
                                                                   series_col_name, forecast_len, dtformat, type,
                                                                   metric,
                                                                   covariables)
                    save_metrics(data_name, metric, hypertsmetric, hypertscost, shape, type, task, forecast_len,
                                 run_kwargs)
                    # except Exception as e:
                    #     print(e)
                    #     print(" Error: " + train_file_path)

time_end = time.time()

print("end  ", time.strftime("%Y-%m-%d %H:%M:%S"))
print("total cost", time_end - time_start)
