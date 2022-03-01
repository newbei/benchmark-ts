# Loading the package

import pandas as pd
import hyperts
from hyperts.utils import metrics
import time
import os
import yaml
import sys
import traceback
from hypernets import hyperctl
from sktime.datatypes._panel._convert import from_2d_array_to_nested, is_nested_dataframe

try:
    params = hyperctl.get_job_params()
    data_base_path = params['data_path']
    report_base_path = params['report_path']
    mode = params['env']
    max_trials = params['max_trials']
except:
    traceback.print_exc()
    vers = hyperts.__version__
    f = open("../config.yaml", 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    data_base_path = config['data_path']
    report_base_path = config['report_path']
    mode = config['env']
    max_trials = config['max_trials']
    f.close()

trained_data_names = []

if len(sys.argv) > 1:
    result_file_path = report_base_path + os.sep + sys.argv[1] + '.csv'
else:
    result_file_path = report_base_path + os.sep + 'benchmark_hyperts_forecast_dl_' + time.strftime("%Y%m%d_%H%M%S",
                                                                                                    time.localtime()) + '.csv'
if os.path.exists(result_file_path):
    trained_data_names = pd.read_csv(result_file_path)['dataset'].values

types = ['classification']
data_sizes = ['small', 'medium', 'large']


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
                          max_trials=max_trials,
                          target='y',
                          optimize_direction='max'
                          )

    model = exp.run()
    X_test, y_test = model.split_X_y(test_df.copy())
    y_pred = model.predict(X_test)
    time2_end = time.time()
    return y_pred, (time2_end - time2_start), exp.run_kwargs


def score_calc(y_true, y_pred1, metric):
    return metrics.calc_score(y_true, y_pred1, metrics=[metric])[metric]


def trail(TRAIN_APTH, TEST_PATH, Date_Col_Name, Series_Col_name, forecast_length, format, task, metric, covariables):
    # load data
    df_train = pd.read_csv(TRAIN_APTH)
    df_test = pd.read_csv(TEST_PATH)

    df_test, run_kwargs, time_cost, y_pred = trail_forecast(Date_Col_Name, Series_Col_name, covariables, df_test,
                                                            df_train, format, metric, task)
    # Metrics
    return score_calc(df_test.drop(Date_Col_Name, 1), y_pred.drop(Date_Col_Name, 1), metric), time_cost, run_kwargs


def trail_forecast(Date_Col_Name, Series_Col_name, covariables, df_test, df_train, format, metric, task):
    if Series_Col_name != None and covariables != None:
        Series_Col_name = Series_Col_name + covariables
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

    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts
    train_df = df_train.copy(deep=True)

    exp = make_experiment(train_df,
                          mode='dl',
                          timestamp=Date_Col_Name,
                          task='forecast',
                          reward_metric=metric,
                          timestamp_format=format,
                          covariables=covariables,
                          max_trials=max_trials,
                          optimize_direction='min'
                          )

    # exp = make_experiment(train_df.copy(),
    #                       mode='dl',
    #                       timestamp=timestamp,
    #                       covariables=covariables,
    #                       task=task,
    #                       callbacks=None,
    #                       reward_metric=reward_metric,
    #                       optimize_direction=optimize_direction)

    model = exp.run()
    X_test, y_test = model.split_X_y(df_test.copy())
    y_pred = model.predict(X_test)
    time2_end = time.time()
    # return y_pred, (time2_end - time2_start), exp.run_kwargs
    # y_pred, time_cost, run_kwargs = hpyertstest(df_train, df_test, Date_Col_Name, format, task, covariables, metric)
    return df_test, exp.run_kwargs, (time2_end - time2_start), y_pred


def save_metrics(dataset, metric, hypertsmetric, hypertscost, shape, data_size, task, forecast_len, run_kwargs):
    import pandas as pd
    try:
        metrics_df = pd.read_csv(result_file_path)
    except:
        print(result_file_path + ' is null')
        metrics_df = pd.DataFrame(
            columns=['dataset', 'shape', 'data_size', 'task', 'forecast_len', 'metric', 'HyperTS_SCORE',
                     'HyperTS_duration[s]',
                     'run_kwargs'])

    data = {'dataset': dataset, 'shape': shape, 'data_size': data_size, 'task': task, 'forecast_len': forecast_len,
            'metric': metric,
            'HyperTS_SCORE': round(hypertsmetric, 6),
            'HyperTS_duration[s]': round(hypertscost, 1),
            'run_kwargs': run_kwargs}
    metrics_df = metrics_df.append(data, ignore_index=True)
    metrics_df.to_csv(result_file_path, index=False)


def get_param(config, key):
    return config[key] if key in config else None


time_start = time.time()
print("start", time.strftime("%Y-%m-%d %H:%M:%S"))

for type in types:
    for data_size in data_sizes:
        path = data_base_path + os.sep + type + os.sep + data_size
        if os.path.exists(path):
            list = os.listdir(path)
            for dir in list:
                if dir == '__init__.py' or dir == 'template':
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
                    forecast_len = get_param(config, 'forecast_len')
                    # type = config['type']
                    dtformat = get_param(config, 'dtformat')
                    date_col_name = get_param(config, 'date_col_name')
                    data_name = config['name']
                    metric = config['metric']
                    shape = config['shape']
                    task = config['task']
                    forecast_len = get_param(config, 'forecast_len')
                    series_col_name = config['series_col_name'].split(",") if 'series_col_name' in config else None
                    covariables = config['covariables_col_name'].split(",") if 'covariables_col_name' in config else None
                    f.close()
                    if covariables == None:
                        continue
                    if data_name in trained_data_names:
                        print('==skipped== already trained ', data_name)
                        continue

                    try:
                        hypertsmetric, hypertscost, run_kwargs = trail(train_file_path, test_file_path,
                                                                       date_col_name,
                                                                       series_col_name, forecast_len, dtformat, type,
                                                                       metric,
                                                                       covariables)
                        save_metrics(data_name, metric, hypertsmetric, hypertscost, shape, data_size, task,
                                     forecast_len,
                                     run_kwargs)
                    except Exception:
                        traceback.print_exc()
                        print(" Error: " + train_file_path)

time_end = time.time()

print("end  ", time.strftime("%Y-%m-%d %H:%M:%S"))
print("total cost", time_end - time_start)
