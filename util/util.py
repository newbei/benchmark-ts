import yaml
import hyperts
from hypernets import hyperctl
import traceback
import sys
import os
import pandas as pd
import time


def save_metrics(dataset, metric, hypertsmetric, hypertscost, shape, data_size, task, forecast_len, run_kwargs,
                 result_file_path):
    import pandas as pd
    try:
        metrics_df = pd.read_csv(result_file_path)
    except:
        print(result_file_path + ' is null')
        metrics_df = pd.DataFrame(
            columns=['dataset', 'shape', 'data_size', 'task', 'forecast_len', 'metric', 'metric_score',
                     'metrics_scores',
                     'HyperTS_duration[s]',
                     'run_kwargs'])

    data = {'dataset': dataset, 'shape': shape, 'data_size': data_size, 'task': task, 'forecast_len': forecast_len,
            'metric': metric,
            'metric_score': round(hypertsmetric[metric], 6),
            'metrics_scores': hypertsmetric,
            'HyperTS_duration[s]': round(hypertscost, 1),
            'run_kwargs': run_kwargs}
    metrics_df = metrics_df.append(data, ignore_index=True)
    metrics_df.to_csv(result_file_path, index=False)


def get_param(config, key):
    return config[key] if key in config else None


def initparams():
    try:
        params = hyperctl.get_job_params()
        data_base_path = params['data_path']
        report_base_path = params['report_path']
        mode = params['env']
        max_trials = params['max_trials']
    except:
        traceback.print_exc()
        vers = hyperts.__version__
        f = open("./config.yaml", 'r', encoding='utf-8')
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        data_base_path = config['data_path']
        report_base_path = config['report_path']
        mode = config['env']
        max_trials = config['max_trials']
        f.close()

    return data_base_path, report_base_path, max_trials, mode, vers

def convertdf(df_train,df_test,Date_Col_Name,Series_Col_name,covariables, format):
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
    return df_train,df_test
