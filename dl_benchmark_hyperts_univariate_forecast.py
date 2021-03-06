# Loading the package
from util.util import save_metrics, get_param, initparams, convertdf

import pandas as pd
from core.framework import run
from hyperts.utils import metrics
import time
from sktime.datatypes._panel._convert import from_2d_array_to_nested, is_nested_dataframe

types = ['univariate-forecast']
data_sizes = ['small']
metrics_target = ['smape', 'mape', 'rmse', 'mae']
task_calc_score = 'regression'
task_trail = 'forecast'


def trail(TRAIN_APTH, TEST_PATH, Date_Col_Name, Series_Col_name, forecast_length, format, task, metric, covariables,
          max_trials):
    # if covariables == None:
    #     print("no covariables!!!")
    #     return
    # load data

    # if 'Poverty_Universe' not in TRAIN_APTH:
    #     return

    df_train = pd.read_csv(TRAIN_APTH)
    df_test = pd.read_csv(TEST_PATH)
    y_test, run_kwargs, time_cost, y_pred = _trail(Date_Col_Name, Series_Col_name, covariables,
                                                   df_test, df_train, format, metric, task_trail,
                                                   max_trials)

    # Metrics
    return metrics.calc_score(y_test.drop(columns=[Date_Col_Name], axis=1),
                              y_pred.drop(columns=[Date_Col_Name], axis=1),
                              metrics=metrics_target, task=task_calc_score), time_cost, run_kwargs


def _trail(Date_Col_Name, Series_Col_name, covariables, df_test, df_train, format, metric, task, max_trials):
    df_train, df_test = convertdf(df_train, df_test, Date_Col_Name, Series_Col_name, covariables, format)
    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts

    train_df = df_train.copy(deep=True)

    exp = make_experiment(train_df,
                          mode='dl',
                          timestamp=Date_Col_Name,
                          task=task,
                          reward_metric=metric,
                          timestamp_format=format,
                          covariables=covariables,
                          max_trials=max_trials,
                          optimize_direction='min',
                          verbose=1,
                          log_level='INFO'
                          )

    model = exp.run()
    X_test, y_test = model.split_X_y(df_test.copy())
    y_pred = model.predict(X_test)
    time2_end = time.time()

    return df_test, exp.run_kwargs, (time2_end - time2_start), y_pred


run(types, data_sizes, trail)
