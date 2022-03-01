# Loading the package
from util.util import save_metrics, get_param, initparams,convertdf

import pandas as pd
from core.framework import run
from hyperts.utils import metrics
import time


types = ['univariate-forecast']
data_sizes = ['small', 'medium', 'large']
metrics_target = ['smape', 'mape', 'rmse', 'mae']

def trail(TRAIN_APTH, TEST_PATH, Date_Col_Name, Series_Col_name, forecast_length, format, task, metric, covariables,max_trials):
    # load data
    df_train = pd.read_csv(TRAIN_APTH)
    df_test = pd.read_csv(TEST_PATH)
    df_test, run_kwargs, time_cost, y_pred = trail_forecast(Date_Col_Name, Series_Col_name, covariables, df_test,
                                                            df_train, format, metric, task,max_trials)

    # Metrics
    return metrics.calc_score(df_test.drop(Date_Col_Name, 1), y_pred.drop(Date_Col_Name, 1),
                              metrics=metrics_target), time_cost, run_kwargs


def trail_forecast(Date_Col_Name, Series_Col_name, covariables, df_test, df_train, format, metric, task,max_trials):

    df_train, df_test = convertdf(df_train, df_test, Date_Col_Name,Series_Col_name,covariables, format )

    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts
    train_df = df_train.copy(deep=True)

    exp = make_experiment(train_df,
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
    y_pred = model.predict(df_test)
    time2_end = time.time()

    return df_test, exp.run_kwargs, (time2_end - time2_start), y_pred


run(types, data_sizes, trail)
