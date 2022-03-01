# Loading the package
from util.util import save_metrics, get_param, initparams, convertdf

import pandas as pd
from core.framework_forecasting_uni import run
from hyperts.utils import metrics
import time
from util import data_loader

types = ['forecastingdata-uni']
data_sizes = ['v2']
metrics_target = ['smape', 'mape', 'rmse', 'mae']
task_calc_score = 'rmse'
task_trail = 'univariate-forecast'


def trail(df_train, df_test, max_trials):
    df_test, run_kwargs, time_cost, y_pred = trail_forecast('datetime', None, None, df_test,
                                                            df_train, format, task_calc_score, task_trail, max_trials)

    # Metrics
    return metrics.calc_score(df_test.drop('datetime', 1), y_pred.drop('datetime', 1),
                              metrics=metrics_target,
                              task='regression'), time_cost, run_kwargs, task_calc_score, task_trail


def trail_forecast(Date_Col_Name, Series_Col_name, covariables, df_test, df_train, format, metric, task, max_trials):
    df_train, df_test = convertdf(df_train, df_test, Date_Col_Name, Series_Col_name, covariables, format)

    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts
    train_df = df_train.copy(deep=True)

    exp = make_experiment(train_df,
                          timestamp=Date_Col_Name,
                          task=task,
                          reward_metric=metric,
                          # timestamp_format=format,
                          covariables=covariables,
                          max_trials=max_trials,
                          optimize_direction='min',
                          verbose=1,
                          log_level='INFO',
                          early_stopping_rounds=30
                          )

    model = exp.run()
    y_pred = model.predict(df_test)
    time2_end = time.time()

    return df_test, exp.run_kwargs, (time2_end - time2_start), y_pred


run(types, data_sizes, trail)
