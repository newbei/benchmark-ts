# Loading the package
from util.util import save_metrics, get_param, initparams, convertdf

import pandas as pd
from core.framework_forecasting_uni import run
from hyperts.utils import metrics
import time
from util import data_loader
import math

# import tsfresh

types = ['forecastingdata-uni']
data_sizes = ['v2']
metrics_target = ['smape', 'mape', 'rmse', 'mae']
task_calc_score = 'rmse'
task_trail = 'forecast'


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

    from hypernets.core.search_space import Choice, Int, Real
    from hyperts.macro_search_space import DLForecastSearchSpace

    from hyperts.macro_search_space import StatsForecastSearchSpace

    custom_search_space = DLForecastSearchSpace(task='univariate-forecast',
                                                enable_deepar=False,
                                                enable_hybirdrnn=False,
                                                enable_lstnet=True,
                                                window=[10, 20],
                                                timestamp=Date_Col_Name,
                                                lstnet_init_kwargs={
                                                    'timestamp': Date_Col_Name,
                                                    'task': task,
                                                    'metrics': metric,
                                                    # 'reducelr_patience': 5,
                                                    # 'earlystop_patience': 20,
                                                    'summary': True,
                                                    # 'rnn_type': Choice(['simple_rnn', 'gru', 'lstm']),
                                                    'rnn_type': Choice(['gru']),
                                                    'skip_rnn_type': Choice(['simple_rnn', 'gru', 'lstm']),
                                                    'cnn_filters': Choice([32]),
                                                    'kernel_size': Choice([3]),
                                                    'rnn_units': Choice([32]),
                                                    'skip_rnn_units': Choice([4]),
                                                    'rnn_layers': Choice([4]),
                                                    'skip_rnn_layers': Choice([2]),
                                                    'out_activation': Choice(['linear', 'sigmoid']),
                                                    'drop_rate': Choice([0., 0.1, 0.2]),
                                                    'skip_period': Choice([100]),
                                                    'ar_order': Choice([12]),
                                                    'window': Choice([pow(2, 7)]),
                                                    'y_log': Choice(['logx', 'log-none']),
                                                    'y_scale': Choice(['min_max', 'max_abs'])
                                                },
                                                lstnet_fit_kwargs={
                                                    'epochs': 60,
                                                    'batch_size': None,
                                                    'verbose': 1,
                                                }
                                                )

    # from tsfresh import extract_features
    # train_df['id'] = train_df.index
    # extracted_features = extract_features(train_df, column_id="id", column_sort="datetime",chunksize=10)
    train_df['is_month_end'] = train_df['datetime'].map(lambda x: x.is_month_end)
    train_df['is_month_start'] = train_df['datetime'].map(lambda x: x.is_month_start)
    train_df['is_quarter_end'] = train_df['datetime'].map(lambda x: x.is_quarter_end)
    train_df['is_quarter_start'] = train_df['datetime'].map(lambda x: x.is_quarter_start)
    train_df['is_year_start'] = train_df['datetime'].map(lambda x: x.is_year_start)
    train_df['month'] = train_df['datetime'].map(lambda x: x.month)
    train_df['quarter'] = train_df['datetime'].map(lambda x: x.quarter)
    train_df['day'] = train_df['datetime'].map(lambda x: x.day)
    train_df['dayofweek'] = train_df['datetime'].map(lambda x: x.dayofweek)
    train_df['dayofyear'] = train_df['datetime'].map(lambda x: x.dayofyear)
    train_df['days_in_month'] = train_df['datetime'].map(lambda x: x.days_in_month)

    eval_df = train_df[-df_test.shape[0]:]
    # train_df = train_df[:-df_test.shape[0]]

    exp = make_experiment(train_df,
                          mode='dl',
                          timestamp=Date_Col_Name,
                          # covariables=['is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start',
                          #              'is_year_start', 'month', 'quarter', 'day', 'dayofweek', 'dayofyear',
                          #              'days_in_month'],
                          eval_data=eval_df,
                          task=task,
                          reward_metric=metric,
                          max_trials=max_trials,
                          optimize_direction='min',
                          verbose=1,
                          log_level='INFO',
                          early_stopping_rounds=30,
                          dl_gpu_usage_strategy=1,
                          search_space=custom_search_space,
                          **{'epoch': 2}
                          )

    model = exp.run()
    y_pred = model.predict(df_test)
    time2_end = time.time()

    return df_test, exp.run_kwargs, (time2_end - time2_start), y_pred


run(types, data_sizes, trail)
