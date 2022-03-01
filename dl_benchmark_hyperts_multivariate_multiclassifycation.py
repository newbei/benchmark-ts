# Loading the package
from util.util import convert_3d

import pandas as pd
from core.framework import run
from hyperts.utils import metrics
import time

types = ['classification-multi']
data_sizes = ['small', 'medium', 'large']
metrics_target = ['accuracy', 'precision', 'recall']
task_calc_score = 'multiclass'
task_trail = 'classification'


def trail(TRAIN_APTH, TEST_PATH, Date_Col_Name, Series_Col_name, forecast_length, format, task, metric, covariables,
          max_trials):
    if Series_Col_name == None or len(Series_Col_name) <= 1:
        print("Didin't found Series_Col_name count > 1 , classification-multi")
        return

    try:
        df_train = pd.read_csv(TRAIN_APTH)
        df_test = pd.read_csv(TEST_PATH)
    except:
        df_train = pd.read_pickle(TRAIN_APTH[:-4]+'.pkl')
        df_test = pd.read_pickle(TEST_PATH[:-4]+'.pkl')

    y_test, run_kwargs, time_cost, y_pred = _trail(Date_Col_Name, Series_Col_name, covariables,
                                                   df_test, df_train, format, metric, task_trail,
                                                   max_trials)

    # Metrics
    return metrics.calc_score(y_test['y'], y_pred, metrics=metrics_target, task=task_calc_score), time_cost, run_kwargs


def _trail(Date_Col_Name, Series_Col_name, covariables, df_test, df_train, format, metric, task, max_trials):

    if not any([isinstance(v, pd.Series) for v in df_train.loc[0]]):
        df_train, df_test = convert_3d(df_train.copy(), df_test.copy())

    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts

    train_df = df_train.copy(deep=True)

    exp = make_experiment(train_df,
                          mode='dl',
                          task='classification',
                          reward_metric=metric,
                          max_trials=max_trials,
                          optimize_direction='max',
                          target='y',
                          verbose=1,
                          log_level='INFO'
                          )

    model = exp.run()
    X_test, y_test = model.split_X_y(df_test.copy())
    y_pred = model.predict(X_test)
    time2_end = time.time()

    return df_test, exp.run_kwargs, (time2_end - time2_start), y_pred


run(types, data_sizes, trail)
