# Loading the package
from util.util import convert_3d
from sktime.datatypes._panel._convert import from_2d_array_to_nested, is_nested_dataframe

import pandas as pd
from core.framework import run
from hyperts.utils import metrics
import time

types = ['classfication-binary']
data_sizes = ['small', 'medium', 'large']
metrics_target = ['accuracy', 'precision', 'recall', 'roc', 'auc', 'recall', 'f1']
task_calc_score = 'binary'
task_trail = 'classification'


def trail(TRAIN_APTH, TEST_PATH, Date_Col_Name, Series_Col_name, forecast_length, format, task, metric, covariables,
          max_trials):
    if Series_Col_name != None and len(Series_Col_name) > 1:
        print("It is a multivariate-multiclass data")
        return

    df_train = pd.read_csv(TRAIN_APTH)
    df_test = pd.read_csv(TEST_PATH)
    y_test, run_kwargs, time_cost, y_pred = _trail(Date_Col_Name, Series_Col_name, covariables,
                                                   df_test, df_train, format, metric, task_trail,
                                                   max_trials)

    # Metrics
    return metrics.calc_score(y_test['y'], y_pred, metrics=metrics_target, task=task_calc_score), time_cost, run_kwargs


def _trail(Date_Col_Name, Series_Col_name, covariables, df_test, df_train, format, metric, task, max_trials):

    Y_train = pd.DataFrame(df_train['y'])
    df_train = df_train.drop(['y'], axis=1)
    df_train = from_2d_array_to_nested(df_train)
    df_train['y'] = Y_train

    Y_test = pd.DataFrame(df_test['y'])
    df_test = df_test.drop(['y'], axis=1)
    df_test = from_2d_array_to_nested(df_test)
    df_test['y'] = Y_test

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
