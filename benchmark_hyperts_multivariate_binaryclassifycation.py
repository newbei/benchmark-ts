# Loading the package
from util.util import save_metrics, get_param, initparams

import pandas as pd
from core.framework import run
from hyperts.utils import metrics
import time
from sktime.datatypes._panel._convert import from_2d_array_to_nested, is_nested_dataframe

result_file_path, trained_data_names, mode, data_base_path, max_trials = initparams()

types = ['classfication-binary']
data_sizes = ['small', 'medium', 'large']
metrics_target = ['accuracy', 'precision', 'recall', 'roc', 'auc', 'recall', 'f1']
task_calc_score = 'binary'
task_trail = 'multivariate-binaryclass'


def trail(TRAIN_APTH, TEST_PATH, Date_Col_Name, Series_Col_name, forecast_length, format, task, metric, covariables,
          max_trials):
    if Series_Col_name == None or len(Series_Col_name) <= 1:
        print("Didin't found Series_Col_name count > 1 , not multivariate-binaryclass")
        return
    try:
        df_train = pd.read_csv(TRAIN_APTH)
        df_test = pd.read_csv(TEST_PATH)
    except:
        df_train = pd.read_pickle(TRAIN_APTH[:-4]+'.pkl')
        df_test = pd.read_pickle(TEST_PATH[:-4]+'.pkl')

    y_test, run_kwargs, time_cost, y_pred = trail_classfication(Date_Col_Name, Series_Col_name, covariables,
                                                                df_test, df_train, format, metric, task_trail,
                                                                max_trials)

    # Metrics
    return metrics.calc_score(y_test, y_pred, metrics=metrics_target, task=task_calc_score), time_cost, run_kwargs


def to_series(x):
    try:
        value= pd.Series([float(item) for item in x.split('|')])
    except:
        print('error')
        value= pd.Series([float(item) for item in x.split('|')])
    return value



def trail_classfication(Date_Col_Name, Series_Col_name, covariables, df_test, df_train, format, metric, task,
                        max_trials):
    if not any([isinstance(v, pd.Series) for v in df_train.loc[0]]):
        df_train = df_train.copy()
        df_test = df_test.copy()
        Y_train = pd.DataFrame(df_train['y'])
        df_train = df_train.drop(['y'], axis=1)

        for col in df_train.columns:
            df_train[col] = df_train[col].map(to_series)
        df_train['y'] = Y_train

        Y_test = pd.DataFrame(df_test['y'])
        df_test = df_test.drop(['y'], axis=1)
        for col in df_test.columns:
            df_test[col] = df_test[col].map(to_series)
        df_test['y'] = Y_test
    Y_test = pd.DataFrame(df_test['y'])

    y_pred, time_cost, run_kwargs = hpyertstest(df_train, df_test, Date_Col_Name, format, task, covariables,
                                                metric, max_trials)
    print("trail_classfication")
    return Y_test, run_kwargs, time_cost, y_pred


def hpyertstest(train_df, test_df, Date_Col_Name, format, task, covariables, metric, max_trials):
    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts
    train_df = train_df.copy(deep=True)
    params = {'pos_label': 1}

    exp = make_experiment(train_df,
                          timestamp=Date_Col_Name,
                          task=task,
                          reward_metric=metric,
                          timestamp_format=format,
                          covariables=covariables,
                          max_trials=max_trials,
                          target='y',
                          optimize_direction='max', **params,
                          verbose=1,
                          log_level='INFO'
                          )

    model = exp.run()
    X_test, y_test = model.split_X_y(test_df.copy())
    y_pred = model.predict(X_test)
    time2_end = time.time()
    return y_pred, (time2_end - time2_start), exp.run_kwargs


run(types, data_sizes, trail)
