# Loading the package

import matplotlib.pyplot as plt
import pandas as pd
from hyperts.utils import metrics
import time


def autotstest(train_df, forecast_length, Date_Col_Name, Series_Col_name):
    time1_start = time.time()
    from autots import AutoTS
    train_df = train_df.copy(deep=True)
    train_df.index = pd.DatetimeIndex(train_df[Date_Col_Name])
    metric_weighting = {'rmse_weighting': 1}
    model = AutoTS(forecast_length=forecast_length,
                   drop_data_older_than_periods=366, metric_weighting=metric_weighting,
                   max_generations=1)  # TODO max_generations 去掉  drop_data_older_than_periods 这个值给多少？

    train_df.drop(Date_Col_Name, 1, inplace=True)
    model = model.fit(train_df)

    prediction = model.predict()
    forecast = prediction.forecast
    time1_end = time.time()
    return forecast, (time1_end - time1_start)


def hpyertstest(train_df, test_df, Date_Col_Name):
    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts
    train_df = train_df.copy(deep=True)
    # train_df.drop(Date_Col_Name, 1, inplace=True)

    model = make_experiment(train_df,
                            timestamp=Date_Col_Name,
                            task=consts.Task_MULTIVARIATE_FORECAST,
                            reward_metric=consts.Metric_RMSE
                            ).run()
    y_pred = model.predict(test_df)[-forecast_length:]
    time2_end = time.time()
    return y_pred, (time2_end - time2_start)


def metricsprint(y_true, y_pred1, y_pred2, cost1, cost2):
    # Metrics
    print("--------------------------------------------------------------")
    print(" Metrics, rmse, mae,cost")
    print(' AutoTS, ', metrics.rmse(y_true, y_pred1),
          metrics.mae(y_true, y_pred1),
          cost1, 's')
    print(' HyperTS,', metrics.rmse(y_true, y_pred2)
          , metrics.mae(y_true, y_pred2)
          , cost2, 's')
    print("--------------------------------------------------------------")

    return metrics.rmse(y_true, y_pred1), metrics.rmse(y_true, y_pred2), cost1, cost2


def trail(CSV_APTH, Date_Col_Name, Series_Col_name, forecast_length, format):
    # load data
    df = pd.read_csv(CSV_APTH)
    df[Date_Col_Name] = pd.to_datetime(df[Date_Col_Name], format=format)  # TODO

    if Series_Col_name != None:
        df[Series_Col_name] = df[Series_Col_name].astype(float)  # TODO
        # split data
        df = df[[Date_Col_Name, Series_Col_name]]
        train_df = df[:-forecast_length]
        test_df = df[-forecast_length:]


    else:
        df.columns
        for col in df.columns:
            if col == Date_Col_Name:
                continue
            df[col] = df[col].astype(float)
        # split data
        train_df = df[:-forecast_length]
        test_df = df[-forecast_length:]

    # TEST FOR HyperTS
    y_pred, cost2 = hpyertstest(train_df, test_df, Date_Col_Name)

    # # TEST FOR AutoTS
    forecast, cost1 = autotstest(train_df, forecast_length, Date_Col_Name, Series_Col_name)


    # Metrics
    return metricsprint(df[-forecast_length:].drop(Date_Col_Name, 1), forecast, y_pred, cost1, cost2)


def save_metrics(ind, dataset, metric, autotsmetric, hypertsmetric, autscost, hypertscost):
    import pandas as pd
    try:
        metrics_df = pd.read_csv('metrics_multivariate.csv')
    except:
        print('metrics.csv is null')
        metrics_df = pd.DataFrame(
            columns=['ind', 'dataset', 'metric', 'AutoTS_RSME', 'HyperTS_RSME', 'AutoTS_duration[s]',
                     'HyperTS_duration[s]'])

    data = {'ind': ind, 'dataset': dataset, 'metric': metric, 'AutoTS_RSME': autotsmetric,
            'HyperTS_RSME': hypertsmetric,
            'AutoTS_duration[s]': autscost,
            'HyperTS_duration[s]': hypertscost}
    metrics_df = metrics_df.append(data, ignore_index=True)
    metrics_df.to_csv('metrics_multivariate.csv', index=False)


time_start = time.time()
print("start", time.strftime("%Y-%m-%d %H:%M:%S"))

# load params
params = [
    [0, 'datas/test0/yahoo_stock.csv', 'Date', None, 14, None],
    [4, 'datas/test4/DailyDelhiClimateTrain.csv', 'date', None, 14, None],
    [5, 'datas/test5/AABA_2006-01-01_to_2018-01-01.csv', 'Date', None, 4, None],
    [9, 'datas/test9/Month_Value_1.csv', 'Period', None, 6, None]
]

# [1, 'datas/test1/10AUTOMOTIVE.csv', 'date', '10AUTOMOTIVE', 14],

for param in params:
    index = param[0]
    CSV_APTH = param[1]
    Date_Col_Name = param[2]
    Series_Col_name = param[3]
    forecast_length = param[4]
    dtformat = param[5]
    autotsmetric, hypertsmetric, autscost, hypertscost = trail(CSV_APTH, Date_Col_Name, Series_Col_name,
                                                               forecast_length, dtformat)

    save_metrics(index, CSV_APTH, 'rmse', autotsmetric, hypertsmetric, autscost, hypertscost)

time_end = time.time()

print("end  ", time.strftime("%Y-%m-%d %H:%M:%S"))
print("total cost", time_end - time_start)
