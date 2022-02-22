# Loading the package

import matplotlib.pyplot as plt
import pandas as pd
from hyperts.utils import metrics
import time


def autotstest(train_df, forecast_length):
    time1_start = time.time()
    from autots import AutoTS

    metric_weighting = {'rmse_weighting': 1}
    model = AutoTS(forecast_length=forecast_length,
                   drop_data_older_than_periods=366, metric_weighting=metric_weighting)
    model = model.fit(train_df, date_col=Date_Col_Name, value_col=Series_Col_name, id_col=None)

    prediction = model.predict()
    forecast = prediction.forecast
    time1_end = time.time()
    return forecast, (time1_end - time1_start)


def hpyertstest(train_df, test_df):
    time2_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts

    model = make_experiment(train_df,
                            timestamp='Date',
                            task=consts.Task_UNIVARIATE_FORECAST,
                            reward_metric=consts.Metric_RMSE
                            ).run()
    y_pred = model.predict(test_df)
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




# load params TODO


# load Date_Col_Name、 Series_Col_name、forecast_length
CSV_APTH = 'datas/test0/yahoo_stock.csv'
Date_Col_Name = 'Date'
Series_Col_name = 'Close'
forecast_length = 14

CSV_APTH = 'datas/test1/10AUTOMOTIVE.csv'
Date_Col_Name = 'date'
Series_Col_name = '10AUTOMOTIVE'
forecast_length = 14


df = pd.read_csv(CSV_APTH)
df[Series_Col_name] = df[Series_Col_name].astype(float)  # TODO
df[Date_Col_Name] = pd.to_datetime(df[Date_Col_Name])  # TODO

# Plot to see the data:
df = df[[Date_Col_Name, Series_Col_name]]
df[Date_Col_Name] = pd.to_datetime(df[Date_Col_Name])
train_df = df[:-forecast_length]
test_df = df[-forecast_length:]

# TEST FOR AutoTS
forecast, cost1 = autotstest(train_df, forecast_length)
# TEST FOR HyperTS
y_pred, cost2 = hpyertstest(train_df, test_df, forecast_length)

# Metrics
metricsprint(df[-forecast_length:][Series_Col_name], forecast[Series_Col_name], y_pred, cost1, cost2)
