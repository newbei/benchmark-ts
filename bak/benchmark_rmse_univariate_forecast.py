# Loading the package

import matplotlib.pyplot as plt
import pandas as pd
from hyperts.utils import metrics
import time
import traceback


def autotstest(train_df, forecast_length, Date_Col_Name, Series_Col_name, y_true):
    time_start = time.time()
    try:
        from autots import AutoTS

        metric_weighting = {'rmse_weighting': 1}
        model = AutoTS(forecast_length=forecast_length,
                       drop_data_older_than_periods=366, metric_weighting=metric_weighting,
                       random_seed=9527,
                       max_generations=1)  # TODO max_generations 去掉
        model = model.fit(train_df, date_col=Date_Col_Name, value_col=Series_Col_name, id_col=None)

        prediction = model.predict()
        forecast = prediction.forecast
        autotsmetric = metrics.smape(y_true, forecast)
    except Exception as e:
        traceback.print_exc()
        autotsmetric = 'error'
    return autotsmetric, (time.time() - time_start)


def hpyertstest(train_df, test_df, Date_Col_Name, y_true):
    time_start = time.time()
    try:
        from hyperts.experiment import make_experiment
        from hyperts.utils import consts

        model = make_experiment(train_df,
                                timestamp=Date_Col_Name,
                                task=consts.Task_UNIVARIATE_FORECAST,
                                reward_metric=consts.Metric_RMSE,
                                random_state=9527,
                                ).run()
        y_pred = model.predict(test_df)[-forecast_length:]
        hypertsmetric = metrics.smape(y_true, y_pred)
    except Exception as e:
        traceback.print_exc()
        hypertsmetric = 'error'
    return hypertsmetric, (time.time() - time_start)


def hpyertsdltest(train_df, test_df, Date_Col_Name, y_true):
    time_start = time.time()
    from hyperts.experiment import make_experiment
    from hyperts.utils import consts

    try:
        model = make_experiment(train_df.copy(),
                                mode='dl',
                                timestamp=Date_Col_Name,
                                task=consts.Task_UNIVARIATE_FORECAST,
                                reward_metric=consts.Metric_RMSE,
                                optimize_direction=consts.OptimizeDirection_MINIMIZE,
                                random_state=9527).run()
        y_pred3 = model.predict(test_df)[-forecast_length:]
        hypertsdlmetric = metrics.smape(y_true, y_pred3)
    except Exception as e:
        traceback.print_exc()
        hypertsdlmetric = 'error'

    return hypertsdlmetric, (time.time() - time_start)


def metricsprint(y_true, autotstest_pred, hpyertstest_pred, hpyertsdltest_pred, autotstest_cost, hpyertstest_cost,
                 hpyertsdltest_cost):
    return metrics.rmse(y_true, autotstest_pred), metrics.rmse(y_true, hpyertstest_pred), metrics.rmse(y_true,
                                                                                                       hpyertsdltest_pred), autotstest_cost, hpyertstest_cost, hpyertsdltest_cost


def trail(CSV_APTH, Date_Col_Name, Series_Col_name, forecast_length, format):
    # load data
    df = pd.read_csv(CSV_APTH)
    df[Series_Col_name] = df[Series_Col_name].astype(float)  # TODO
    df[Date_Col_Name] = pd.to_datetime(df[Date_Col_Name], format=format)  # TODO

    # split data
    df = df[[Date_Col_Name, Series_Col_name]]
    train_df = df[:-forecast_length]
    test_df = df[-forecast_length:]

    y_true = df[-forecast_length:][Series_Col_name]


    # TEST FOR HyperTS
    hypertsmetric, hypertscost = hpyertstest(train_df, test_df, Date_Col_Name,y_true)
    # TEST FOR HyperDLTS
    hypertsdlmetric, hypertsdlcost = hpyertsdltest(train_df, test_df, Date_Col_Name, y_true)
    # TEST FOR AutoTS
    autotsmetric, autscost = autotstest(train_df, forecast_length, Date_Col_Name, Series_Col_name ,y_true)

    # Metrics
    return autotsmetric, hypertsmetric, hypertsdlmetric, autscost, hypertscost, hypertsdlcost


def save_metrics(ind, dataset, metric, autotsmetric, hypertsmetric, hypertsdlmetric, autscost, hypertscost,
                 hypertsdlcost):
    import pandas as pd
    try:
        metrics_df = pd.read_csv('metrics_univariate.csv')
    except:
        print('metrics.csv is null')
        metrics_df = pd.DataFrame(
            columns=['ind', 'dataset', 'metric', 'AutoTS_RSME', 'HyperTS_RSME', 'HyperTSDL_RSME', 'AutoTS_duration[s]',
                     'HyperTS_duration[s]', 'HyperTSDL_duration[s]'])

    data = {'ind': ind, 'dataset': dataset, 'metric': metric, 'AutoTS_RSME': autotsmetric,
            'HyperTS_RSME': hypertsmetric,
            'HyperTSDL_RSME': hypertsdlmetric,
            'AutoTS_duration[s]': autscost,
            'HyperTS_duration[s]': hypertscost,
            'HyperTSDL_duration[s]': hypertsdlcost
            }
    metrics_df = metrics_df.append(data, ignore_index=True)
    metrics_df.to_csv('metrics_univariate.csv', index=False)


time_start = time.time()
print("start", time.strftime("%Y-%m-%d %H:%M:%S"))

# load params
params = [
    [0, 'datas/test0/yahoo_stock.csv', 'Date', 'Close', 14, None],
    [1, 'datas/test1/10AUTOMOTIVE.csv', 'date', '10AUTOMOTIVE', 14, None],
    [4, 'datas/test4/DailyDelhiClimateTrain.csv', 'date', 'meanpressure', 14, None],
    [5, 'datas/test5/AABA_2006-01-01_to_2018-01-01.csv', 'Date', 'Close', 4, None],
    [6, 'datas/test6/raw_sales2602.csv', 'datesold', 'price', 14, None],
    [7, 'datas/test7/infy_stock.csv', 'Date', 'Close', 14, None], # 不规则时间序列
    [8, 'datas/test8/POP.csv', 'date', 'value', 6, None],
    [9, 'datas/test9/Month_Value_1.csv', 'Period', 'Sales_quantity', 6, '%d.%m.%Y'],
    [10, 'datas/test10/Electric_Production.csv', 'DATE', 'Value', 6, None]
]

for param in params:
    index = param[0]
    CSV_APTH = param[1]
    Date_Col_Name = param[2]
    Series_Col_name = param[3]
    forecast_length = param[4]
    dtformat = param[5]
    autotsmetric, hypertsmetric, hypertsdlmetric, autscost, hypertscost, hypertsdlcost = trail(CSV_APTH, Date_Col_Name,
                                                                                               Series_Col_name,
                                                                                               forecast_length,
                                                                                               dtformat)

    save_metrics(index, CSV_APTH, 'smape', autotsmetric, hypertsmetric, hypertsdlmetric, autscost, hypertscost,
                 hypertsdlcost)

time_end = time.time()

print("end  ", time.strftime("%Y-%m-%d %H:%M:%S"))
print("total cost", time_end - time_start)
