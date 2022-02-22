from hyperts.utils import consts, metrics
from hyperts.utils._base import get_tool_box
from hyperts.experiment import make_experiment
import pandas as pd


# TODO Evaluation Metric: RMSLE
# TODO 序列的平稳性检验,序列的差分处理，对于商超来说，序列的应该如何选取


class Test_Multivariate_Forecast_Metrics():
    def test_multivariate_forecast_metrics_mape(self):
        _test_multivariate_forecast(consts.Metric_RMSE)

def _test_multivariate_forecast(metric):
    df = pd.read_csv('../datas/test.csv')
    # df['store_nbr_family'] = df['store_nbr'].map(str) + df['family'].map(str)
    # df.drop(['store_nbr', 'family', 'id', 'onpromotion'], inplace=True, axis=1)
    # df = df.pivot(index='date', columns='store_nbr_family', values='sales')
    # df['TimeStamp'] = df.index
    # df = df.drop(df.columns.values[:1778], axis=1) # TODO 序列跑不通
    # tb = get_tool_box(df)
    # train_df, test_df = tb.temporal_train_test_split(df, test_size=0.1)

    timestamp = 'Date'
    task = consts.Task_MULTIVARIATE_FORECAST
    reward_metric = metric
    optimize_direction = consts.OptimizeDirection_MINIMIZE
    params = {'maxlags': 1}
    exp = make_experiment(df,
                          timestamp=timestamp,
                          task=task,
                          callbacks=None,
                          reward_metric=reward_metric,
                          optimize_direction=optimize_direction,
                          early_stopping_rounds=30, **params) # TODO early stopping 的次数默认值不足30次，导致基本都是不满30次就结束了，最终的随机参数选择，默认值可以优化
    # TODO 需要支持RMSLE指标

    # TODO VAR model 总共有120的参数空间，在小数据量级的情况下，效率还是比价快的
    # TODO 需要 wape swape 的评价指标，在hypernets 中?

    model = exp.run(max_trials=100)

    # X_test, y_test = process_test_data(df, timestamp=timestamp, impute=True)
    # y_pred = model.predict(X_test)
    # assert y_pred.shape == y_test.shape
    # score = metrics.rmse(y_test, y_pred)
    # print('multivariate_forecast rmse', metric, ': ', score)
    print('end')
