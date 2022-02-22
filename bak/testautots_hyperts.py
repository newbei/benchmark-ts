from hyperts.utils import consts, metrics
from hyperts.utils._base import get_tool_box
from hyperts.experiment import make_experiment, process_test_data
import pandas as pd
import matplotlib.pyplot as plt


forecast_length = 40

df = pd.read_csv('../datas/test0/yahoo_stock.csv')

df['Close'] = df['Close'].astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df = df[["Date", "Close"]]
df["Date"] = pd.to_datetime(df.Date)

train_df = df[:-forecast_length]

tb = get_tool_box(df)
train_df, test_df = tb.temporal_train_test_split(train_df, test_size=40)

exp = make_experiment(train_df,
                      timestamp='Date',
                      task=consts.Task_UNIVARIATE_FORECAST,
                      reward_metric=consts.Metric_RMSE
                      )

model = exp.run()

X_test = df[-40:]
y_pred = model.predict(X_test)
forecast = X_test
forecast['Close'] = y_pred

score = metrics.rmse(df[-40:]['Close'], forecast['Close'])
print(y_pred)
print('rmse', ': ', score)

df = df.set_index('Date')
forecast = forecast.set_index('Date')

df['Close'].plot(figsize=(15, 8), title='Train', fontsize=18, label='Train')
forecast['Close'].plot(figsize=(15, 8), title='Test', fontsize=18, label='Test')
plt.legend()
plt.grid()
plt.show()