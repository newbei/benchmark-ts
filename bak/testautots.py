# Loading the package
from autots import AutoTS
import matplotlib.pyplot as plt
import pandas as pd
from hyperts.utils import metrics

# Reading the data
df = pd.read_csv('../datas/test0/yahoo_stock.csv')
df['Close'] = df['Close'].astype(float)
df['Date'] = pd.to_datetime(df['Date'])

# Plot to see the data:
df = df[["Date", "Close"]]
df["Date"] = pd.to_datetime(df.Date)
train_df = df[:-40]
temp_df = df.set_index('Date')
metric_weighting = {'rmse_weighting': 1}

model = AutoTS(max_generations=1, forecast_length=40, frequency='infer', ensemble='simple',
               drop_data_older_than_periods=100, metric_weighting=metric_weighting)
model = model.fit(train_df, date_col='Date', value_col='Close', id_col=None)

prediction = model.predict()
forecast = prediction.forecast
print("Stock Price Prediction of Apple")
print(forecast)

score = metrics.rmse(df[-40:]['Close'], forecast['Close'])
print('rmse', ': ', score)

df = df.set_index('Date')
df['Close'].plot(figsize=(15, 8), title='AAPL Stock Price', fontsize=18, label='Train')
forecast['Close'].plot(figsize=(15, 8), title='AAPL Stock Price', fontsize=18, label='Test')
plt.legend()
plt.grid()
plt.show()

print("test")
