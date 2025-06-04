import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

data = yf.download(stock, start, end)

model = load_model("C:\\Users\\PAWAN\\Mini Project\\latest_stock_price_model1.keras")
st.subheader("Stock Data")
st.write(data)

splitting_len = int(len(data)*70//100)
# x_test = pd.DataFrame(data.Close[splitting_len:])

x_test = pd.DataFrame(data['Close'][splitting_len:])
x_test.columns = ['Close']

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
data['MA_for_250_days'] = data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), data['MA_for_250_days'],data,0))

st.subheader('Original Close Price and MA for 200 days')
data['MA_for_200_days'] = data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), data['MA_for_200_days'],data,0))

st.subheader('Original Close Price and MA for 100 days')
data['MA_for_100_days'] = data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), data['MA_for_100_days'],data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), data['MA_for_100_days'],data,1,data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

last_100 = data["Close"].values[-100:]
last_100 = last_100.reshape(-1, 1)  
last_100_scaled = scaler.transform(last_100)
input_data = np.reshape(last_100_scaled, (1, last_100_scaled.shape[0], 1))
predicted_scaled = model.predict(input_data)
next_day_predicted_closing_value = scaler.inverse_transform(predicted_scaled)


st.subheader("Next day predicted closing value")
st.subheader(f"{next_day_predicted_closing_value[0][0]}")


print(x_data.shape)