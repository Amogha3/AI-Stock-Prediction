# ============================================================
# AI-Based Stock Market Prediction & Trading Assistant
# Final Year Major Project - ISE
# Run using: streamlit run app.py
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# ---------------- UI ----------------
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Market Prediction & Trading Assistant")

st.sidebar.header("User Input")
stock = st.sidebar.text_input("Enter Stock Symbol (Example: RELIANCE.NS)", "RELIANCE.NS")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
lookback = st.sidebar.slider("LSTM Window Size", 30, 120, 60)

# ---------------- Data ----------------
@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

data = load_data(stock, start, end)

if data.empty:
    st.error("No Data Found. Check Stock Symbol.")
    st.stop()

st.subheader("Stock Data")
st.dataframe(data.tail())

# ---------------- Preprocessing ----------------
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[['Close']])

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_close, lookback)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------- Model ----------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

with st.spinner("Training Model..."):
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# ---------------- Prediction ----------------
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test)

# ---------------- Metrics ----------------
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
mae = mean_absolute_error(real_prices, predicted_prices)

st.subheader("Model Performance")
st.write("RMSE:", round(rmse,2))
st.write("MAE:", round(mae,2))

# ---------------- Trading Signal ----------------
last_real = real_prices[-1][0]
last_pred = predicted_prices[-1][0]

if last_pred > last_real:
    signal = "BUY"
elif last_pred < last_real:
    signal = "SELL"
else:
    signal = "HOLD"

st.subheader("Trading Recommendation")
st.success(f"Suggested Action: {signal}")
st.write("Current Price:", round(last_real,2))
st.write("Predicted Next Price:", round(last_pred,2))

# ---------------- Graph ----------------
st.subheader("Actual vs Predicted Prices")
fig = plt.figure(figsize=(10,5))
plt.plot(real_prices, label="Actual")
plt.plot(predicted_prices, label="Predicted")
plt.legend()
st.pyplot(fig)

# ---------------- Paper Trading ----------------
st.subheader("Paper Trading Simulation")
capital = st.number_input("Initial Capital", value=10000)

profit = 0
for i in range(1, len(predicted_prices)):
    if predicted_prices[i] > real_prices[i-1]:
        profit += real_prices[i] - real_prices[i-1]

st.write(f"Estimated Profit: ₹{round(profit, 2)}")
st.write("Final Capital:", round(capital + profit,2))

st.markdown("---")

st.write("Developed using LSTM + Streamlit | Final Year ISE Major Project")
