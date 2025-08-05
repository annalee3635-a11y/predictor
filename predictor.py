import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

# Fetch historical stock data for Tesla
stock_data = yf.download('TSLA', start='2016-10-01', end='2025-08-01')

# Display the first few rows of the dataset
# stock_data.head()

# Use only the 'Close' column for price prediction
close_prices = stock_data['Close'].values

# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

# Split the data into training (90%) and testing (10%) sets
train_size = int(len(scaled_data) * 0.9)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Create sequences from the training and test data
seq_length = 60  # Use the last 60 days to predict the next day's price
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Reshape the input data to be compatible with LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Initialize the model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Add output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Predict stock prices on the test data
predictions = model.predict(x_test)

# Inverse transform the predictions back to original price scale
predictions = scaler.inverse_transform(predictions)

# Inverse transform the actual test data
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create a plotly figure
fig = go.Figure()

# Add trace for actual prices
fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=y_test_scaled.flatten(), mode='lines', name='Actual Price'))

# Add trace for predicted prices
fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=predictions.flatten(), mode='lines', name='Predicted Price'))

# Add titles and labels
fig.update_layout(title='Tesla Stock Price Prediction', xaxis_title='Date', yaxis_title='Stock Price (USD)')

# Show the figure
fig.show()