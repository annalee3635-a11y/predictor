import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Fetch historical stock data for Tesla
end_date = '2025-08-01'
stock_data = yf.download('TSLA', start='2016-10-01', end=end_date)

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

model.add(LSTM(units=100))
model.add(Dropout(0.2))

# Add output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Predict stock prices on the test data
predictions = model.predict(x_test)

# Inverse transform the predictions back to original price scale
predictions = scaler.inverse_transform(predictions)

# Inverse transform the actual test data
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

future_dates = pd.date_range(start=end_date, periods=seq_length)
reshape_prices = close_prices.reshape(-1, 1)
last_prices = reshape_prices[-seq_length:]
last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1))
#x_pred = np.array([last_prices_scaled[-seq_length:, 0]])
x_pred = last_prices_scaled
x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
predicted_prices_scaled = model.predict(x_pred)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

print(predicted_prices)
print(len(future_dates))
print(len(predicted_prices.flatten()))
future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})
print(future_data)
"""
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
"""