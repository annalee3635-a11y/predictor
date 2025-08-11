import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import date

def predict(code):
    # Fetch historical data for stock
    end_date = date.today()
    stock_data = yf.download(code, start='2021-01-01', end=end_date)

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
    model.fit(x_train, y_train, epochs=20, batch_size=32)

    # Predict stock prices on the test data
    predictions = model.predict(x_test)

    # Inverse transform the predictions back to original price scale
    predictions = scaler.inverse_transform(predictions)

    # Inverse transform the actual test data
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # make a future data set
    future_dates = pd.date_range(start=end_date, periods=seq_length)
    reshape_prices = close_prices.reshape(-1, 1)
    last_prices = reshape_prices[-seq_length:]
    last_prices_scaled = scaler.transform(last_prices)
    x_pred = last_prices_scaled
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    
    #predict the prices for the future dates
    predicted_prices_scaled = model.predict(x_pred)
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

    
    name = yf.Ticker(code).info.get('shortName', 'N/A')
    # Create plotly figures
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=(name + " Stock Price Past Prediction", name + " Stock Price Past Prediction")
    )

    past = go.Figure()
    future = go.Figure()

    # Add trace for actual prices
    fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=y_test_scaled.flatten(), mode='lines', name='Actual Price'), row=1, col=1)

    # Add trace for predicted prices
    fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=predictions.flatten(), mode='lines', name='Predicted Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=future_dates[-len(y_test):], y=predicted_prices.flatten(), mode='lines', name='prediction'), row=1, col=2)

    # Add titles and labels
    #name = yf.Ticker(code).info.get('shortName', 'N/A')
    #fig.update_layout(title= name + ' Stock Price Past Prediction', xaxis_title='Date', yaxis_title='Stock Price (USD)')
    #fig.update_layout(title= name + ' Stock Price Future Prediction', xaxis_title='Date', yaxis_title='Stock Price (USD)')
    fig.update_yaxes(title_text="Stock Price (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)

    # Show the figure
    fig.show()

predict('A')