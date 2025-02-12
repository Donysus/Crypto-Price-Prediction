
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime

API_KEY = "YOUR ALPHA VANTAGE API KEY"

# Step 1: Fetch Cryptocurrency Data
def fetch_crypto_data(symbol="BTC", market="USD"):
    """
    Fetch cryptocurrency data from Alpha Vantage API.
    """
    cc = CryptoCurrencies(key=API_KEY, output_format='pandas')
    data, meta_data = cc.get_digital_currency_daily(symbol=symbol, market=market)

    # Print columns to debug structure
    print("Data columns returned by Alpha Vantage:", data.columns)

    # Check for '4. close' column and use it directly
    if '4. close' in data.columns:
        data = data[['4. close']]
    else:
        raise ValueError("Expected close price column ('4. close') not found in Alpha Vantage response.")

    # Rename and preprocess
    data.columns = ['Close']
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()  # Sort by date
    return data

# Step 2: Preprocess Data
def preprocess_data(data, window_size=60):
    """
    Preprocess data for LSTM.
    - Scale data to range [0, 1].
    - Create sequences of the specified window size.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Step 3: Create LSTM Model
def create_lstm_model(input_shape):
    """
    Create an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Predict and Visualize
def predict_and_plot(model, X_test, y_test, scaler, data):
    """
    Predict using the model and plot results.
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Inverse scaling for actual values
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(y_test):], y_test_scaled, label="Actual Prices", color="blue")
    plt.plot(data.index[-len(y_test):], predictions, label="Predicted Prices", color="red")
    plt.title("Cryptocurrency Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    return predictions

# Step 5: Generate Recommendation
def generate_recommendation(predictions):
    """
    Generate a Buy/Sell recommendation based on predicted trend.
    """
    if predictions[-1] > predictions[-2]:
        return "Buy: The price is expected to increase."
    else:
        return "Sell: The price is expected to decrease."

# Main Execution
if __name__ == "__main__":
    # Fetch cryptocurrency data
    symbol = "BTC"  # Change to desired cryptocurrency symbol
    market = "USD"
    print(f"Fetching data for {symbol}/{market}...")
    crypto_data = fetch_crypto_data(symbol, market)

    # Preprocess the data
    print("Preprocessing data...")
    window_size = 60
    X, y, scaler = preprocess_data(crypto_data.values, window_size)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and train the LSTM model
    print("Building and training the LSTM model...")
    lstm_model = create_lstm_model((X_train.shape[1], 1))
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Predict and plot results
    print("Making predictions and plotting results...")
    predictions = predict_and_plot(lstm_model, X_test, y_test, scaler, crypto_data)

    # Generate Buy/Sell recommendation
    recommendation = generate_recommendation(predictions)
    print(f"Recommendation: {recommendation}")
