# 🚀 AI-Powered Cryptocurrency Price Prediction Using LSTMs  

This project is an **AI-driven cryptocurrency price prediction system** that fetches **real-time Bitcoin price data**, applies **deep learning (LSTMs)** for trend forecasting, and generates **buy/sell recommendations** based on predicted price movements.

## 📌 Features  
✅ **Real-Time Market Data** – Fetches **live cryptocurrency prices** from **Alpha Vantage** API.  
✅ **Deep Learning-Based Predictions** – Uses **LSTM (Long Short-Term Memory) neural networks** for accurate price forecasting.  
✅ **Technical Indicator-Based Analysis** – Uses **MinMax Scaling** for normalized feature extraction.  
✅ **Real-Time Visualization** – Displays **actual vs predicted price trends** dynamically.  
✅ **Buy/Sell Recommendations** – Provides **trading signals** based on predicted trends.  

---

## ⚡ Installation

### Step 1: Install Dependencies  
Run the following command to install required libraries:  
```bash
pip install -r requirements.txt
```
### Step 2: Run the Code
Execute the Python script for real-time cryptocurrency price prediction:
```bash
python crypto_prediction.py
```

## 🛠 Project Structure
``` bash
crypto-price-prediction/
│── crypto_prediction.py       # Main AI-powered cryptocurrency price prediction script
│── requirements.txt           # List of dependencies
│── README.md                  # Documentation
│── .gitignore                 # Ignore unnecessary files
│── models/                    # Saved trained LSTM models
│── results/                    # Model performance visualizations
```

## 📊 How It Works
1️⃣ Fetches real-time cryptocurrency prices from Alpha Vantage (alpha_vantage).
2️⃣ Applies data preprocessing & scaling using MinMaxScaler.
3️⃣ Trains an LSTM-based deep learning model to predict future prices.
4️⃣ Generates real-time predictions & displays actual vs predicted price trends.
5️⃣ Provides Buy/Sell recommendations based on trend direction.

📌 Check out the full implementation: GitHub Repo

💬 Let's discuss AI-powered finance! 🚀

#Crypto #DeepLearning #AI #Bitcoin #Finance #LSTM #DataScience #Python
