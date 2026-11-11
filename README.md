# AI-Based Forecasting of Complex Time Series: Alphabet Stock Prices

## Overview
This project compares three different approaches to modeling and forecasting complex time series data, specifically focusing on Alphabet (Google) stock prices. The models evaluated include a naive persistence model, linear regression models, and Feed-Forward Neural Networks (FFNN). The project explores the impact of Autoregression (AR) and Autoregression with Exogenous variables (ARX), as well as the difference between predicting raw prices versus price differences.

## Author
Izabela Bargieł

## Dataset
The dataset contains Alphabet stock prices spanning multiple years, along with essential market drivers.

**Features used:**
* `adjprice`: Adjusted closing price of Alphabet (Target Variable)
* `volume`: Trading volume
* `oil`: Oil prices
* `gold`: Gold prices
* `sp500`: S&P 500 index
* `vix`: Volatility Index
* `rates_10y`: 10-year Treasury yield rates

**Data Preprocessing:**
* Removed non-numeric and text columns.
* Handled anomalies by converting negative values to `NaN`.
* Imputed missing values using K-Nearest Neighbors (`KNNImputer`).
* Due to significant structural changes and fluctuations in the latter half of the data, the experiments were focused on a more consistent subset (the first half of the dataset) to ensure stable model evaluation.

## Models Implemented
1. **Persistence Model (Naive):** Assumes the next value will be exactly the same as the current value. Used as a baseline metric.
2. **Linear Regression:**
    * **AR (Autoregressive):** Predicting future prices based solely on past prices (optimal lags found: 10).
    * **ARX (Autoregressive with Exogenous inputs):** Incorporating market drivers (optimal lags found: 1).
3. **Feed-Forward Neural Network (FFNN):** Built and optimized using PyTorch.
    * **AR on Differences:** Predicting the day-to-day price difference using only past prices.
    * **ARX on Differences:** Predicting the price difference using past prices and exogenous variables.
    * **ARX on Raw Prices:** Predicting the actual price value directly.

## Key Findings & Results
* **The Naive Model Wins:** Surprisingly, the simple Persistence Model achieved the best overall performance (lowest RMSE: ~0.43). This highlights the highly random-walk nature of financial markets and the inherent difficulty in outperforming naive baselines on daily stock data.
* **Stationarity is Crucial for Neural Networks:** The worst-performing model was the FFNN predicting raw, unscaled prices directly (RMSE: ~0.71). Neural networks often struggle to extrapolate outside their training range. However, when the problem was reframed to predict *price differences* (making the series stationary), the FFNN's performance improved dramatically (RMSE: ~0.47), making it highly competitive with the linear models.
* **Data Quality Over Complexity:** The experiments demonstrated that feeding cleaner, less volatile data (the first half of the dataset) improved the models' performance almost tenfold compared to the full, highly fluctuating dataset. This emphasizes how tricky and irregular financial data can be.

## Technologies Used
* **Data Manipulation:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn (KNNImputer, LinearRegression, StandardScaler)
* **Deep Learning:** PyTorch
* **Data Visualization:** Matplotlib, Seaborn
