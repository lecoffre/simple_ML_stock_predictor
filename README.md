
# Simple Stock Market Prediction using Machine Learning

This script uses machine learning with TensorFlow to predict the future movement of the stock market. It downloads historical data from Yahoo Finance using the `yfinance` library and trains a neural network model to classify whether the market will go up or down.

## Requirements

- `yfinance` library: Used to download historical stock market data from Yahoo Finance.
- `pandas` library: Used for data manipulation and preprocessing.
- `numpy` library: Used for numerical operations on the data.
- `tensorflow` library: Used to create and train the neural network model.
- `matplotlib` library: Used to plot the stock market data and generate the prediction graph.
- `scikit-learn` library: Used for splitting the data into training and testing sets.

Install the required libraries using the following command:
```
pip install yfinance pandas numpy tensorflow matplotlib scikit-learn
```

### Results
The script will print the accuracy of the model on the testing set and generate a graph showing the historical stock market data with a prediction arrow indicating the expected future movement. The arrow will be green (up_arrow.png) if the prediction suggests an upward movement, or red (down_arrow.png) for a downward movement. The percentage displayed on the graph represents the confidence level of the prediction.

![alt text](https://github.com/lecoffre/simple_ML_stock_predictor/blob/main/results/stock_prediction_NVDA_2023-06-06.png)
