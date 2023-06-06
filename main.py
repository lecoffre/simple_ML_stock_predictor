import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.model_selection import train_test_split
import datetime, os

path = os.path.dirname(__file__)
# Define the stock market index ticker symbol
ticker_symbol= input('select ticker (NVDA, TSLA, SPY): ')  # Example: TSLA, SPY for S&P 500
# Download historical data from Yahoo Finance
data = yf.download(ticker_symbol, start='2015-01-01', end='2023-06-05')
# Prepare the data for machine learning
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data['Direction'] = np.where(data['Returns'] > 0, 1, 0)
data = data.dropna()
# Select the features and the target variable
features = ['Returns', 'Volume']
target = 'Direction'
x = data[features]
y = data[target]
print(x,y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Normalize the input features
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()
# Create a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
# Predict the future movement of the stock market
last_data = X_train.iloc[[-1]].values
prediction = model.predict(last_data)[0][0]
prediction_percentage = prediction * 100
# Generate the graph with the predicted arrow
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Close Price')
plt.legend()
arrow_img = path+'\\assets\\images\\up_arrow.png' if prediction > 0.5 else path+'\\assets\\images\\down_arrow.png'
imagebox = OffsetImage(plt.imread(arrow_img), zoom=0.1)
ab = AnnotationBbox(imagebox, (data.index[-1], data['Close'].iloc[-1]), frameon=False)
plt.gca().add_artist(ab)
plt.title(f'{ticker_symbol} Stock Market Prediction, go up ({prediction_percentage:.2f}%, accuracy: {accuracy:.2f})')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid(True)
# Save the graph as an image
plt.savefig(path+'\\results\\stock_prediction_'+str(ticker_symbol)+'_'+str(datetime.date.today())+'.png')
#plt.show()
