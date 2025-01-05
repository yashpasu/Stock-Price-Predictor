#Install the yfinance library to get live stock prices from yahoo finance
import yfinance as yf
#Install numpy to deal with large sets of market stocks prices
import numpy as np
#Install pandas to allow effective analysis of these stock prices
import pandas as pd
#Install Scikit Model Selections' train_test_split function to split data into training and test groups (usually 80/20 split)
from sklearn.model_selection import train_test_split
#Install Scikit LinearRegression function as linear regression is the learning method used to train a model with supervised, labelled data. Stock prices are labelled
from sklearn.linear_model import LinearRegression
#Install Scikit Metric's mean_squared_error function to implements the mean square error classifications to our stock prices dataset
from sklearn.metrics import mean_squared_error
#Inport matplot lib pyplot to visually respresent the predicated rpices of a graph
import matplotlib.pyplot as plt

#Ask the user to enter the code of the stock that they want information on
stockCode = input("Enter a stock code (e.g. AAPL):")

#Receive the latest stock infomration for last 1 year on user entered stock from yfinance import
stockData = yf.download(stockCode, start = "2024-01-01", end = "2025-01-01", progress = False)

#The prices that sghould be determined as part of the dataset are the closing prices of the stock
stockData = stockData[['Close']].reset_index()

#Organise data so that it is learnable by the model. We shall train the model recusrively on an every 5 day interval
noOfDays = 5

#Create a data frame to structure the model that will be trained. This is where pandas is used
modelledData = pd.DataFrame()

#Iteratively go through each stock index and identify where it closes, and when it closes move onto the next index
for i in range(noOfDays):
    modelledData[f"lag_{i+1}"] = data['Close'].shift(i+1)

#Set todays price as the target column for model to work with
modelledData['target'] = data['Close']

#If there are values that are NaN, drop them
modelledData = modelledData.dropna()

#Split the data into training and test sets in X andY, where X is 5 day intervals and Y is the target to be reached
X = modelledData[[f"lag_{i+1}" for i in range(n_days)]]
Y = modelledData['target']

#Proceed to categorise the train and test data using the Scikit train_test_split function (testSize = 0.2 means 20% of data will be used to test and randState is a random number to test with)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, testSize = 0.2, randState = 40)

#Now train the price predicting model using linear regression
pricePredictor = LinearRegression()
pricePredictor.fit(X_train, y_train)

#Using the test data we should now evaluate the data through predictions
Y_pred = pricePredictor.predict(X_test)

#Find the mean squared error using the Scikit metrics. The mse finds a calcaulted difference between the tests outcome and the prediction
mse = mean_squared_error(Y_test, Y_pred)

