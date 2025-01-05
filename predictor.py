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
#Import matplot lib pyplot to visually respresent the predicated rpices of a graph
import matplotlib.pyplot as plt
#Import matplotlibs dates to use on x-axis for date intervals
import matplotlib.dates as mdates
#Import datetime module
from datetime import date, timedelta

#Ask the user to enter the code of the stock that they want information on
stockCode = input("Enter a stock code (e.g. AAPL):")

#Receive the latest stock infomration for last 1 year on user entered stock from yfinance import
stockData = yf.download(stockCode, start = "2024-01-01", end = date.today(), progress = False)

#The prices that sghould be determined as part of the dataset are the closing prices of the stock
stockData = stockData[['Close']].reset_index()

#Organise data so that it is learnable by the model. We shall train the model recusrively on an every 5 day interval
noOfDays = 5

#Create a data frame to structure the model that will be trained. This is where pandas is used
modelledData = pd.DataFrame()

#Iteratively go through each stock index and identify where it closes, and when it closes move onto the next index
for i in range(noOfDays):
    modelledData[f"lag_{i+1}"] = stockData['Close'].shift(i+1)

#Set todays price as the target column for model to work with
modelledData['target'] = stockData['Close']

#If there are values that are NaN, drop them
modelledData = modelledData.dropna()

#Split the data into training and test sets in X and Y, where X is 5 day intervals and Y is the target to be reached
X = modelledData[[f"lag_{i+1}" for i in range(noOfDays)]]
Y = modelledData['target']

#Proceed to categorise the train and test data using the Scikit train_test_split function (testSize = 0.2 means 20% of data will be used to test and randState is a random number to test with)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 40)

#Now train the price predicting model using linear regression
pricePredictor = LinearRegression()
pricePredictor.fit(X_train, Y_train)

#Using the test data we should now evaluate the data through predictions
Y_pred = pricePredictor.predict(X_test)

#Find the mean squared error using the Scikit metrics. The mse finds a calcaulted difference between the tests outcome and the prediction
mse = mean_squared_error(Y_test, Y_pred)

#Generate and display the predicated prices with the dates
testDates = stockData['Date'].iloc[-len(Y_test):].dt.strftime('%Y-%m-%d')
predictedPrices = pd.DataFrame({
    'Date' : testDates,
    'Current Price' : Y_test.values,
    'Predicted Price': Y_pred
})

#Round 'prices to 2 decimal places
predictedPrices['Current Price'] = predictedPrices['Current Price'].round(2)
predictedPrices['Predicted Price'] = predictedPrices['Predicted Price'].round(2)

print(predictedPrices)

#Now determine and display tomorrow predicted price
latestStatus = stockData['Close'].iloc[-noOfDays:].values.reshape(1,-1)
predTomorrowsPrice = pricePredictor.predict(latestStatus)[0]

print(f"Predicted Price for tomorrow ({pd.Timestamp.now().date() + timedelta(days=1)}): ${predTomorrowsPrice:.2f}")

#Use matplotlib to plot these prices on a graph as visual representation
plt.figure(figsize=(14,8))

#Draw line for current price
plt.plot(range(len(Y_test)), Y_test, label= "Current Prices", color = 'blue', linewidth = 2)
#Draw line for predicted price
plt.plot(range(len(Y_pred)), Y_pred, label= "Predicted Prices", color = 'orange', linewidth = 2)

#Format the x-axis to accomodate the 5 day intervals
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

#Rotate x axis for better visibiltiy and correct dates
plt.xticks(ticks=range(len(testDates)), labels = stockData['Date'].iloc[-len(Y_test):].dt.strftime('%Y-%m-%d'), rotation=45) 

#Add attributes to graph
plt.legend()
plt.title("Stock Market Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.show()

