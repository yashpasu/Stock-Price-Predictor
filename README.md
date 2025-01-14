# Stock-Price-Predictor
A model that uses scikit, numpy, pandas, and matplotlib libraries to predict stock prices based on a set amount of data given to it.

The library used to extract live stock market data is the yfinance import, which is Yahoo Finance's library for a simple access of stock market data. I have set the program to look at data over a 1 year period and over that period it will train itself in 5 day increments. How the model learns is through linear regression which is the preferred training method when a model is required to predict a value, which in this case is the stock price. The data supplied into the model is labeeled data as it will learn based on yesterdays closing price as input, and todays closing price as output to learn and predict tomorrows price. This is known as supervised learning.

Once the model predicts outputs it is displayed on a line chart that is created using matplotlib, and then it will output the predicted price for tomorrow for an entered stock. This model isnt accurate at the moment as it is only learning based off of the last years worht of data but this can be easily adjusted by changing the start date to an earlier date, which is given to the yfinance library that will be held in stockData.
![image](https://github.com/user-attachments/assets/3821ad79-009d-4d1c-9328-a583c21add98)
