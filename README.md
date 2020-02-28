# Using Facebook's prophet to predict Schiphol passenger numbers

Here I'll introduce a straightforward application of [facebook's prophet package](https://facebook.github.io/prophet/) by using it to predict airport passenger numbers through Schiphol airport.

## Introduction to Prophet

Time series datasets are omniprescent; whether we are charting the daily close price of a stock, the population decrease of a species, or the sales of a certain product, we are recording some variable of interest at (ir)regular time intervals to observe and, hopefully, understand how it is affected by time. The ultimate goal of course is forecasting, we'd like to measure our variable of interest (stock price, population, sales) for a while and then use that data to predict the value our variable will take in the next minute/month/millenium.

Perhaps the most well known time series model is the Autoregressive integrated moving average or ["ARIMA"](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) model. Simply put, auto-regressive models are built on the idea that for time series data, the value of a variable at a given time is a function of the values it held at previous times. This idea is rather intuitive to us; if the temperature at 2pm is 28°C