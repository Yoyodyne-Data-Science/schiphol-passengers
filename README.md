![alt text](./images/yoyodyne_data_science_logo.png "Yoydyne Data Science")

# Using Facebook's prophet to predict Schiphol passenger numbers

Here I'll introduce a straightforward application of [facebook's prophet package](https://facebook.github.io/prophet/) by using it to predict airport passenger numbers through Schiphol airport.

## Introduction to Prophet

Time series datasets are omniprescent; whether we are charting the daily close price of a stock, the population decrease of a species, or the sales of a certain product, we are recording some variable of interest at (ir)regular time intervals to observe and, hopefully, understand how it is affected by time. The ultimate goal of course is forecasting, we'd like to measure our variable of interest (stock price, population, sales) for a while and then use that data to predict the value our variable will take in the next minute/month/millenium.

Perhaps the most well known time series model is the *Autoregressive integrated moving average* or ["ARIMA"](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) model. Simply put, auto-regressive models are built on the idea that for time series data, the value of a variable at a given time is a function of the values it held at previous times. This idea is rather intuitive to us; if the average temperature at 2pm was 28°C, the temperature at 3pm is more likely to be 29°C than -1°C. However, it can be difficult for such models to capture seasonality ([as demonstrated in the prophet paper itself](https://peerj.com/preprints/3190/#) ). For example, though the temperature in the previous hours might well be sufficient for us to estimate the following hour's temperature, we also know that seasonal effects (day is warmer than night, winter is colder than summer) have a huge influence.

Realizing the importance of such features, the prophet model has seasonality built-in *a priori*. It takes a completely different tack from ARIMA and related models and more closely resembles a curve fitting exercise. Concretely, prophet decomposes a time series *y(t)* as follows:

<p align="center">
<img src="images/prophet_eq.gif" title="y(t) = g(t) +s(t)+h(t)+\epsilon_t" />
</p>

Here, *g(t)* is the trend function capturing non-periodic changes, *s(t)* represents periodic changes (e.g. daily/weekly/yearly seasonality), *h(t)* represents the effects of holidays, and &#1013;<sub>*t*</sub> is a normally distributed error term.

Ok, enough of the introduction to prophet, please read [the paper itself](https://peerj.com/preprints/3190/#) for further details. Let's crack on with our example.

## Schiphol passenger data

Schiphol monthly passenger data is freely available from the [Schiphol website](https://www.schiphol.nl/en/schiphol-group/page/transport-and-traffic-statistics/), and though there are several interesting figures published, we'll be inerested in the value labelled *Passengers (incl transit-direct)*, for which data is available from January 2010 until January 2020. Let's take a quick look at the data:

<p align="center">
  <img width="600" src="images/schiphol_data.png">
</p>

There are a two key things to note about the behaviour of passenger numbers displayed here:

 - Seasonality. We see clear seasonality in the behaviour of passenger numbers, for example passengers seem to travel more frequently in summer months. You might also be able to convince yourself of an uptick in December and January related to Christmas travel.
 
 - Trend. Passenger numbers can be seen to be steadily increasing, with this increase possibly flattening-out over recent years. 
 
 These two behaviours are, as we explained, already built into prophet, so this should be a great example problem for it to solve. 
 
 Note finally the colour difference of the last 6 data points, we've plotted them in grey to denote the fact that these are the points we're going to ask prophet to predict later on. In the notebook you'll see that the number of held back months can easily be varied, and you might want to play around with it to see how prophet fares when predicting further or nearer into the future.
 
 ## Modelling Passenger numbers with Prophet
 
 Let's get started with prophet, note all the code to generate the following figures is included in the [notebook](https://github.com/Yoyodyne-Data-Science/schiphol-passengers/blob/master/Schiphol_Passengers.ipynb), and I've provided the monthly data in a clean csv [schiphol_millions_pax_monthly.csv](https://github.com/Yoyodyne-Data-Science/schiphol-passengers/blob/master/data/schiphol_millions_pax_monthly.csv).
 
 ### Installing prophet
 
 Sometimes the installation of prophet isn't so straightforward, I've put some helpful tips and links in the markdown [notes_on_prophet_installation.md](https://github.com/Yoyodyne-Data-Science/schiphol-passengers/blob/master/notes_on_prophet_installation.md)
 
 ### Understanding patterns in the data
 
 Remember that we held out the final 6 months of data so that we can see how good prophet is at predicting unseen data. This is the equivalent of a test dataset in the usual parlance of data science train-test splits. Before trying to predict those data points and evaluating the correctness of such a prediction, let's first fit prophet on the training data itself. Though this might feel like quite a strange thing to do, as typically we are interested in how a model performs on unseen data (knowing from experience that models can perform arbitrarily well on training data, e.g. see ["overfitting"](https://en.wikipedia.org/wiki/Overfitting)), it is quite common in time series analysis, e.g. see ["backtesting"](https://en.wikipedia.org/wiki/Backtesting). Such a fit typically allows us to isolate -- and hopefully, understand -- patterns in historic data.
 
 Perhaps we could say that the benefits of such a backtesting are only apparent when we have a model with a high bias. That is, when we have made quite a few assumptions as to how our timeseries should be behaving, our backtest will confirm our initial beliefs and tell us in detail the resulting values of (well understood) parameters on our data. Whereas conversely, fitting a high variance model on the training set probably won't tell us too much; it will likely perfectly (over)fit the training data, and we'll have little idea how to interpret the parameters (think of fitting a lasso regression vs a huge random forest on a dataset).
 
 Anyway, we're getting off-topic somewhat. The prophet model has a lot of assumptions built in; namely seasonality and trend, so we can convince ourselves that fitting it on our training data will gives us a good idea of how season and trend are affecting passenger numbers in our data.
 
 Let's have a look at the training data (i.e. all data bar the last 6 months) fit with prophet:
 
 <p align="center">
  <img width="600" src="images/schiphol_data_fit.png">
</p>

Prophet has fit the data (black points) with the solid blue line, and also provided us a confidence interval (by default this is 80%). By eye, we can see that the fit to the data is pretty good, but the real power of prophet is its ability to decompose seasonality and trend. Let's plot, for example the stationary component of the above fit (i.e. seasonality *s(t)* from the equation above):

<p align="center">
  <img width="600" src="images/schiphol_data_stationary.png">
</p>

Now we can clearly see the passenger behaviour we ascertained before; passengers tend to travel during the summer months (presumably for their summer holidays) while very few people take flights late or early in the year.

We can make this seasonality even clearer by averaging this staionary component over all years in the data set, the resulting pattern is shown below:

<p align="center">
  <img width="600" src="images/schiphol_data_monthly_variation.png">
</p>

With this "zoomed-in" view, we can clearly see the peak months for flying are July and August, with almost symmetrical reduction in the neighbouring months of May-June and September-October. Indeed, perhaps the curve would be nearly entirely symmetric around July-August were it not for an uptick in flights at December and January time (likely due to festive flights).

Having confirmed our suspicion of the seasonality, let's now inspect the trend, i.e. how are passenger numbers changing year-on-year? Let's go ahead and plot the trend component, *g(t)*, of the prophet fit:

<p align="center">
  <img width="600" src="images/schiphol_data_trend.png">
</p>

We can see the overall trend is that more and more people are flying as time goes on. Further inspection reveals subtler effects, e.g. at somepoint in 2017 the rate of passenger number increase seems to have rather abruptly decreased.

### Predicting the next 6 months
 
Now we move on to the task of predicting unseen data. As mentioned previously, we'll try to predict the last six months of data, having trained on all previous data points. Again, the details of running this fit are contained in the [notebook](https://github.com/Yoyodyne-Data-Science/schiphol-passengers/blob/master/Schiphol_Passengers.ipynb), here I'll just focus on the results.

Below, we can see the fit to unseen data in grey:

<p align="center">
  <img width="600" src="images/schiphol_data_fit_held_back_plus_orig.png">
</p>

Let's zoom-in and take a closer look at the last six months:

<p align="center">
  <img width="600" src="images/schiphol_data_fit_held_back.png">
</p>

We can see that prophet performs pretty well on the unseen data, yielding a root mean square error (RMSE) of around 0.17 million passengers. N.B. I chose [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) to have the error be in units of passenger numbers, but obviously there are lots of other metrics one can use to measure the goodness-of-fit of a model's predictions versus actual values. Please feel free to increase the number of held out months and see how prophet copes with having to predict further into the future.

One thing I've found from playing around with prophet and this data is that it is far better capturing the seasonality of unseen data than it is at capturing its trend. This is perhaps, unsurprising. By definition, we expect seasonal effects to vary very little year on year. Prophet has 10 years to fit this seasonality to, and happily projects it forward onto unseen data. The trend is altogether harder to predict, there are clearly many environmental, economic and social factors at play which simply aren't captured by only considering previous values of passenger numbers. We can actually see this in the example here, though prophet nicely follows the trend of passenger number seasonality, it seems to overshoot each data point; a symptom of over estimating the trend in passenger number increase.

## Conclusions

In this example we've seen that prophet is a powerful tool for modelling timeseries data, especially data which shows strong seasonality, as is the case with passenger numbers through Schiphol airport. We touched briefly on the limitations of prophet, and perhaps for the problem considered here a blended approach would have been good, where prophet is used to decompose the timeseries into seasonality and trend and alternative data is sourced to train a machine learning model on the trend component.

For those curious as to why passenger numbers through Schiphol don't keep growing at a constant rate, [this page](http://trafficreview2018.schiphol.tangelo.nl/movements) explains some of the environmental regulations Schiphol airport must adhere to. Specifically, being limited to 500,000 movements per operational year leaves little room for passenger growth without a commensurate reduction in freight flights.