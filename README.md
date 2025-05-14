# Overview
The idea was to create a neural network that could predict weather data given past data. We chose 2 metrics to predict: High temp and low temp. The reason for this is that this is the data on most weather apps that is most visible. In fact, this is sometimes the ONLY data given by weather apps besides hourly temps. We ignore hourly temperatures because that's not the focus of the network. We aim to provide long term daily forecasts rather than providing hourly predictions.
# Data Collection
The data we used to train our model comes from the exapnded version of NOAA's NOWData, [SC ACIS2](https://scacis.rcc-acis.org/
). We collected all data available except snow depth from the MENOMONIE station between the dates 1/1/2020 and 4/21/2025 (The date prior to our data collection).
# Data Cleaning
We changed the M (Missing) tags in the data to "nan" to flag them easier. We changed the T (Trace) values to half of their respective cutoffs because we felt it was important to include that the was precipitation, no matter how trivial an amount. Specifically, for rainfall all T values were changes to .0025 inches and for snowfall T values were changed to .025 inches. We also deleted February 29th, 2020 for the purpose of potentially easier computation in the time series analysis by keeping our years to a static 365 days. This was a potentially unnecessary change. During runtime we delete rows that contain missing data. Of the 1937 days there are 57 that contain missing data. That's just shy of 3% of the total data, or equivalent to 1 missing day per month. This isn't ideal, but it's a loss we're okay accepting. We also took out the final 30 days to be our test set.
# Time Series Analysis
Time series models are used to predict future values based on historical data​. We used an ARIMA model​. It utilizes 3 parameters: p(autoregression), d(differencing), and q(moving average)​  
Autoregression: The relationship between current observations and past observations​  
Differencing: Difference between observations and previous/different observations (stationary)​  
Moving Average: The dependency between an observation and error.  
  
The RMSE for our two metrics were:  
Max_Temp: RMSE = 8.41  
Min_Temp: RMSE = 8.47

# Neural Network
Our neural network has 2 fully connected hidden layers. Specifically the number of nodes go 345 -> 128 -> 64 -> 2 where 345 is 23 features over the 15 previous days and 2 is our min and max temperatures for the day. This gave us an RMSE of about 8 over the test set.
