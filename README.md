# Overview
The idea was to create a neural network that could predict weather data given past data. We chose 3 metrics to predict: High temp, low temp, and precipitation amount. The reason for this is that this is the data on most weather apps that is most visible. In fact, this is sometimes the ONLY data given by weather apps besides hourly temps. We ignore hourly temperatures because that's not the focus of the network. We aim to provide long term daily forecasts rather than providing hourly predictions.
# Data
The data we used to train our model comes from the exapnded version of NOAA's NOWData, [SC ACIS2](https://scacis.rcc-acis.org/
). We collected all data available from the MENOMONIE station between the dates 1/1/2020 and 4/21/2025 (The date prior to our data collection). 
Link: https://scacis.rcc-acis.org/
