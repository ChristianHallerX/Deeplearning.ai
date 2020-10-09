import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


######### Helper ###########

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time (days)")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level



############ Series Generation ###############

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.01
noise_level = 2

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


############## Series Split #############

split_time = 1100 # YOUR CODE HERE

time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()



############ Naive Forecast ############

naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))

# blue plot, original
plot_series(time_valid, x_valid)

# orange plot, naive forecast
plot_series(time_valid, naive_forecast)

print("Naive Forecast")
print("MSE: ",keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy()) # YOUR CODE HERE
print("MAE: ",keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()) # YOUR CODE HERE




########### Moving Average ############

def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast) # YOUR CODE HERE

moving_avg = moving_average_forecast(series, 30)[split_time - 30:] # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

print("Moving Average Forecast")
print("MSE: ",keras.metrics.mean_squared_error(x_valid, moving_avg).numpy()) # YOUR CODE HERE
print("MAE: ",keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy()) # YOUR CODE HERE



############# Seasonality Difference #########

# we assume the x axis is time in days -> seasonality is a year's rythm
diff_series = (series[365:] - series[:-365]) # YOUR CODE HERE
diff_time = time[365:] # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()


############### Moving average with year difference ############

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:] # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:]) # YOUR CODE HERE
plot_series(time_valid, diff_moving_avg) # YOUR CODE HERE
plt.show()

diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid) # YOUR CODE HERE
plot_series(time_valid, diff_moving_avg_plus_past) # YOUR CODE HERE
plt.show()

print("diff_moving_avg_plus_past")
print("MSE: ",keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy()) # YOUR CODE HERE
print("MAE: ",keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy()) # YOUR CODE HERE



diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid) # YOUR CODE HERE
plot_series(time_valid, diff_moving_avg_plus_smooth_past) # YOUR CODE HERE
plt.show()

print("diff_moving_avg_plus_smooth_past")
print("MSE: ",keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()) # YOUR CODE HERE
print("MAE: ",keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()) # YOUR CODE HERE