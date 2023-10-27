import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def expand_data(dataframe):
    dataframe_copy = dataframe.copy()
    dataframe_copy['Time'] = pd.to_datetime(dataframe_copy['charging_hour'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
    start_index = dataframe_copy.iloc[0, -1].date() #starting point is the day of the first index
    end_index = dataframe_copy.iloc[-1, -1].date() # ending pojnt is the day of the last index
    indices = pd.date_range(start_index, end_index, freq='1H')
    dataframe_copy.set_index('Time', inplace=True)
    new_dataframe = pd.DataFrame(index=indices, data=dataframe_copy)
    new_dataframe['charging_hour'] = new_dataframe['charging_hour'].fillna(np.nan)
    new_dataframe['utilization'] = new_dataframe['utilization'].fillna(0.0)
    return new_dataframe

data = pd.read_csv("test_input.csv")
data = data[["utilization","charging_hour"]]
expanded_data_df = expand_data(data)
input = expanded_data_df['utilization'].tolist()
#print(input)