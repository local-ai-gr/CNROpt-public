# On how to use the code "Forecasting with location included final model.ipynb
This file explains all the steps in the code and also the data structures, preprocessing etc.

### Importing libraries and connecting to the database
* Connecting to the database results in a dataframe **data_stations**. There are 257 charging stations in the database - all of them are used in model development.

```python
len(data_stations['location_friendlyname'].unique())
```
* Dropping the unnecessary columns (city name, address, house number)

### Data preparation
* **expand_data** function - Dataset is expanded with zero values (for hours when there is no charging) for each EV station
  * *list_names* - list of all EV stations
* **create_data_plots** function - plotting of consumption and utilization for selected station (selection based on name)
* Creation of train/validation/test/real datasets
  *  In for loop we go through each station and use 70% of data for training and rest for validation/testing/real dataset
  *  Then 4 big dataframes are made (one for train/validation/test/real dataset)
  *  **total_id_locations** - list of all IDs in dataset; this will be used for encoding the location
* **TimeSeriesDataset** class - encoding of location and time, used for features and labels
  * Since the number of features and labels of each dataset should be always the same (even when there are less months in the set or less stations), there are two if conditions:
    ```python
    if num_months < 12:
    ...
    ```
    * If there is less than 12 months of data (e.g., we have only 8 months of data for one station), the dataset is encoded with zeros on months that are missing
    ```python
    if num_loc < len(all_loc_id):
    ...
    ```
    * If there are missing stations in a dataset, it is encoded with zeros
  * There are **280 features** and **2 labels** -- saved in variables *_data_filtered
  * **train_size, val_size, test_size, real_size** - size of data, it will be used later when calculating losses
  * Data loaders - inputs are *_data_filtered and **batch_size = 1024**
 
### LSTM Model training and testing
* **LSTM_TimeSeries** class
  * inputs are: **input_size = number of features, hidden_size, layer_size, output_size = 2**
* **calculate_mse** function - for losses
* **dict_lstm_models_op** is a dictionary used for saving the model parameters and results for different configurations
  * best configuration: **batch_size = 1024, LR = 0.001, number of hidden neurons = 64, number of hidden layers = 2, optimizer = RMSprop**
* We should also create additional lists and dataframes that will be used when storing the results during training
* Training and validation of the model
  * **counter** - variable that is used for early stopping of the model
    * As long as the current loss (**val_loss**) is minimizing, model is learning. If there is no progress during 30 epochs (manually selected), model stops.
  * Best model is saved in **best_model** and best weights are saved in **min_weights** - this will be later used in the implementation in the app
* Testing of the model is done in the same way (but on the best model) - results are stored in **pred_consumption_test** and **pred_utilization_test** and in a dictionary **dict_lstm_models_op**
* **plot_everything** function - plotting of losses and predictions on validation and test set
* Saving of the best model's weights
* Testing the model's performance on each station - parameters and results are stored in **result_sum**
  * for loop goes thorugh each station, extracts the corresponding dataset and tests the model (function **test_lstm_model**)
  * **result_sum** dataframe saved as CSV
 
### LSTM User input for selected station
* Variables **time_start, time_end, location_id** should be manually selected - data is sliced based on these values from the original dataframe (**real_data**)
* Selected dataframe goes through **TimeSeriesDataset** and **DataLoader**
* **results_selected_location** is a dataframe that has predicted values and time frame for selected charging station

### Transformer model data preparation, training and testing
* **TransformerNet** class
  * inputs: **input_size = number of features, hidden_size, output_size = 2, num_layers, num_heads, dropout**
* **TransformerDataset** class - used for data preparation (since we use more historical data steps, the data should be modeled accordingly)
  * sequence_length is equal to **lookback = 12**
  * *_data_transformer are corresponding datasets
* Data loading in the same way as for the LSTM model
* **dict_transformer_model** is a dictionary used for storing model parameters and results for different configurations
  * best results: **batch_size = 512, lookback = 12, hidden_size = 128, num_layers = 2, num_heads = 1, LR = 0.001, optimizer = RMSprop**
* We also create additional lists and dataframes for storing results during training
* Testing and plotting of the results is done in the same way as for the LSTM model
* Model perfomrance is saved to CSV as **result_sum_transformer**
* User input testing is done in the same way as for the LSTM model
