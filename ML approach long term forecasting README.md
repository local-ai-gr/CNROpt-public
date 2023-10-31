# On how to use the code "Analytical approach forecasting one station.ipynb"
This file explains all the steps in the code (also the data structures, preprocessing, etc.)

### Importing the necessary libraries, connecting to the database
* Connecting to the database and dropping of the unnecessary columns results in a dataframe **df**

### Data preparation
* **expand_data** function - Dataset is expanded with zero values (for hours when there is no charging) for each EV station -- **expanded_data_all**
  * *list_names* - list of all EV stations
 
### User selection
* User should select the type of aggregation of values (average, sum, etc.), and the time period (monthly, daily, yearly aggregation)

### Dataset reshaping
* Here we select columns with multiple future and past values for **multi-step ahead forecasting** -- based on **lookback** and **future_steps**
* Past values = (t-1)
* Future values = (t+1)
* column **utilization(t+0)** represents the real time value, while other columns contain values for learning and forecasting

### ML dataset
* x dataset -- those columns with 't - X'
* y dataset -- columns with 't + X'
* Split dataset to train/validation/test set, reshape into 3D and create tensors
* DataLoaders - for bigger datasets, here we should select batch size

### Training and validation
* ML model is implemented in the same way as for short term multi-step ahead forecasting
  * we must input **input_size**, **hidden_size**, **num_stacked_layers**, **output_size**
  * training is done when validation loss stops decreasing
  * final step: testing and visualization
    * NOTE: we must input enough monthly/yearly data for model to learn good relationship between input and output dataset   
