# On how to use the code "Clustering charging stations.ipynb"
This file explains all the steps in the code (also the data structures, preprocessing, etc.)

### Importing the necessary libraries, connecting to the database
* Connecting to the database and dropping of the unnecessary columns results in a dataframe **df**
* We have to include information on coordinates for each station -- EV stations location.csv

### Data preparation
* **expand_data** function - Dataset is expanded with zero values (for hours when there is no charging) for each EV station
  * *list_names* - list of all EV stations
* Since we don't want to forecast for whole time period, we select time frame -- **time_start, time_end**
  * for loop: slice part of dataframe for each station depending on the time frame
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
  * There are **280 features** and **2 labels** -- saved in variables real_data_filtered
  * Data loaders - inputs are real_data_filtered and **batch_size = 1024**
 
### Loading of trained model
* **LSTM_TimeSeries** class -- the same class should be used as for the training of the model
  * inputs are: **input_size = number of features, hidden_size, layer_size, output_size = 2**
* **calculate_mse** function - for losses
* Loading of the weights of the trained model - if the loading of pre-trained model was successful, the output should be _"All keys matched successfully."_
* criterion and optimizer should be defined

### Predictions
* Predictions are done in the same way as for the testing of the model - results are stored in **pred_consumption_test** and **pred_utilization_test**
  * Forecasting of both consumption and utilization (even though the clustering is done only on utilization data)
 
### Manual selection of time range and location
* **time_start, time_end** - variables that define time range
* **lat_init, long_init** - coordinates of randomly chosen point
* **radius** - radius from the randomly chosen where we cluster stations
* **find_in_radius** function - the function that selects charging stations that are inside the chosen radius and returns a list of IDs (**ids_in_radius**)
* **create_new_df** function - uses list of selected IDs (**ids_in_radius**) and creates a new dataframe with predicted utilization in selected time frame (**selected_stations**)
  * dimensions: T x N, where N is number of charging stations inside the chosen radius

### Clustering
* user should define **num_cluster**
* **K_Means** class - for clustering of charging stations
  * Since we want to cluster based on busy hour values, user should ignore really small values
    ```python
    data_init[data_init < 0.001] = np.nan
    ```
  * **max_mean** is the maximum value between all average values, it is used for creating equal ranges
    * For example, if **num_cluster = 5** and **max_mean = 10**, the first cluster contains stations with average values between 0 and 2 (1 * max_mean / num_cluster), second cluster contains stations with average values between 2 and 4 (2 * max_mean / num_cluster) and so on. The idea behind this algorithm is to cluster based on these ranges.
  * Centroids are chosen based on ranges - first we select all stations with mean values in chosen ranges and then the first station is set as a centroid.
    * If there are no stations in one of the ranges, cluster is left empty and number of cluster is saved in the list **list_uncompleted** -- it will be later used for plotting and selection of new centroids
  * After the centroids are chosen, the distance is calculated between each station and each centroid.
    ```python
    distances = [np.linalg.norm(data[featureset]-self.centroids[centroid]) for centroid in self.centroids]
    ```
    * Then the minimal distance is found and to selected centroid ID of the station is added.
  * New centroids are calculated as the mean of all clustered stations - the algorithm ends when there is no change between old and new centroids.

### Elbow method
* before performing the clustering, user should determine the optimal number of clusters for the selected dataset -- this is done using the 'elbow method'
* We track the decrease in the model inertia (i.e., error) and determine the optimal number of clusters as the number where there is a significant change (that is the elbow)
* optimal number of clusters is stored as **optim_cluster**

### Visualization
* **plot_stations** function - uses clustered IDs, information on coordinates, initial point (**init_lat, init_long**) and list of empty clusters for plotting
```python
plot_stations(model.classifications, geoloc_pom, lat_init, long_init, model.empty)
```


