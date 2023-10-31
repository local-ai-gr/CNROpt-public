# On how to use the code "Analytical approach forecasting one station.ipynb"
This file explains all the steps in the code (also the data structures, preprocessing, etc.)

### Importing the necessary libraries, connecting to the database
* Connecting to the database and dropping of the unnecessary columns results in a dataframe **df**

### Data preparation
* **expand_data** function - Dataset is expanded with zero values (for hours when there is no charging) for each EV station -- **expanded_data_all**
  * *list_names* - list of all EV stations

### User selection
* User should select the type of aggregation of values (average, sum, etc.), and the time period (monthly, daily, yearly aggregation)
* **time_future**, **time_past** present future time period for forecasting and past time period for learning from data
* aggregation based on the user selection -- **df_agg**

### Implementation of the analytical methods
* 4 methods are implemented: Gompertz, polynomial, exponential, logarithmic function
  * **gompertz()**
  * **exponential()**
  * **polynomial()**
  * **logarithmic()**
* In each method, first necessary parameters are calculated (for example, sum of all time periods, etc.). Then parameters **a, b** are calculated from the equations.
  * **parameters_X** - list with resulting parameters **a, b**
* **error_function** - for estimating errors/correlation -- based on this value, the best method can be selected
  * the best method is the one with the largest value of this function
* Functions for the application of analytical methods (here parameters **a,b** are used): 
  * **use_gompertz()**
  * **use_exp()**
  * **use_poly()**
  * **use_log()**
* Based on the error value, the best method is used for forecasting and visualizations

