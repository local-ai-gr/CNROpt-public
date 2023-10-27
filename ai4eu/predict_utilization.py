def predict_utilization(input):
  import json
  import pandas as pd
  import numpy as np
  from datetime import datetime

  import torch
  from torch.utils.data import Dataset, DataLoader
  import torch.nn as nn
  import torch.optim as optim
  from statsmodels.tsa.arima.model import ARIMA
  from sklearn.metrics import mean_squared_error
  from torch.utils.data import DataLoader
  from torch.utils.data import Dataset

  from copy import deepcopy as dc
  def prepare_dataframe_for_lstm(df, n_steps):
      df = dc(df)
      for i in range(1, n_steps+1):
          df[f'util(t-{i})'] = df['utilization'].shift(i)
      return df

  class TimeSeriesDataset(Dataset):
      def __init__(self, X, y):
          self.X = X
          self.y = y
      def __len__(self):
          return len(self.X)
      def __getitem__(self, i):
          return self.X[i], self.y[i]

  class LSTMModel(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, output_size):
          super(LSTMModel, self).__init__()
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
          self.fc = nn.Linear(hidden_size, output_size)

      def forward(self, x, h0, c0):
          h0 = torch.zeros(num_layers,  hidden_size).to(device)
          c0 = torch.zeros(num_layers,  hidden_size).to(device)
          out, (hn, cn) = self.lstm(x, (h0, c0))
          out = self.fc(out)
          return out, (hn, cn)
      
  data = pd.DataFrame(list(input))

  data.columns = ['utilization']
  lookback = 128
  shifted_df = prepare_dataframe_for_lstm(data, lookback)
  for i in range(1, lookback+1):
    shifted_df[f'util(t-{i})'] = shifted_df[f'util(t-{i})'].fillna(0.0)
  shifted_df_as_np = shifted_df.to_numpy()

  X = shifted_df_as_np[:, 1:]
  y = shifted_df_as_np[:, 0]

  X = dc(np.flip(X, axis=1))
  X[0]

  train_size = int(len(data) * 0.8)
  val_size = int(len(data) * 0.1)

  X_train = X[:train_size]
  X_val = X[train_size:train_size+val_size]
  X_test = X[train_size+ val_size:]

  y_train = y[:train_size]
  y_val = y[train_size:train_size+val_size]
  y_test = y[train_size+ val_size:]

  X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

  X_train = X_train.reshape((-1, lookback))
  X_val = X_val.reshape((-1, lookback))
  X_test = X_test.reshape((-1, lookback))

  y_train = y_train.reshape((-1, 1))
  y_val = y_val.reshape((-1, 1))
  y_test = y_test.reshape((-1, 1))

  X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

  X_train = torch.tensor(X_train).float()
  y_train = torch.tensor(y_train).float()
  X_val = torch.tensor(X_val).float()
  y_val = torch.tensor(y_val).float()
  X_test = torch.tensor(X_test).float()
  y_test = torch.tensor(y_test).float()

  X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

  train_dataset = TimeSeriesDataset(X_train, y_train)
  val_dataset = TimeSeriesDataset(X_train, y_train)

  batch_size = 128

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader, Dataset
  import pandas as pd
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  input_size = train_dataset.X.size(1)  # Number of features in the input data
  hidden_size = 250  # Number of units in the LSTM hidden layer
  num_layers = 4  # Number of LSTM layers
  output_size = 1  # Number of output features (energy consumption and utilization)
  learning_rate = 0.0005
  num_epochs = 150
  batch_size = 128

  # Create data loaders

  model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Initialize hidden states and cell states on the specified device
  h0 = torch.zeros(num_layers,  hidden_size).to(device)
  c0 = torch.zeros(num_layers,  hidden_size).to(device)

  # Placeholder for the best validation accuracy
  best_val_loss = float('inf')  # Initialize with a large value
  best_model_state = None

  # Training loop
  for epoch in range(num_epochs):

      h0 = torch.zeros(num_layers,  hidden_size).to(device)
      c0 = torch.zeros(num_layers,  hidden_size).to(device)
      model.train()
      total_loss = 0
      for batch_idx, (features, labels) in enumerate(train_loader):

          features, labels = features.to(device), labels.to(device)  # Move data to the specified device
          h0 = h0.to(device)
          c0 = c0.to(device)
          # Forward pass with hidden states
          #print(features.shape)
          outputs, (h0, c0) = model(features, h0.detach(), c0.detach())
          loss = criterion(outputs, labels)

          # Backpropagation and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          total_loss += loss.item()

      average_train_loss = total_loss / len(train_loader)

      # Validation
      model.eval()
      val_loss = 0
      predictions = []
      with torch.no_grad():
          # Initialize validation hidden states and cell states with the final states of the training set
          h0_val = h0.detach().clone()
          c0_val = c0.detach().clone()

          for batch_idx, (features, labels) in enumerate(val_loader):
              features, labels = features.to(device), labels.to(device)

              # Forward pass with hidden states
              outputs, (h0_val, c0_val) = model(features, h0_val, c0_val)
              loss = criterion(outputs, labels)
              val_loss += loss.item()

              # store predictions
              predictions.extend(outputs.cpu().numpy())

      average_val_loss = val_loss / len(val_loader)
      # keep the model with the best performance in the validation set
      if average_val_loss < best_val_loss:
          best_val_loss = average_val_loss
          best_model_state = model.state_dict()
          best_predictions = predictions

      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

  print('Training finished.')
  print(best_val_loss)

  model.load_state_dict(best_model_state)

  with torch.no_grad():
      predicted, (h0_val,c0_val) = model(X_test.to(device),h0_val, c0_val)
      #print((h0_val,c0_val))


  new_in = predicted[-129:-1]
  new_in = torch.unsqueeze(torch.squeeze(new_in, 1),0)
  predicted, (h0_val,c0_val) = model(new_in,h0_val, c0_val)
  new_in = torch.cat((new_in,predicted),1)
  new_in = new_in[0][1:]
  new_in = torch.unsqueeze(new_in,0)

  prediction = []
  for i in range(200):
    predicted, (h0_val,c0_val) = model(new_in,h0_val, c0_val)
    new_in = torch.cat((new_in,predicted),1)
    new_in = new_in[0][1:]
    new_in = torch.unsqueeze(new_in,0)
    prediction.append(round(predicted.item(), 2))
  print('prediction completed')
  return (prediction)