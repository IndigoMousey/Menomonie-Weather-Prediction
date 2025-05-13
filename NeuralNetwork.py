import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class WeatherData(Dataset):
    def __init__(self, sequence_length=15,
                 test_days=30):  # sequence_length is how many previous days we use to predict each future day, holdout_days is how many days at the end will be in our test set
        self.sequence_length = sequence_length  # Save the sequence length
        self.test_days = test_days  # Save the number of test days

        df = pd.read_csv("Menomonie_Weather.csv")  # Load data
        df.dropna(inplace=True)  # Drop any rows with missing values
        df = df.drop(columns=["Date"])  # Drop the date column if it exists

        self.feature_names = df.columns.tolist()  # Save column names
        self.scaler = StandardScaler()  # Create a standard scaler
        self.scaled_data = self.scaler.fit_transform(df.to_numpy())  # Fit and transform the data
        data_tensor = torch.tensor(self.scaled_data, dtype=torch.float32)  # Convert scaled data to tensor

        X_list = []
        y_list = []
        for i in range(len(data_tensor) - sequence_length - test_days):  # Loop over training range
            X_seq = data_tensor[i: i + sequence_length].flatten()  # Flatten past sequence of features
            y_target = data_tensor[i + sequence_length][df.columns.get_loc("Min_Temp")], \
                data_tensor[i + sequence_length][df.columns.get_loc("Max_Temp")]  # Get y values
            X_list.append(X_seq)  # Add sequence to input list
            y_list.append(torch.tensor(y_target, dtype=torch.float32))  # Add y values to output list

        self.X = torch.stack(X_list)
        self.y = torch.stack(y_list)
        self.len = len(self.X)  # Store dataset length

    def __getitem__(self, item):  # Return one data item
        return self.X[item], self.y[item]  # Return x, y pair

    def __len__(self):  # Return dataset size
        return self.len

    def to_numpy(self):  # Convert dataset to NumPy arrays
        return self.X.numpy(), self.y.numpy()

    def get_holdout_set(self):  # Get test/holdout data
        data_tensor = torch.tensor(self.scaled_data, dtype=torch.float32)  # Convert to tensor again
        X_list = []  # Test X data
        y_list = []  # Test y data

        start_idx = len(data_tensor) - self.test_days - self.sequence_length  # Start of test set
        for i in range(start_idx, len(data_tensor) - self.sequence_length):  # Loop over test set
            X_seq = data_tensor[i: i + self.sequence_length].flatten()  # Create input sequence
            y_target = data_tensor[i + self.sequence_length][self.feature_names.index("Min_Temp")], \
                data_tensor[i + self.sequence_length][self.feature_names.index("Max_Temp")]  # Create y values
            X_list.append(X_seq)  # Append input
            y_list.append(torch.tensor(y_target, dtype=torch.float32))  # Append y values

        return torch.stack(X_list), torch.stack(y_list)  # Return stacked test inputs and outputs


class WeatherNet(nn.Module):
    def __init__(self, input_dim):  # Initialize with input size (Can change to be static but idk why you'd do that)
        super(WeatherNet, self).__init__()  # Inherit from nn.Module
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Final layer outputs two values (Min_Temp, Max_Temp)

    def forward(self, x):  # Define forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_weather_model(epochs=20, batch_size=16, lr=0.001, sequence_length=15):  # Training function
    dataset = WeatherData(sequence_length=sequence_length)  # Create training dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader for batching

    input_dim = dataset.X.shape[1]  # Get input dimension
    model = WeatherNet(input_dim=input_dim)  # Instantiate model

    criterion = nn.MSELoss()  # MSE loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Initialize loss to 0
        for X_batch, y_batch in data_loader:  # Loop over batches
            optimizer.zero_grad()  # Reset gradients
            preds = model(X_batch)  # Forward pass
            loss = criterion(preds, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()  # Add loss

        print(f"Epoch {epoch + 1}/{epochs}, MSE Loss: {running_loss / len(dataset):.4f}")  # Print epoch loss

    print("-" * 50)
    return model, dataset  # Return trained model and dataset


def evaluate_on_holdout(model, dataset: WeatherData):  # Evaluation function
    model.eval()  # Set model to evaluation mode
    X_test, y_true = dataset.get_holdout_set()  # Get test set

    with torch.no_grad():  # Disable gradient calculation
        y_pred = model(X_test)  # Get predictions

    scaler = dataset.scaler  # Get scaler
    feature_names = dataset.feature_names  # Get feature names
    idx_min = feature_names.index("Min_Temp")  # Get index of Min_Temp
    idx_max = feature_names.index("Max_Temp")  # Get index of Max_Temp

    def inverse_targets(y_scaled):  # Function to undo normalization
        y_full = np.zeros((len(y_scaled), len(feature_names)))  # Create zero matrix
        y_full[:, idx_min] = y_scaled[:, 0]  # Fill Min_Temp
        y_full[:, idx_max] = y_scaled[:, 1]  # Fill Max_Temp
        y_inversed = scaler.inverse_transform(y_full)  # Inverse transform
        return y_inversed[:, idx_min], y_inversed[:, idx_max]  # Return real values

    min_pred, max_pred = inverse_targets(y_pred.numpy())  # Inverse predicted
    min_true, max_true = inverse_targets(y_true.numpy())  # Inverse true

    min_rmse = np.sqrt(np.mean((min_pred - min_true) ** 2))  # Compute RMSE for Min_Temp
    max_rmse = np.sqrt(np.mean((max_pred - max_true) ** 2))  # Compute RMSE for Max_Temp

    print(f"Test Set Max_Temp MSE: {max_rmse:.4f}")  # Print Max_Temp RMSE
    print(f"Test Set Min_Temp MSE: {min_rmse:.4f}")  # Print Min_Temp RMSE

    days = np.arange(len(min_true))  # Create days array

    plt.figure(figsize=(12, 5))  # Start plot

    plt.subplot(1, 2, 1)  # First subplot
    plt.plot(days, min_true, label="Actual Min Temp", marker='o')  # Plot actual
    plt.plot(days, min_pred, label="Predicted Min Temp", linestyle="--", marker='x')  # Plot predicted
    plt.title("Min Temperature Forecast")  # Title
    plt.xlabel("Days")  # X-axis label
    plt.ylabel("Temperature")  # Y-axis label
    plt.legend()  # Show legend

    plt.subplot(1, 2, 2)  # Second subplot
    plt.plot(days, max_true, label="Actual Max Temp", marker='o')  # Plot actual
    plt.plot(days, max_pred, label="Predicted Max Temp", linestyle="--", marker='x')  # Plot predicted
    plt.title("Max Temperature Forecast")  # Title
    plt.xlabel("Days")  # X-axis label
    plt.ylabel("Temperature")  # Y-axis label
    plt.legend()  # Show legend

    plt.tight_layout()  # Adjust layout
    plt.show()

    return min_pred, min_true, max_pred, max_true


# Run everything
model, dataset = train_weather_model(epochs=500)  # Train model
evaluate_on_holdout(model, dataset)  # Evaluate model
