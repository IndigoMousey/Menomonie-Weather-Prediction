import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class WeatherData(Dataset):
    def __init__(self):
        # Training set
        df = pd.read_csv("Menomonie_Weather.csv")
        df.dropna(inplace=True)
        X = torch.tensor(df.iloc[:, 1:].to_numpy(), dtype=torch.float)
        y = torch.tensor(df.iloc[1:][["Avg_Temp", "Precipitation"]].to_numpy(), dtype=torch.float)

        scaler = StandardScaler()
        X_scaled = scaler.transform(X)

        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X), np.array(self.y)


class BenchPress(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(BenchPress, self).__init__()

        # Single hidden layer neural network with 20 nodes in hidden layer
        self.in_to_h1 = nn.Linear(10, 50)
        self.h1_to_h2 = nn.Linear(50, 10)
        self.h2_to_out = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        return self.h2_to_out(x)


def trainNN(epochs=5, batch_size=16, lr=0.001):
    # Load the Dataset
    ld = WeatherData()

    # Create data loader
    data_loader = DataLoader(ld, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create an instance of the NN
    bp = BenchPress()

    # Mean Square Error loss function
    mse_loss = nn.MSELoss(reduction='sum')

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(bp.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        for _, data in enumerate(data_loader, 0):
            x, y = data

            optimizer.zero_grad()

            output = bp(x)

            loss = mse_loss(output.view(-1), y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch} of {epochs} MSE (Train): {running_loss / len(ld)}")
        running_loss = 0.0
        with torch.no_grad():
            output = bp(ld.X_valid).view(-1)
        print(f"Epoch {epoch} of {epochs} MSE (Validation): {torch.mean((output - ld.y_valid) ** 2.0)}")
        print("-" * 50)
    return bp


trainNN(epochs=10)
