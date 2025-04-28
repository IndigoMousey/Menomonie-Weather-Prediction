import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class LiftData(Dataset):
    def __init__(self):
        # Training set
        df = pd.read_csv("TrainClean.csv")
        self.X = torch.tensor(df.iloc[:, 1:-1].to_numpy(), dtype=torch.float)
        self.y = torch.tensor(df.iloc[:, -1].to_numpy(), dtype=torch.float)
        self.len = len(df)

        # Validation set
        df_valid = pd.read_csv("ValidClean.csv")
        self.X_valid = torch.tensor(df_valid.iloc[:, 1:-1].to_numpy(), dtype=torch.float)
        self.y_valid = torch.tensor(df_valid.iloc[:, -1].to_numpy(), dtype=torch.float)

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
    ld = LiftData()

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


# SVR Performance on Validation set (MSE): 634.6198690427688
# and R^2 = 0.78427481498
# SVR had C=50.0, default gamma, and normalized features
trainNN(epochs=10)