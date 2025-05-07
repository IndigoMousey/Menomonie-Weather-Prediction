import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class WeatherData(Dataset):
    def __init__(self, sequence_length=15,
                 test_days=30):  # sequence is how many days to use to predict each future day, test days is how many days in our test set
        self.sequence_length = sequence_length
        self.test_days = test_days

        df = pd.read_csv("Menomonie_Weather.csv")  # read in data
        df.dropna(
            inplace=True)  # delete rows with missing data. There are 57 such rows in the first 1937 days (About 3% of days or 1 day per month)
        df = df.drop(columns=["Date"])  # delete the date column. Will throw an error if date column doesnt exist

        self.feature_names = df.columns.tolist()
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(df.to_numpy())
        data_tensor = torch.tensor(self.scaled_data, dtype=torch.float32)

        X_list = []
        y_list = []
        for i in range(len(data_tensor) - sequence_length - test_days):
            X_seq = data_tensor[i: i + sequence_length].flatten()
            y_target = data_tensor[i + sequence_length][df.columns.get_loc("Avg_Temp")], \
                data_tensor[i + sequence_length][df.columns.get_loc("Precipitation")]
            X_list.append(X_seq)
            y_list.append(torch.tensor(y_target, dtype=torch.float32))

        self.X = torch.stack(X_list)
        self.y = torch.stack(y_list)
        self.len = len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return self.X.numpy(), self.y.numpy()

    def get_holdout_set(self):
        data_tensor = torch.tensor(self.scaled_data, dtype=torch.float32)
        X_list = []
        y_list = []

        start_idx = len(data_tensor) - self.test_days - self.sequence_length
        for i in range(start_idx, len(data_tensor) - self.sequence_length):
            X_seq = data_tensor[i: i + self.sequence_length].flatten()
            y_target = data_tensor[i + self.sequence_length][
                self.feature_names.index("Avg_Temp")
            ], data_tensor[i + self.sequence_length][
                self.feature_names.index("Precipitation")
            ]
            X_list.append(X_seq)
            y_list.append(torch.tensor(y_target, dtype=torch.float32))

        return torch.stack(X_list), torch.stack(y_list)


class WeatherNet(nn.Module):
    def __init__(self, input_dim):
        super(WeatherNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Predicts [Avg_Temp, Precipitation]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_weather_model(epochs=20, batch_size=16, lr=0.001, sequence_length=15):
    dataset = WeatherData(sequence_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]  # = 15 * num_features
    model = WeatherNet(input_dim=input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, MSE Loss: {running_loss / len(dataset):.4f}")

    print("-" * 50)
    return model, dataset


def evaluate_on_holdout(model, dataset: WeatherData):
    model.eval()
    X_test, y_true = dataset.get_holdout_set()

    with torch.no_grad():
        y_pred = model(X_test)

    scaler = dataset.scaler
    feature_names = dataset.feature_names
    idx_temp = feature_names.index("Avg_Temp")
    idx_precip = feature_names.index("Precipitation")

    def inverse_targets(y_scaled):
        y_full = np.zeros((len(y_scaled), len(feature_names)))
        y_full[:, idx_temp] = y_scaled[:, 0]
        y_full[:, idx_precip] = y_scaled[:, 1]
        y_inversed = scaler.inverse_transform(y_full)
        return y_inversed[:, idx_temp], y_inversed[:, idx_precip]

    temp_pred, precip_pred = inverse_targets(y_pred.numpy())
    temp_true, precip_true = inverse_targets(y_true.numpy())

    temp_mse = np.mean((temp_pred - temp_true) ** 2)
    precip_mse = np.mean((precip_pred - precip_true) ** 2)

    print(f"Holdout (last 30 days) MSE:")
    print(f"  Avg_Temp:       {temp_mse:.2f}")
    print(f"  Precipitation:  {precip_mse:.2f}")

    # Plot results
    days = np.arange(len(temp_true))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(days, temp_true, label="Actual Avg Temp", marker='o')
    plt.plot(days, temp_pred, label="Predicted Avg Temp", linestyle="--", marker='x')
    plt.title("Avg Temperature Forecast")
    plt.xlabel("Days")
    plt.ylabel("Temperature")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(days, precip_true, label="Actual Precipitation", marker='o')
    plt.plot(days, precip_pred, label="Predicted Precipitation", linestyle="--", marker='x')
    plt.title("Precipitation Forecast")
    plt.xlabel("Days")
    plt.ylabel("Precipitation")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return temp_pred, temp_true, precip_pred, precip_true


# Run everything
if __name__ == "__main__":
    model, dataset = train_weather_model(epochs=20)
    evaluate_on_holdout(model, dataset)
