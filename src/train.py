from model import DeutscheBahnGNN
import json
from pathlib import Path
import pandas as pd
import torch
import numpy as np


DATA = Path("data/processed")

df_train = pd.read_parquet(DATA / "df_train.parquet")

with open(DATA / "station_to_id.json") as f:
    station_to_id = json.load(f)

edge_index = torch.load(DATA / "edge_index.pt")

df_train["station_id"] = df_train["xml_station_name"].map(station_to_id)

start_time = df_train["time"].min().floor("h")
end_time = df_train["time"].max().ceil("h")
hours = pd.date_range(start=start_time, end=end_time, freq="h")

avg_delay_1h = np.full((len(hours), len(station_to_id)), np.nan)
for i, h in enumerate(hours):
    rides_in_h = df_train[ (df_train["time"] >= h - pd.Timedelta("1h")) 
                        & (df_train["time"] < h)]
    means = rides_in_h.groupby("station_id")["delay_in_min"].mean()
    avg_delay_1h[i, means.index] = means.values

avg_delay_6h = np.full((len(hours), len(station_to_id)), np.nan)
for i, t in enumerate(hours):
    window = df_train[(df_train["time"] >= t - pd.Timedelta("6h")) &
                   (df_train["time"] <  t)]
    means = window.groupby("station_id")["delay_in_min"].mean()
    avg_delay_6h[i, means.index] = means.values

avg_delay_24h = np.full((len(hours), len(station_to_id)), np.nan)
for i, t in enumerate(hours):
    window = df_train[(df_train["time"] >= t - pd.Timedelta("24h")) &
                   (df_train["time"] <  t)]
    means = window.groupby("station_id")["delay_in_min"].mean()
    avg_delay_24h[i, means.index] = means.values
    

number_of_rides = np.full((len(hours), len(station_to_id)), 0, dtype=int)
for i, h in enumerate(hours):
    rides_in_h = df_train[ (df_train["time"] >= h - pd.Timedelta("1h")) 
                        & (df_train["time"] < h)]
    sum_of_trains = rides_in_h.groupby("station_id").size()
    number_of_rides[i, sum_of_trains.index] = sum_of_trains.values
    
hour_sin = np.full((len(hours), len(station_to_id)), 0.0)
hour_cos = np.full((len(hours), len(station_to_id)), 0.0)

for i, t in enumerate(hours):
    hour_sin[i, :] = np.sin(2 * np.pi * t.hour / 24)
    hour_cos[i, :] = np.cos(2 * np.pi * t.hour / 24)
    
day_sin = np.full((len(hours), len(station_to_id)), 0.0)
day_cos = np.full((len(hours), len(station_to_id)), 0.0)

for i, t in enumerate(hours):
    day_sin[i, :] = np.sin(2 * np.pi * t.day_of_week / 7)
    day_cos[i, :] = np.cos(2 * np.pi * t.day_of_week / 7)
    
x_snapshots = np.stack([
    avg_delay_1h, avg_delay_6h, avg_delay_24h,
    number_of_rides,
    hour_sin, hour_cos, day_sin, day_cos], axis=-1)

x_snapshots = np.nan_to_num(x_snapshots, nan=0.0) 

ride_snapshot_time = df_train["time"].dt.floor("h")

hours_to_snapshot_id = {t: i for i, t in enumerate(hours)}

ride_to_snapshot_id = ride_snapshot_time.map(hours_to_snapshot_id).to_numpy()

ride_station_id = df_train["station_id"].to_numpy()

ride_targets = df_train["delay_in_min"].to_numpy()

train_type_encoded = pd.get_dummies(df_train["train_type"]).astype(float)

station_num_normalized = (df_train["train_line_station_num"] / df_train["train_line_station_num"].max())

ride_features = pd.concat([train_type_encoded, station_num_normalized], axis=1).to_numpy()

#convert to tensors to use it in the model
x_snapshots = torch.tensor(x_snapshots, dtype=torch.float32)

ride_to_snapshot_id = torch.tensor(ride_to_snapshot_id, dtype=torch.long)

ride_station_id = torch.tensor(ride_station_id, dtype=torch.long)

ride_targets = torch.tensor(ride_targets, dtype=torch.float32)

ride_features = torch.tensor(ride_features, dtype=torch.float32)

print(x_snapshots.shape)      
print(ride_features.shape) 

deutscheBahnGNN = DeutscheBahnGNN(8, 32, 32, 54, 32)

optimizer = torch.optim.Adam(deutscheBahnGNN.parameters(), lr=0.01)

loss_fn = torch.nn.MSELoss()

batch_size = 512
num_epochs = 5

for epoch in range(num_epochs):
    df_train["snapshot_time"] = df_train.apply(lambda x: math.ceil)
    for i in range(5):
        snapshot_id = ride_to_snapshot_id[i]
        x = x_snapshots[snapshot_id]

        pred = deutscheBahnGNN(x, edge_index, ride_station_id[i:i+1], ride_features[i:i+1])
        loss = loss_fn(pred, ride_targets[i:i+1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss.item())
