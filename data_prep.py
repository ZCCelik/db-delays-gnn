import json
from pathlib import Path
import pandas as pd
import torch
from datasets import load_dataset
import numpy as np
from torch_geometric.utils import to_undirected


OUT = Path("data/processed")
STATION_MAP_PATH = OUT / "station_to_id.json"
EDGE_INDEX_PATH = OUT / "edge_index.pt"
TRAIN_PARQUET   = OUT / "df_train.parquet"
CANCELLED_PARQUET = OUT / "df_cancelled.parquet"

def load_raw():
    dataset_dict = load_dataset("piebro/deutsche-bahn-data", data_dir="monthly_processed_data", streaming=True)
    return pd.DataFrame(list(dataset_dict['train'].take(1000000)))

def clean(df): 
    df = df[df["train_line_ride_id"] != ""]
    
    assert df["xml_station_name"].notna().all(), "Found rows with missing xml_station_name — investigate before continuing"
        
    return df
    

def build_station_mapping(df):
    all_stations = sorted(df["xml_station_name"].unique())
    station_to_id = {name: i for i, name in enumerate(all_stations)}
    
    with open(STATION_MAP_PATH, "w") as f:
        json.dump(station_to_id, f, indent=2, ensure_ascii=False)
        
    return station_to_id

def split(df):
    df_train = df[df.is_canceled == False]    
    df_cancelled = df[df.is_canceled == True]
    
    return df_train, df_cancelled

    
def transform_delays(df_train):
    df_train = df_train.copy()
    df_train["delay_in_min"] = df_train["delay_in_min"].clip(lower=0)
    df_train["delay_in_min"] = np.log1p(df_train["delay_in_min"])
    return df_train
    

def build_edge_index(df, station_to_id):
    station_pairs = []
    grouped = df.sort_values("train_line_station_num").groupby(["train_line_ride_id", df["time"].dt.date])
    
    for _, group in grouped:
        stations = group["xml_station_name"].tolist()
        station_pairs.extend(zip(stations[:-1], stations[1:]))

    unique_pairs = set(station_pairs)

    id_pairs = [(station_to_id[a], station_to_id[b]) for a, b in unique_pairs]

    edge_index = torch.tensor(id_pairs, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    return edge_index
    
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df = clean(df)
    station_to_id = build_station_mapping(df)
    df_train, df_cancelled = split(df)
    df_train = transform_delays(df_train)
    edge_index = build_edge_index(df, station_to_id)
    df_train.to_parquet(TRAIN_PARQUET)
    df_cancelled.to_parquet(CANCELLED_PARQUET)
    torch.save(edge_index, EDGE_INDEX_PATH)
    
if __name__ == "__main__":
    main()