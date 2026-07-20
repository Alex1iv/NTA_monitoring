import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import os
import joblib


from utils.click_writer import DBWriter
from utils.logging_utils import get_logger
from utils.models import (
    TCNRegressor, get_tch_predictions, split_N_sequences, get_intervals)

os.environ["USE_NNPACK"] = "0"

import torch
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
#torch.backends.nnpack.enabled = False
torch.backends.nnpack.set_flags(False)

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent 

# Load config
with open(BASE_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

# Load credentials
with open(BASE_DIR / "secrets.yaml") as f:
    cred = yaml.safe_load(f)


logger = get_logger(    
    log_dir=BASE_DIR / config["LOG_DIR"],
    file_name="backend.log",
    enabled=config["logging"]
)

# fix random seed in Numpy
np.random.seed(config['random_seed'])

# fix random seed in PyTorch (CPU & all GPUs)
torch.manual_seed(config['random_seed'])
torch.cuda.manual_seed_all(config['random_seed'])

device = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)

def main():    
        
    db = DBWriter(
        config['collector_db'],
        username=cred['clickhouse_user'],
        password=cred['clickhouse_pass'], 
        logger=None)
    
    query = """
    WITH 
        toTimeZone(minute,'Europe/Moscow') AS dt, 
        ROUND(sum(bytes) /8 * 1e-9, 4) AS Gbit,
        if(country_code ='RU', 'RU', 'F') AS origin
    
    SELECT 
        dt, 
        Gbit, 
        origin
    FROM traffic_country_1m
    WHERE dt > now() - INTERVAL 10 minutes
        AND country_side='src'
        AND source_id='netflow'
        AND country_basis='asn'
        AND direction='in'
    GROUP BY dt, 
        origin
    ORDER BY dt
    """
    df = db.import_data(query)
    
    df["dt"] = df["dt"].dt.tz_localize(None)
    df = df.set_index('dt').rename_axis(None)
    df = df[df['origin']=='RU'][['Gbit']]
    
    minutes = (df.index.hour*60 + df.index.minute)
    df["sin_time"] = np.sin(2*np.pi * minutes/1440)
    df["cos_time"] = np.cos(2*np.pi * minutes/1440)

    scaler = joblib.load(os.path.join(
        config['PATH_MODELS'], 
        config['intervals_country_ru']['scaler_filename']))
  
    checkpoint = torch.load(
        os.path.join(
            config['PATH_MODELS'], 
            config['intervals_country_ru']['chkpoint_filename']),
        map_location="cpu"
    )

    model = TCNRegressor(
        in_channels=checkpoint["n_features"],
        out_channels=checkpoint["out_channels"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    df['bytes_scaled']= scaler.transform(df[['Gbit']]) #df[['Gbit']].values
    df = df[["bytes_scaled", "sin_time", "cos_time"]]
    
    # split into samples
    X_splitted, _ = split_N_sequences(
        df.values,
        df['bytes_scaled'],
        n_steps=3
    )
    last_prediction = torch.as_tensor(X_splitted[-1:], dtype=torch.float32)
    
    predictions = get_tch_predictions(
        num_periods_to_forecast=config['pred_period_minutes'],
        last_prediction=last_prediction,
        last_index=df.index[-1],
        scaler=scaler,
        model=model #device=device
    )
    
    intervals=get_intervals(
        predictions,
        config['intervals_country_ru']['ci_low'], 
        config['intervals_country_ru']['ci_high']
    )
    
    intervals['ci_low'] = np.where(
        intervals['ci_low'] < config['intervals_country_ru']['ci_minimum'], 
        config['intervals_country_ru']['ci_minimum'], 
        intervals['ci_low']
    )
    
    intervals['feature_name']= config['intervals_country_ru']['feature_name']
    
    intervals = intervals.reset_index().rename(columns={'index':'dt'})
    intervals = intervals[['dt', 'feature_name', 'ci_low','ci_high']]
    
    temp = intervals.copy()
    temp['feature_name'] = "country_F" #config['intervals_country_ru']['feature_name']
    intervals = pd.concat([intervals, temp]).sort_values(['dt', 'feature_name'])
    
    #intervals = intervals[intervals['dt']<"2026-07-18 07:02:00"]
    #print(intervals[['ci_low','ci_high']].sum())
    db.export_data(intervals)
    
if __name__ == "__main__":
    main()
