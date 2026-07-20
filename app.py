import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import os
import joblib
from datetime import datetime
#from keras.models import load_model

from utils.click_writer import DBWriter
#from utils.models import get_predictions, get_intervals, split_sequences
from utils.logging_utils import get_logger

import torch
#torch.backends.nnpack.enabled = False
torch.backends.nnpack.set_flags(False)

from utils.models import (
    TCNRegressor, get_tch_predictions, split_N_sequences, get_intervals)

import warnings
warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent  #Path("D:/IDE/NTA_monitoring") #


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

#fix random seed
np.random.seed(config['random_seed'])
# 3. PyTorch total seed (CPU & all GPUs)
torch.manual_seed(config['random_seed'])
torch.cuda.manual_seed_all(config['random_seed'])

device = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)

def main():    
    # df = (pd.read_csv('./data/traffic_dashboard_1m.zip', 
    #     index_col=0, parse_dates=['dt'])
    # ).set_index('dt').rename_axis(None)
    
    db = DBWriter(
        config['collector_db'],
        username=cred['clickhouse_user'],
        password=cred['clickhouse_pass'], 
        logger=None)
    
    query = """
    SELECT 
        toTimeZone(minute,'Europe/Moscow') AS dt,
        --source_id,
        total_bytes /8 *1e-9 AS total_trafic_Gbit
    FROM traffic_dashboard_1m
    WHERE dt > now() - INTERVAL 10 minutes
        AND source_id='netflow'
    ORDER BY dt
    """
    df = db.import_data(query)
    df["dt"] = df["dt"].dt.tz_localize(None)
    df = df.set_index('dt').rename_axis(None)

    df["minutes"] = (df.index.hour*60 + df.index.minute)
    df["sin_time"] = np.sin(2*np.pi * df["minutes"]/1440)
    df["cos_time"] = np.cos(2*np.pi * df["minutes"]/1440)
    df = df.drop(["minutes"], axis=1)

    scaler = joblib.load(os.path.join(config['PATH_MODELS'], 'scaler.gz'))
  
    checkpoint = torch.load(
        os.path.join(config['PATH_MODELS'], "tcn_checkpoint.pt"),
        map_location="cpu"
    )

    model = TCNRegressor(
        in_channels=checkpoint["n_features"],
        out_channels=checkpoint["out_channels"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval() #Switch the model to evaluation mode
    
    df['bytes_scaled']= scaler.transform(df[['total_trafic_Gbit']].values)
    df = df[["bytes_scaled", "sin_time", "cos_time"]]
    #print(df.sum())
    
    # split into samples
    X_splitted, _ = split_N_sequences(
        df.values,
        df['bytes_scaled'],
        n_steps=3
    )
    last_prediction = torch.as_tensor(X_splitted[-1:], dtype=torch.float32)
    
    predictions = get_tch_predictions(
        num_periods_to_forecast=config['pred_period_minutes'],
        last_prediction=last_prediction, #.unsqueeze(0)
        last_index=df.index[-1],
        scaler=scaler,
        model=model #device=device
    )
    
    intervals=get_intervals(
        predictions,
        config['intervals']['ci_low'], 
        config['intervals']['ci_high']
    )
    
    intervals['feature_name']= 'total_bytes'
    
    intervals = intervals.reset_index().rename(columns={'index':'dt'})
    intervals = intervals[['dt', 'feature_name', 'ci_low','ci_high']]
    
    intervals['ci_low'] = np.where(
        intervals['ci_low']<config['intervals']['ci_minimum'], 
        config['intervals']['ci_minimum'], 
        intervals['ci_low']
    )

    
    db.export_data(intervals)
    
if __name__ == "__main__":
    main()
