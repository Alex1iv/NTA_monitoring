import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import os
import joblib
from datetime import datetime
from keras.models import load_model

from utils.click_writer import DBWriter
from utils.models import get_predictions, get_intervals, split_sequences
from utils.logging_utils import get_logger

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


def main():
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
    df = db.import_data(query)#.set_index(['dt'])
    df["dt"] = df["dt"].dt.tz_localize(None)
    df = df.set_index(['dt']).rename_axis(None)
    
    df["minutes"] = (df.index.hour*60 + df.index.minute)
    df["sin_time"] = np.sin(2*np.pi * df["minutes"]/1440)
    df["cos_time"] = np.cos(2*np.pi * df["minutes"]/1440)
    df = df.drop(["minutes"], axis=1)

    
    scaler = joblib.load(os.path.join(config['PATH_MODELS'], 'scaler.gz'))
    model = load_model(os.path.join(config['PATH_MODELS'], 'model_LSTM.keras'))

    df['bytes_scaled']= scaler.transform(df[['total_trafic_Gbit']].values)
    df = df[["bytes_scaled", "sin_time", "cos_time"]]
    
    # split into samples
    n_steps = 3
    
    X_splitted, _ = split_sequences(
        X=df,
        Y=df['bytes_scaled'], 
        n_steps=n_steps
    )


    prediction = get_predictions(
        num_periods_to_forecast=5, 
        last_prediction=X_splitted[-1:,:], 
        last_index=df.index[-1], 
        scaler=scaler,
        model=model, 
        n_steps=3
    )
    
    mask = prediction.index < (
        datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')


    tmp=get_intervals(
        prediction[mask], 
        config['intervals']['ci_low'], 
        config['intervals']['ci_high']
    )
    
    tmp['feature_name']= 'total_bytes'
    
    tmp = tmp.reset_index().rename(columns={'index':'dt'})
    tmp = tmp[['dt', 'feature_name', 'ci_low','ci_high']]
    
    db.export_data(tmp)
    
if __name__ == "__main__":
    main()
