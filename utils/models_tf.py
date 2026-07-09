import numpy as np
import pandas as pd
from keras.models import Model

import tensorflow as tf
# from keras.models import Sequential
from keras.layers import LSTM, Dense
#from utils.reader_config import config_reader
# Import parameters
#config = config_reader('../config/config.json')
PATH_FIGURES = "../figures/"

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

def split_sequences(X:np.array, Y:np.array, n_steps:int)->np.array:
    """ Split a univariate sequence into samples

    Args:
        sequence (np.array): time series (1 feature)
        n_steps (int): lag

    Returns:
        np.array: _description_
    """    
    X_out, y_out = list(), list()
 
    for i in range(len(X)):
        # find the end of this pattern
        end_ix = i + n_steps
        
        # check if we are beyond the sequence
        if end_ix >=  len(X): #if end_ix >=  len(X)-1:
            break
        
        # gather input and output parts of the pattern
        # seq_x, seq_y = X[i:end_ix], Y[end_ix]
        
        X_out.append(X[i:end_ix])
        y_out.append(Y[end_ix])
  
    return np.array(X_out), np.array(y_out)

class ModelLSTM(Model):
    """ Model inherits the LSTM class from Keras.
    Параметры:
    ----------
    Model - data
   
    """
    def __init__(self, data, learning_rate):
    
        super().__init__()
        # ------- Parameters ------------
        _, self.n_timesteps, self.n_channels = data.shape
        self.learning_rate = learning_rate
        # -------- Model layers ----------------
        self.input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels)) 
      
        x = tf.keras.layers.LSTM(
            units=64, 
            return_sequences=True,     
            dropout=0.2,
            #recurrent_dropout=0.2
            )(x)
        
        x = tf.keras.layers.LSTM(
            units=32,    
            dropout=0.2,
            #recurrent_dropout=0.2
            )(x)
        x = tf.keras.layers.Dense(units=6)(x)
        self.output_channels = tf.keras.layers.Dense(units=1)(x)
        
        print(f"input_shape = {(self.n_timesteps, self.n_channels)}\noutput_units = {(self.output_channels)}")

    def build_model(self):
        """Initiate the model
        """
        model = tf.keras.Model(
            inputs=self.input_channels,
            outputs=self.output_channels,
            name="model_LSTM"
        )        
        model.summary()
        
        #--------------
        # compile model
        #--------------
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        
        model.compile(
            loss = tf.keras.losses.Huber(), #= 'mse', 
            metrics = ['mae', 'mse'], 
            optimizer = optimizer
        )

        return model
    

def callbacks(
    monitor,
    factor,
    mode,
    restore_best_weights,
    patience,
    stop_patience,
    verbose,
    learning_rate,
    model_name:str=None
    ):
    """Model training setup function

    Args:
        min_lr (_float_): lower boundary of the learning rate to stop training
        monitor (str) - metric name 
        mode (str)- modes {"auto", "min", "max"}. Max - stop training if the metric doesn't improve
        reduce_patience (_int_): number of epochs to evaluate the learning rate improvement
        stop_patience (_int_):  number of epochs before training will be ended
        path_models (_str_): path to save the model
        save_best_only (bool): If True then saves the model with the best metric.
    """
    
     
    
    # Save the best model
    # checkpoint = ModelCheckpoint(
    #     filepath=os.path.join(config.path_models + model_name + '.hdf5'), 
    #     #filepath=path,
    #     monitor=config.monitor, 
    #     verbose=config.verbose, 
    #     mode=config.mode, 
    #     save_best_only=True, 
    #     #save_weights_only=True
    # )
    # checkpoint = ModelCheckpoint(
    #     filepath= '../models/model_LSTM.hdf5', #os.path.join(config.PATH_MODEL + model_name +'.hdf5'), 
    #     monitor=config.monitor, 
    #     verbose=1, 
    #     mode=config.mode, 
    #     save_best_only=True
    # )


    # stop training if the metric doesn't improve
    earlystop = EarlyStopping(
        monitor=monitor, 
        mode=mode, 
        patience=stop_patience, 
        restore_best_weights=restore_best_weights
    )

    # reduce the learning rate if the metric doesn't improve
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor, 
        #mode=mode,  
        factor=factor, 
        patience=patience,  # might be 10
        verbose=verbose, 
        min_lr=learning_rate/1000
    )
    
    return [#checkpoint, 
            earlystop, reduce_lr]
    
    
import matplotlib.pyplot as plt


def plot_history_regr(history:dict=None, model_name:str=None, plot_counter:int=None):
    """Training history visualization
    
    Аргументы:
    history (keras.callbacks.History) - Training history data,
    model_name (str) - figure title. Use: model.name
    plot_counter (int) - figure id.      
    """
    mse_train =  history.history['mse'] 
    mse_val =  history.history['val_mse']  # validation sample
        
    train_loss =  history.history['loss']
    val_loss =  history.history['val_loss']

    epochs = range(len(mse_train))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    ax[0].plot(epochs, train_loss, 'b', label='Train')
    ax[0].plot(epochs, val_loss, 'r', label='Valid')
    ax[0].set_xlabel('Epoch', size=11)
    ax[0].set_ylabel('Loss', size=11)
    ax[0].set_title('Loss')
    ax[0].legend(['train', 'val'])

    ax[1].plot(epochs, mse_train, 'b', label='Train')
    ax[1].plot(epochs, mse_val, 'r', label='Valid')
    ax[1].set_xlabel('Epoch', size=11)
    ax[1].set_ylabel('MSE value', size=11)
    ax[1].set_title(f"MSE")
    ax[1].legend(['train', 'val'])

    if plot_counter is not None:
        plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=0.05, fontsize=14)
        plt.savefig(os.path.join(PATH_FIGURES + f'fig_{plot_counter}.png'))
    
    else: 
        plot_counter = 1
        plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=-0.1, fontsize=14)  
    plt.tight_layout();


def get_predictions(
    num_periods_to_forecast:int,
    # period_step:int,
    last_prediction:np.array, 
    last_index:pd.Timestamp, 
    scaler, 
    model, 
    n_steps:int, 
    #n_features:int=1
    )->pd.Series:
    """Generate predictions for future periods

    Args:
        num_periods_to_forecast (int): number of periods to predict
        period_step (int): period step
        last_prediction (np.array): _description_
        last_index (pd.Timestamp): _description_
        scaler (_type_): _description_
        model (_type_): _description_
        n_steps (int): _description_
        n_features (int, optional): _description_. Defaults to 1.

    Returns:
        pd.Series: _description_
    """
    future_indexes_array = []
    predicted_arr = np.array([], dtype='float32')

    for i in range(1, num_periods_to_forecast+1):
        # join new indexes
        future_indexes = last_index + pd.Timedelta(minutes=i)
        future_indexes_array.append(future_indexes)  #*period_step
   
        y_hat = model.predict(last_prediction, verbose=0)#.squeeze()
        predicted_arr = np.append(predicted_arr, y_hat)
        
        minutes = (future_indexes.hour*60 + future_indexes.minute)
        sin_time = np.sin(2*np.pi * minutes/1440)
        cos_time = np.cos(2*np.pi * minutes/1440)
        
        new_prediction = np.append(y_hat, [sin_time, cos_time]).reshape(1,1, n_steps)
        
        # append and roll the prediction
        last_prediction = np.concatenate((last_prediction,  new_prediction), axis=1)[:,1:,:]
    
    predictions = pd.Series(
        scaler.inverse_transform(predicted_arr.reshape(-1,1)).squeeze(), 
        name='future', 
        index=future_indexes_array)#.asfreq(freq='min')
    
    return predictions

def get_intervals(ts:pd.Series, ci_high:float=1.0, ci_low:float=1.0):
    df = pd.DataFrame(ts)
    df['ci_low' ] = ts - ci_low  * ts.std()
    df['ci_high'] = ts + ci_high * ts.std()
    
    return df