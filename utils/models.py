import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
#from torch import mean
import torch
import pandas as pd

class ResidualTCNBlock(nn.Module):
    """
    Temporal Convolutional Network for one-step forecasting.
    
    Input:
        (batch, sequence_length, features)
    
    Parameters
    ----------
    in_channels:int
        number of features
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of convolution kernel (features).

    Output:
        (batch, out_channels, seq_len)
    """

    def __init__(        
        self, 
        in_channels, 
        out_channels=32, 
        kernel_size=3, 
        dilation=1, 
        dropout=0.2):

        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding=0, #causal convolutions instead of symmentric padding
                dilation=dilation
            )
        )

        self.conv2 = weight_norm(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding=0,
                dilation=self.dilation
            )
        )
        
        #residual (skip) connections
        if in_channels != out_channels:
            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.residual = nn.Identity()
            
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(out_channels, 1)
  
    def forward(self, x):
        """
        Input shape:
            (batch, seq_len, features)

        Conv1d expects:
            (batch, features, seq_len)
        """
        # Save the original input
        residual = self.residual(x)
        
        # X is the original input of shape (batch, n_features, seq_len) 
        #x = x.transpose(1, 2)
        
        # ---------- Conv1 ----------
        # add residual, which shape becomes (batch, out_channels, seq_len) after convolution
        left_padding = (self.kernel_size - 1) * self.dilation #(3 - 1) * 1
        x = F.pad(x, (left_padding, 0))

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # ---------- Conv2 ----------
        #left_padding = (self.kernel_size - 1) * self.dilation*2 #(3 - 1) * 2
        x = F.pad(x, (left_padding, 0))
        
        x = self.conv2(x)
        x = x + residual # add residual
        x = self.relu(x)
        x = self.dropout(x)

        return x
    
    
class TCNRegressor(nn.Module):
    """
    Temporal Convolutional Network for one-step forecasting.

    Input shape:
        (batch, sequence_length, n_features)

    Output:
        (batch, 1)
    """

    def __init__(
        self,
        in_channels,
        out_channels=32,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()

        # Stack several residual blocks with increasing dilations.
        # Increasing the dilation enlarges the receptive field
        # without increasing the number of parameters significantly.

        self.blocks = nn.Sequential(

            ResidualTCNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=1,
                dropout=dropout,
            ),

            # ResidualTCNBlock(
            #     in_channels=out_channels,
            #     out_channels=out_channels,
            #     kernel_size=kernel_size,
            #     dilation=2,
            #     dropout=dropout,
            # ),

            # ResidualTCNBlock(
            #     in_channels=out_channels,
            #     out_channels=out_channels,
            #     kernel_size=kernel_size,
            #     dilation=4,
            #     dropout=dropout,
            # ),
        )

        # Final regression layer
        self.regressor = nn.Linear(out_channels, 1)

    def forward(self, x):
        """
        # x shape (batch, seq_len, features) ->
        #         (batch, features, seq_len)
        """

        # Conv1D expects
        # (batch, channels, sequence_length)
        x = x.transpose(1, 2)

        # Pass through all residual blocks
        x = self.blocks(x)

        # # Take the representation of the last time step
        # x = x[:, :, -1]
        
        #global average pooling
        x = torch.mean(x, dim=2) #torch.mean

        # Regression output
        x = self.regressor(x)

        return x
    
        
import matplotlib.pyplot as plt
import os
    
def plot_history_clf(
    history:dict=None, 
    PATH_FIGURES:str="../figures/", 
    model_name:str=None, 
    plot_counter:int=None):
    """Training history visualization
    
    Аргументы:
    history (keras.callbacks.History) - Training history data,
    PATH_FIGURES - path to the figures directory. Default: "../figures/"
    model_name (str) - figure title. Use: model.name
    plot_counter (int) - figure id.      
    """
    epochs = range(len(history['trn_loss']))
    if len(history.keys())==4:
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

        ax[0].plot(epochs, history['trn_loss'], 'b', label='Train')
        ax[0].plot(epochs, history['val_loss'], 'r', label='Valid')
        ax[0].set_xlabel('Epoch', size=11)
        ax[0].set_ylabel('Loss', size=11)
        ax[0].set_title('Loss')
        ax[0].legend(['train', 'val'])

        ax[1].plot(epochs, history['trn_rmse'], 'b', label='Train')
        ax[1].plot(epochs, history['val_rmse'], 'r', label='Valid')
        ax[1].set_xlabel('Epoch', size=11)
        ax[1].set_ylabel('MSE value', size=11)
        ax[1].set_title(f"MSE")
        ax[1].legend(['train', 'val'])
    
    else:
        plt.figure(figsize=(6, 4))
        
        plt.plot(epochs, history['trn_loss'], 'b', label='Train')
        plt.plot(epochs, history['val_loss'], 'r', label='Valid')
        plt.xlabel('Epoch', size=11)
        plt.ylabel('Huber Loss', size=11)
        plt.title('Loss')
        plt.legend(['train', 'val'])

        if plot_counter is not None:
            plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=0.05, fontsize=14)
            plt.savefig(os.path.join(PATH_FIGURES + f'fig_{plot_counter}.png'))
        
        else: 
            plot_counter = 1
            plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=-0.1, fontsize=14)  
        plt.tight_layout();
        
import numpy as np

def split_N_sequences(X:np.array, Y:np.array, n_steps:int)->np.array:
    """ Split a univariate sequence into samples

    Args:
        sequence (np.array): time series (N features). Shape (N, n_features)
        n_steps (int): lag. Shape (N,)

    Returns:
        np.array: _description_
    """    
    X_out, y_out = [], []
 
    for i in range(len(X) - n_steps):
        
        X_out.append(X[i:i+n_steps])
        y_out.append(Y[i+n_steps])
  
    return np.array(X_out), np.array(y_out)


def get_tch_predictions(
    num_periods_to_forecast,
    last_prediction,
    last_index,
    scaler,
    model:TCNRegressor,
    #device,
):
    """
    Recursive forecasting using a trained PyTorch TCN.

    Parameters
    ----------
    last_prediction : ndarray
        Shape = (1, seq_len, n_features)

    model : torch.nn.Module
        Trained model.
    """

    model.eval()
    future_indexes, predicted = [], []
    window = last_prediction.float() #torch.tensor(last_prediction, dtype=torch.float32)

    with torch.no_grad():

        for i in range(num_periods_to_forecast):

            y_hat = model(window)
            value = y_hat.item()
            
            predicted.append(value)

            timestamp = last_index + pd.Timedelta(minutes=i + 1)

            future_indexes.append(timestamp)

            minutes = timestamp.hour * 60 + timestamp.minute
            sin_time = np.sin(2 * np.pi * minutes / 1440)
            cos_time = np.cos(2 * np.pi * minutes / 1440)

            new_step = torch.tensor(
                [[[value, sin_time, cos_time]]],
                dtype=torch.float32,
                #device=device
            )

            window = torch.cat((window[:, 1:, :], new_step, ),dim=1)

    prediction = scaler.inverse_transform(
        np.array(predicted).reshape(-1, 1)
    ).squeeze()

    return pd.Series(
        prediction,
        index=future_indexes,
        name="future",
    )
    
def get_intervals(
    ts:pd.Series, 
    ci_low:float=1.0, 
    ci_high:float=1.0,
    ci_minimum:float=0.5):
    """Create confidence interval

    Args:
        ts (pd.Series): time series
        ci_high (float, optional): upper boundary. Defaults to 1.0.
        ci_low (float, optional): lower boundary. Defaults to 1.0.
        ci_minimum(float): minimal boundary of controlled parameter

    Returns:
        _type_: Parameter with confidence interval
    """    
    df = pd.DataFrame(ts)
    #print(f'ts.std() = {ts.std():.3}')
    df['ci_low' ] = ts - ci_low  #ts - ci_low  * ts.std()
    df['ci_high'] = ts + ci_high #ts + ci_high * ts.std()

    df['ci_low' ] = np.where(df['ci_low' ] < ci_minimum, ci_minimum, df['ci_low'])
    
    return df