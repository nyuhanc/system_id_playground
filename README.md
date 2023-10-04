# Instructions
This repo contains multiple scripts that can be used to train and test a different data-driven models for the 
prediction of the time series, such as the classical ARX (yet to be implemented), the popular and efficient XGBoost, and different variants
of NNs, such as classic NNs, LSTMs and even some more advanced models that are based on the attention mechanism (yet to be
implemented). The model hyperparameters are tuned using various libraries, such as Optuna, Kerastuner and Scikit.

The data is taken from the Tenesse Eastman Process (TEP) dataset, which is a benchmark dataset for fault detection
but is in our case used for time series prediction. The TEP has 11 input variables and 41 output variables. We use
all the input variables (referred to as 'xmv_j') and a subset of the output variables (xmeas_i for i=1,7,,10,12,13,
15,16,17,18,19,20,21). The data can be accessed at https://in.mathworks.com/help/deeplearning/ug/chemical-process-fault-detection-using-deep-learning.html.
Before using the data, the data has to be converted from .mat to.csv and placed in the /data folder.

# Notes:
- The regressors of the target variable(s) are taken from t=-1,...,-n_lags.
The regressors of the exogenous variable(s) are taken from t=0,...,-(n_lags-1). This is due to the properties
of the data. For general purposes, the regressors of the exogenous variable(s) should as well be taken from 
t=-1,...,-n_lags.

- NRMSE is pretty much the same as RMSE since the data is z-normalized. 