# Instructions [to be written]

# Notes:
- The regressors of the target variable(s) are taken from t=-1,...,-(n_lags+1).
The regressors of the exogenous variable(s) are taken from t=0,...,(n_lags). This is due to the properties of the data. 
For general purposes, the regressors of the exogenous variable(s) should be taken from t=-1,...,-(n_lags+1).

- NRMSE is pretty much the same as RMSE since the data is z-normalized. 