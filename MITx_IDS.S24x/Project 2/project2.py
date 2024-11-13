import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def construct_Hankel(series, m, L):
    series = series.to_numpy()
    num_cols = L-m+1
    H = series[:m][:, np.newaxis]
    for j in range(1, num_cols):
        H = np.hstack([H, series[j:m+j][:, np.newaxis]])
    return H

def construct_H_hat(H, r):
    U, S, Vh = np.linalg.svd(H)
    r = 2
    S_r = np.diag(S)
    S_r[r:, r:] = 0
    S_r = np.hstack([S_r, np.zeros((S_r.shape[0], Vh.shape[0] - S_r.shape[1]))])
    H_hat = U @ S_r @ Vh
    print("H_hat.shape =", H_hat.shape)
    
    return H_hat

def read_between_dates(filename, start, end):
    df = pd.read_csv(filename)
    data = df[df["RegionName"].isin(["Boston, MA"])].iloc[0, 5:]
    data = data.astype(np.float64)
    data.index = pd.to_datetime(data.index)
    df = data.loc[start:end]
    return df

# Dates
train_start = "2010-01-01"
train_end = "2017-12-31"
test_start = "2018-01-01"
test_end = "2019-12-31"

filename = r"C:\Users\adevr\WQU_DP\MITx_IDS.S24x\Project 2\data_zillow_house_prices-proj2.csv"

train = read_between_dates(filename, train_start, train_end)
test = read_between_dates(filename, test_start, test_end)
all = read_between_dates(filename, train_start, test_end)

# Hankel
m = 10
L = 50
r = 2

# To predict last row
H_train = construct_Hankel(train[-50:], m, L)
H_hat_train = construct_H_hat(H_train, r)

# Endog and exog
y = H_train[-1, :][:, np.newaxis]  # col vector
X = H_hat_train[:-1, :].T

print("y.shape =", y.shape)
print("X.shape =", X.shape)

model = sm.OLS(y, X).fit()

beta = model.params[:, np.newaxis]

# Check results
OLS_forecast = (H_train[:-1, :].T @ beta).flatten()
true_row_value = y.T.flatten()

print("Max abs residual:", np.abs(model.resid).max())

fig, axs = plt.subplots(2, 1)
axs[0].plot(OLS_forecast, label="OLS forecast")
axs[0].plot(true_row_value, label="True row value")
axs[1].plot(model.resid)
plt.legend()
plt.grid()
plt.show()

# To predict next row
new_row_forecast = beta.T @ H_train[1:, :]
print(new_row_forecast/1000)

print("True last train value:", train[-1])
print("True first test value:", test[0])