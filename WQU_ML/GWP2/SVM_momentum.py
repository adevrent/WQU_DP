import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay
import yfinance as yf
import datetime as dt

def gen_asset_data(ticker, start, end):
    df = yf.download(ticker, start, end).loc[:, ["Adj Close", "Volume"]]
    df.columns = ["price", "volume"]
    df["returns"] = df.loc[:, "price"].pct_change()
    df["20ma"] = df["returns"].rolling(20).mean()
    df["200ma"] = df["returns"].rolling(200).mean()
    df["bullish"] = (df["20ma"] > df["200ma"]).astype(int)
    df = df.dropna()
    df = df.loc[:, ["20ma", "200ma", "bullish"]]
    
    return df

# Params
n_days = 1000
end = dt.datetime.today()
start = end - dt.timedelta(n_days)
ticker = "AMZN"

# Generate dataframe
df = gen_asset_data(ticker, start, end)
print(df.head())

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 2], cmap=cm.coolwarm, s=10, alpha=0.6)
# ax.set_xlim((-0.5, 0.5))
# ax.set_ylim((-0.05, 0.05))
fig.show()

# Split features and labels data
X = df.iloc[:, 0:2].to_numpy()
y = df.iloc[:, 2].to_numpy()

# Split data into training and test sets
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.2)

# Use SVC class from sklearn library to fit SVM with different choice of kernel functions
svm_rbf = SVC(kernel='poly', degree=1, C=0.5, gamma='scale', probability=True)
svm_rbf.fit(X_train, y_train)

# Get class predictions for test datasets
y_hat = svm_rbf.predict(X_test)

# Print confusion tables to show accuracy
print("RBF (Radial) Confusion Table:")
print(confusion_table(y_test, y_hat))

# Plotting decision boundaries and performance curves
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].set_title("RBF - Decision Boundary")
plot_svm(X_train, y_train, svm_rbf, ax=axes[0])
# axes[0, 0].scatter(X_test_rad[:, 0], X_test_rad[:, 1], c=y_test_rad, marker='x', cmap='coolwarm', alpha=0.7)
axes[0].legend()

axes[1].set_title("RBF - ROC Curve")
RocCurveDisplay.from_estimator(svm_rbf, X_test, y_test, ax=axes[1])