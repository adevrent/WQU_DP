import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

def S(lamb, mu, delta, r, sigma, T, S0, n):
    dt = T / n
    r_j = lamb * (np.exp(mu + delta**2 / 2) - 1)
    S_arr = np.zeros(n)
    S_arr[0] = S0
    for t in range(1, n):
        S_arr[t] = np.maximum(0.0001, (S_arr[t-1] * (np.exp((r - r_j - sigma**2 / 2)*dt + sigma*np.sqrt(dt)*np.random.standard_normal(1)) + (np.exp(mu + delta * np.random.standard_normal(1)) - 1)*np.random.poisson(lamb*dt, 1))))
    
    return S_arr

def merton_MC(M, lamb, mu, delta, r, sigma, T, S0, n):
    S_MC = np.zeros((M, n))  # each row is one path's time series
    for m in range(M):
        S_MC[m, :] = S(lamb, mu, delta, r, sigma, T, S0, n)
    
    return S_MC

def merton_opt_price_MC(merton_paramlist, K, t, call_or_put="C"):
    M, lamb, mu, delta, r, sigma, T, S0, n = merton_paramlist
    T_t = T - t
    S_MC = merton_MC(M, lamb, mu, delta, r, sigma, T_t, S0, n)
    if call_or_put == "C":
        payoffs = np.maximum(0, S_MC[:, -1] - K)
    else:
        payoffs = np.maximum(0, K - S_MC[:, -1])
    price = np.exp(-r * T_t) * payoffs.mean()
    
    return price

lamb = 0.75  # Lambda of the model
mu = 0.5  # Mu
delta = 0.25  # Delta

r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 1  # Maturity/time period (in years)
S0 = 50  # Current Stock Price

M = 10000  # Number of simulations (paths)
n = 25  # Number of steps

merton_paramlist = [M, lamb, mu, delta, r, sigma, T, S0, n]

# Option params
K = 40
t = 0
call_or_put = "P"

# SEED
np.random.seed(0)

merton_opt_price = merton_opt_price_MC(merton_paramlist, K, t, call_or_put)
print("merton_opt_price =", merton_opt_price)