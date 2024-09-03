import numpy as np
import scipy.stats as ss

def MC_price(S0, K, r, sigma, T, M, Ite, t, call_or_put="C"):
    
    T = T - t
    dt = T / M
    S = np.zeros((M+1, Ite))
    S[0] = S0  # at t=0, all prices are S0. (first row is S0 repeated.)
    rn = np.random.standard_normal((M, Ite))
    multipliers = np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * rn)
    # print("multipliers.cumprod() =\n", multipliers.cumprod(axis=0))
    S[1:, :] = S0 * multipliers.cumprod(axis=0)  # price paths are generated
    print("S =\n", S)
    
    if call_or_put == "C":
        average_payoff = np.maximum(S[-1, :] - K, 0).mean()
    else:
        average_payoff = np.maximum(K - S[-1, :], 0).mean()
    
    opt_price = np.exp(-r*dt) * average_payoff
    return opt_price

r = 0.05  # Risk-free rate
sigma = 0.45  # Volatility
T = 3/12  # Maturity/time period (in years)
S0 = 200  # Current Stock Price

M = 20000  # Number of simulations (paths)
n = 1  # Number of steps

# Option params
K = 195
t = 0
call_or_put = "C"

# SEED
np.random.seed(2)

bs_opt_price = MC_price(S0, K, r, sigma, T, n, M, t, call_or_put)
print("bs_opt_price =", bs_opt_price)