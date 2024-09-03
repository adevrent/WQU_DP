import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

def gen_dZ(rho, n, T):
    """Generates correlated standard brownian motions dZ1 and dZ2

    Args:
        rho (float): correlation between the "asset brownian motion" dZ1
                    and the "variance brownian motion" dZ2
        n (int): number of timesteps
    """
    dt = T/n
    
    # dZ1 ~ N(0, dt)
    dZ1 = np.random.standard_normal(n) * np.sqrt(dt)  # create the first brownian motion elements, for asset price process
    temp_arr = np.random.standard_normal(n)  # draw independent standard gaussians to use for correlated random variables
    dZ2 = rho * dZ1 + np.sqrt(1 - rho**2) * temp_arr * np.sqrt(dt)  # create the second brownian motion elements, for variance process
    return dZ1, dZ2

def v(v0, kappa, theta, sigma, n, T, dZ2):
    """Calculates the variance process

    Args:
        v0 (float): initial variance value to start from
        kappa (float): parameter for the speed of mean-reversion
        theta (float): long term average of the variance
        dt (float): length of timestep
        sigma (float): volatility of the variance process
        rho (float): correlation between the "asset brownian motion" dZ1
                    and the "variance brownian motion" dZ2
        n (int): number of timesteps to simulate
        T (float): Time at the end of the process to be simulated (maturity date)
        dZ2 (np.array): Standard brownian motion interval values
        which are correlated to dZ1 with rho
    """
    dt = T/n
    v_arr = np.zeros(n)
    v_arr[0] = v0
    # print("v_arr =", v_arr)
    for t in range(1, n):
        v_arr[t] = np.maximum(v_arr[t-1] + kappa * (theta - v_arr[t-1]) * dt + sigma * np.sqrt(v_arr[t-1]) * dZ2[t], 0)
    
    return v_arr

def S(S0, r, v_arr, T, dZ1):
    """Calculates the asset price process

    Args:
        S0 (float): Initial asset price at start
        r (float): risk-free rate
        sigma (float): volatility of the variance process
        v_arr (np.array): variance process generated by function v()
        T (float): Time at the end of the process to be simulated (maturity date)
        dZ1 (np.array): Standard brownian motion interval values
    """
    n = len(v_arr)
    dt = T/n
    
    S_arr = np.zeros(n)
    S_arr[0] = S0
    for t in range(1, n):
        S_arr[t] = S_arr[t-1] * np.exp((r - v_arr[t]/2)*dt + np.sqrt(v_arr[t])*dZ1[t])
        
    return S_arr

def MC(M, v0, kappa, theta, sigma, n, T, rho, r, S0=None):
    if S0 is None:
        v_MC = np.zeros((M, n))
        for m in range(M):
            dZ1, dZ2 = gen_dZ(rho, n)
            v_MC[m, :] = np.maximum(v(v0, kappa, theta, sigma, n, T, dZ2), 0)
        return v_MC
    else:
        v_MC = np.zeros((M, n))
        S_MC = np.zeros((M, n))
        for m in range(M):
            dZ1, dZ2 = gen_dZ(rho, n, T)
            v_arr = v(v0, kappa, theta, sigma, n, T, dZ2)
            v_MC[m, :] = np.maximum(v_arr, 0)
            S_MC[m, :] = S(S0, r, v_arr, T, dZ1)
    return v_MC, S_MC

def heston_opt_price_MC(K, t, MC_params, call_or_put="C"):
    M, v0, kappa, theta, sigma, n, T, rho, r, S0 = MC_params
    v_MC, S_MC = MC(M, v0, kappa, theta, sigma, n, T, rho, r, S0)
    if call_or_put == "C":
        payoffs = np.maximum(0, S_MC[:, -1] - K)
    else:
        payoffs = np.maximum(0, K - S_MC[:, -1])
        
    return np.exp(-r * (T-t)) * payoffs.mean()

# MC params
v0 = 0.05
kappa = 2
sigma = 0.3
theta = 0.04
rho = -0.9
S0 = 100  # Current underlying asset price
r = 0.05  # Risk-free rate
T = 1  # Number of years
n = 5
M = 1  # Number of simulations

# Option params
K = 100
t = 0
call_or_put = "C"

MC_params = [M, v0, kappa, theta, sigma, n, T, rho, r, S0]

# SEED
np.random.seed(2)

# heston_opt_price = heston_opt_price_MC(K, t, MC_params, call_or_put)
# print("heston_opt_price =", heston_opt_price)

v_MC, S_MC = MC(M, v0, kappa, theta, sigma, n, T, rho, r, S0)
print("v_MC =", v_MC)