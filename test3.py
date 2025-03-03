import numpy as np
import time

def simulate_stock_price_exact(S0, r, sigma, t, epsilon):
    return S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * epsilon * np.sqrt(t))

def MonteCarloAverageWorstOfPutTwoSteps(S1, S2, sigma1, sigma2, T, r, paths):
    dt1 = 0.25  
    dt2 = 0.5   
    epsilon1_S1 = np.random.standard_normal(paths)  
    epsilon1_S2 = np.random.standard_normal(paths) 
    S1_t1 = simulate_stock_price_exact(S1, r, sigma1, dt1, epsilon1_S1)
    S2_t1 = simulate_stock_price_exact(S2, r, sigma2, dt1, epsilon1_S2)

    epsilon2_S1 = np.random.standard_normal(paths)  
    epsilon2_S2 = np.random.standard_normal(paths)  
    S1_t2 = simulate_stock_price_exact(S1_t1, r, sigma1, dt2, epsilon2_S1)
    S2_t2 = simulate_stock_price_exact(S2_t1, r, sigma2, dt2, epsilon2_S2)

    B1 = np.minimum(S1_t1 / S1, S2_t1 / S2)  # B1 = min(S1,1/S1,0, S2,1/S2,0)
    B2 = np.minimum(S1_t2 / S1, S2_t2 / S2)  # B2 = min(S1,2/S1,0, S2,2/S2,0)
    A = (B1 + B2) / 2  # A = (B1 + B2) / 2
    payoff = np.maximum(1 - A, 0)  # max(100% - A, 0)
    option_price = np.exp(-r * T) * payoff.mean()
    return option_price


S1 = 12.18  
S2 = 6.03   
sigma1 = 0.436  
sigma2 = 0.30   
T = 0.75    
r = 0.0417  

# (a) 10,000
start_time = time.time()
price_10000 = MonteCarloAverageWorstOfPutTwoSteps(S1, S2, sigma1, sigma2, T, r, paths=10000)
end_time = time.time()
print(f"10,000 条路径的期权价格: {price_10000:.4f}")
print(f"计算时间: {end_time - start_time:.2f} 秒")

# (b) 300,000
start_time = time.time()
price_300000 = MonteCarloAverageWorstOfPutTwoSteps(S1, S2, sigma1, sigma2, T, r, paths=300000)
end_time = time.time()
print(f"300,000 条路径的期权价格: {price_300000:.4f}")
print(f"计算时间: {end_time - start_time:.2f} 秒")