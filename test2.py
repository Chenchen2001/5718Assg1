import numpy as np
import time

def MonteCarloAverageWorstOfPut(S1, S2, sigma1, sigma2, T, r, steps, paths):
    dt = T / steps  
    S1_path = np.zeros((steps + 1, paths))  
    S2_path = np.zeros((steps + 1, paths))  
    S1_path[0] = S1  
    S2_path[0] = S2  

    for step in range(1, steps + 1):
        epsilon1 = np.random.standard_normal(paths)  
        epsilon2 = np.random.standard_normal(paths)  
        
        delta_S1 = r * S1_path[step - 1] * dt + sigma1 * S1_path[step - 1] * epsilon1 * np.sqrt(dt)
        delta_S2 = r * S2_path[step - 1] * dt + sigma2 * S2_path[step - 1] * epsilon2 * np.sqrt(dt)
        S1_path[step] = S1_path[step - 1] + delta_S1
        S2_path[step] = S2_path[step - 1] + delta_S2

    B1 = np.minimum(S1_path[int(steps * 0.25)] / S1, S2_path[int(steps * 0.25)] / S2)  # B1 = min(S1,1/S1,0, S2,1/S2,0)
    B2 = np.minimum(S1_path[-1] / S1, S2_path[-1] / S2)  # B2 = min(S1,2/S1,0, S2,2/S2,0)
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
steps = 150  

# (a) 10,000 
start_time = time.time()
price_10000 = MonteCarloAverageWorstOfPut(S1, S2, sigma1, sigma2, T, r, steps, paths=10000)
end_time = time.time()
print(f"10,000 条路径的期权价格: {price_10000:.4f}")
print(f"计算时间: {end_time - start_time:.2f} 秒")

# (b) 300,000 
start_time = time.time()
price_300000 = MonteCarloAverageWorstOfPut(S1, S2, sigma1, sigma2, T, r, steps, paths=300000)
end_time = time.time()
print(f"300,000 条路径的期权价格: {price_300000:.4f}")
print(f"计算时间: {end_time - start_time:.2f} 秒")