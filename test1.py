import numpy as np

def MonteCarloPrice(O, S, K, T, r, sigma, steps, paths):
    dt = T / steps 
    S_path = np.zeros((steps + 1, paths))  
    S_path[0] = S  

    for step in range(1, steps + 1):
        epsilon = np.random.standard_normal(paths)  # radom
        # ΔS = rS Δt + σS ε √Δt
        delta_S = r * S_path[step - 1] * dt + sigma * S_path[step - 1] * epsilon * np.sqrt(dt)
        S_path[step] = S_path[step - 1] + delta_S  
        payoff = np.maximum(K - S_path[-1], 0) 
    value = np.exp(-r * T) * payoff.mean()
    return value


C = MonteCarloPrice(O="c", S=12.18, K=12.18, T=0.75, r=0.0417, sigma=0.436, steps=150, paths=10000)
P = MonteCarloPrice(O="p", S=12.18, K=12.18, T=0.75, r=0.0417, sigma=0.436, steps=150, paths=300000)

# 输出结果
print(round(C, 3))  # 看涨期权价格
print(round(P, 3))  # 看跌期权价格