import numpy as np
import matplotlib.pyplot as plt

# (a)
"""
-------------------------------------
Input: N (number of states), gamma (persistence parameter), sigma_e (std. dev. of shocks)
1. Compute sigma_y = sigma_e / sqrt(1 - gamma^2)
2. Create N equally spaced grid points (states) from -sigma_y*sqrt(N-1) to sigma_y*sqrt(N-1)
3. Set p = (1 + gamma) / 2
4. Recursively build the transition matrix P:
   - For N = 2, P = [[p, 1-p], [1-p, p]]
   - For N > 2, construct P_N using the (N-1)-state matrix:
       a. P_N[:N-1, :N-1] += p * P_(N-1)
       b. P_N[:N-1, 1:N] += (1-p) * P_(N-1)
       c. P_N[1:N, :N-1] += (1-p) * P_(N-1)
       d. P_N[1:N, 1:N] += p * P_(N-1)
       e. Divide the middle rows by 2 for normalization
Output: states, transition matrix
"""

def rouwenhorst(N, p):
    """This part is to generates transition matrix using Rouwenhorst’s method"""
    if N == 2:
        return np.array([[p, 1 - p], [1 - p, p]])
    
    P_Nm1 = rouwenhorst(N - 1, p)
    P_N = np.zeros((N, N))
    P_N[:N-1, :N-1] += p * P_Nm1
    P_N[:N-1, 1:N] += (1 - p) * P_Nm1
    P_N[1:N, :N-1] += (1 - p) * P_Nm1
    P_N[1:N, 1:N] += p * P_Nm1
    P_N[1:N-1] /= 2
    return P_N

def discretize_ar1(gamma, sigma_e, N):
    """Discretizes an AR(1) process using Rouwenhorst’s method"""
    sigma_y = sigma_e / np.sqrt(1 - gamma**2)
    states = np.linspace(-sigma_y * np.sqrt(N - 1), sigma_y * np.sqrt(N - 1), N)
    p = (1 + gamma) / 2
    transition_matrix = rouwenhorst(N, p)
    return states, transition_matrix

def simulate_markov_chain(transition_matrix, states, periods, seed=2025):
    """Simulates a Markov chain given a transition matrix"""
    np.random.seed(seed)
    N = len(states)
    state_vector = np.zeros(periods, dtype=int)
    state_vector[0] = np.random.choice(N)  # Start from a random state
    for t in range(1, periods):
        state_vector[t] = np.random.choice(N, p=transition_matrix[state_vector[t-1]])
    return states[state_vector]

# (c)
gamma = 0.85
sigma_e = 1
N = 7
periods = 50

states, transition_matrix = discretize_ar1(gamma, sigma_e, N)
simulated_values = simulate_markov_chain(transition_matrix, states, periods)

plt.figure(figsize=(10, 5))
plt.plot(range(periods), simulated_values, marker='o', linestyle='-', color='b')
plt.xlabel("Time Periods")
plt.ylabel("State Value")
plt.title(f"Markov Chain Simulation for γ = {gamma}")
plt.grid(True)
plt.show()

# (d)
gamma_values = [0.75, 0.85, 0.95, 0.99]
plt.figure(figsize=(10, 6))
for g in gamma_values:
    states, transition_matrix = discretize_ar1(g, sigma_e, N)
    simulated_values = simulate_markov_chain(transition_matrix, states, periods)
    plt.plot(simulated_values, marker='o', linestyle='-', label=f'γ = {g}')
plt.xlabel("Time Periods")
plt.ylabel("State Value")
plt.title("Markov Chain Simulations for Different γ Values")
plt.legend()
plt.grid(True)
plt.show()
