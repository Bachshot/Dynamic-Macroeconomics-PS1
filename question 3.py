import numpy as np
import matplotlib.pyplot as plt

def rouwenhorst(n, p, q):
    """Generate a transition matrix using Rouwenhorst’s method."""
    if n == 2:
        return np.array([[p, 1 - p], [1 - q, q]])

    P_n_minus_1 = rouwenhorst(n - 1, p, q)
    top = np.hstack((P_n_minus_1, np.zeros((n - 1, 1))))
    bottom = np.hstack((np.zeros((n - 1, 1)), P_n_minus_1))
    middle = np.hstack(((1 - p) * P_n_minus_1, p * P_n_minus_1))
    middle = np.vstack((middle, (q * P_n_minus_1, (1 - q) * P_n_minus_1)))

    return middle / middle.sum(axis=1, keepdims=True)

def discretize_ar1(rho, sigma, n=7):
    """Discretize the AR(1) process using Rouwenhorst’s method."""
    sigma_y = np.sqrt(sigma ** 2 / (1 - rho ** 2))
    step = 2 * sigma_y / (n - 1)
    states = np.linspace(-sigma_y, sigma_y, n)
    P = rouwenhorst(n, (1 + rho) / 2, (1 + rho) / 2)
    return states, P

def simulate_markov_chain(P, states, periods=50, seed=2025):
    """Simulate a Markov chain given a transition matrix."""
    np.random.seed(seed)
    n = len(states)
    state_idx = np.random.choice(n)  # Start from a random state
    history = [states[state_idx]]

    for _ in range(periods - 1):
        state_idx = np.random.choice(n, p=P[state_idx])
        history.append(states[state_idx])

    return history

# Part (b) - Discretization
rho_values = [0.75, 0.85, 0.95, 0.99]
sigma = 1  # Assume standard deviation of error term is 1

plt.figure(figsize=(10, 6))

for rho in rho_values:
    states, P = discretize_ar1(rho, sigma)
    history = simulate_markov_chain(P, states)
    plt.plot(history, label=f"γ₁ = {rho}")

# Part (d) - Plot results
plt.xlabel("Time")
plt.ylabel("State Value")
plt.legend()
plt.title("Markov Chain Simulation for Different γ₁ Values")
plt.show()
