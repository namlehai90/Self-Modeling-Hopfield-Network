"""
Created on Mon Mar 13 11:10:56 2023

@author: namlh
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

np.random.seed(2023)

"""
N: Number of nodes (set to 100)
M: Number of modules (set to 10)
K: Number of nodes per module (set to 10)
E: Weight of weakly weighted connections (set to 0.1)
They also use the following parameter:
W: Maximum weight of a constraint (set to 5.0)
"""

N = 150  # number of variables/neurons
P = 0.8  # probability of a constraint being positive
W = 5.0  # maximum weight of a constraint
E = 0.01 # weight of weakly weighted connections
M = 30   # number of modules
K = 5    # number of nodes per module


# Define system parameters
relaxation1 = 300 # relaxation length without learning
relaxation2 = 300 # relaxation length with learning
beta = 1.0 # inverse temperature parameter
learning_rate = 0.005 # learning rate parameter

# Step1: Generate random constraints
constraints = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        if np.random.rand() < P:
            weight = np.random.uniform(0, W)
            if np.random.rand() < 0.5:
                constraints[i, j] = weight
                constraints[j, i] = weight
            else:
                constraints[i, j] = -weight
                constraints[j, i] = -weight
        else:
            constraints[i, j] = 0.0
            constraints[j, i] = 0.0

# Step 2: Generate a modular problem structure with strongly weighted connections 
# within modules and weakly weighted connections between modules.
# Generate adjacency matrix
adjacency = np.zeros((N, N))
for m in range(M):
    module = np.arange(m*K, (m+1)*K)
    adjacency[module.reshape(-1, 1), module] = constraints[module.reshape(-1, 1), module]
    adjacency[module, module.reshape(-1, 1)] = constraints[module, module.reshape(-1, 1)]
for i in range(N):
    for j in range(i+1, N):
        if adjacency[i, j] == 0 and adjacency[j, i] == 0:
            adjacency[i, j] = -E
            adjacency[j, i] = -E

# Define energy function
def energy(state, adjacency):
    return -0.5 * np.dot(np.dot(state, adjacency), state)

# Define update rule
def update_heavyside(state, weights, i):
    activation = np.dot(weights[:, i], state)
    new_state = np.where(activation >= 0, 1, -1)
    return new_state

# Initialize states for both non-learning and learning
init_states = np.random.choice([-1,1], size=(relaxation1+relaxation2, N))

# Initialize weights randomly
init_weights = deepcopy(adjacency)
    
# Relax system first without learning, then continue with learning
energies1 = []
energies2 = []
first_relax = []
first_learning_relax = []

# Shuffle order of state updates
update_order = np.arange(N)
np.random.shuffle(update_order)

for i in range(relaxation1+relaxation2):
    state = init_states[i]
    # Calculate energy of current state
    energy_i = energy(state, init_weights)
    
    # Append energy to respective energy list (without or with learning)
    if i < relaxation1:
        print(f"Energy at relaxation {i+1} without learning is {energy_i:.3f}")
        energies1.append(energy_i)
    else:
        print(f"Energy at relaxation {i+1} with learning is {energy_i:.3f}")
        energies2.append(energy_i)
        
    for j in range(N):
        # Record energy of current state in first_relax or first_learning_relax energy list
        energy_i = energy(state, init_weights)
        if i < 5:
            first_relax.append(energy_i)
        else:
            if i > relaxation1 and i < relaxation1+5:
                first_learning_relax.append(energy_i)
                
        # k = np.random.randint(0, N)
        # pick a state from a permutation
        k = update_order[j]
        state[k] = update_heavyside(state, init_weights, k)
        if i >= relaxation1:
            delta_weights = learning_rate * np.outer(state[k], state)
            init_weights[k] += delta_weights.reshape(N,)

# Plot energy vs state updates for first relaxation
for i in range(5):
    start_idx = i * N
    end_idx = (i + 1) * N
    energy_line = first_relax[start_idx:end_idx]
    plt.plot(range(len(energy_line)), energy_line, label=f'Relaxation {i+1}')
plt.xlabel('State updates')
plt.ylabel('Energy')
plt.title('Energy vs State Updates for Nonlearning (First 5 Relaxations)')
plt.legend()
plt.show()

# Plot energy vs state updates for first 5 relaxations
for i in range(5):
    start_idx = i * N
    end_idx = (i + 1) * N
    energy_line = first_learning_relax[start_idx:end_idx]
    plt.plot(range(len(energy_line)), energy_line, label=f'Relaxation {i+1}')
plt.xlabel('State updates')
plt.ylabel('Energy')
plt.title('Energy vs State Updates for Learning (First 5 Relaxations)')
plt.legend()
plt.show()

# Concatenate the energy lists and create a list of colors for the scatter plot
colors = ['red']*relaxation1 + ['blue']*relaxation2

# Plot energy vs relaxations for nonlearning
plt.scatter(range(relaxation1), energies1, s=2, label='Non-learning')

# Plot energy vs relaxations for learning
plt.scatter(range(relaxation1, relaxation1+relaxation2), energies2, s=2, label='Learning')

# Plot turning point
plt.axvline(x=relaxation1, linestyle='--', color='black', label='Turning Point')
plt.annotate('Non-learning', xy=(0.25, 0.85), xycoords='figure fraction', color='red')
plt.annotate('Learning', xy=(0.65, 0.85), xycoords='figure fraction', color='blue')
plt.xlabel('Relaxation')
plt.ylabel('Energy')
plt.title('Energy vs Relaxations')
plt.legend()
plt.show()

# Plot histograms of attractor energies before and after learning
plt.figure(4)
plt.hist(energies1, bins=20, alpha=0.5, color='red', label='Non-learning')
plt.hist(energies2, bins=20, alpha=0.5, color='blue', label='Learning')
plt.xlabel('Final Energy')
plt.ylabel('Count')
plt.title('Histogram of Attractor Energies Before and After Learning')
plt.legend()
plt.show()
