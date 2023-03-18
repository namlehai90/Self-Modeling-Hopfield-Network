import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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

# Define energy function
def energy(state, adjacency):
    return -0.5 * np.dot(np.dot(state, adjacency), state)

# Define update rule
def update_heavyside(state, adjacency, i):
    activation = np.dot(adjacency[:, i], state)
    new_state = np.where(activation >= 0, 1, -1)
    return new_state

# Run relaxation without learning
num_trials = 100
learning_history = []
nonlearning_history = []

for t in range(num_trials):
    np.random.seed(seed=t)
    print(f'Running trial {t+1}/{num_trials}')
    
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

    # Initialize states for both non-learning and learning
    init_states = np.random.choice([-1,1], size=(relaxation1+relaxation2, N))

    # Initialize weights randomly
    init_weights = deepcopy(adjacency)
    
    # Relax system first without learning, then continue with learning
    energies1 = []
    energies2 = []
    for i in range(relaxation1+relaxation2):
        state = init_states[i]
        energy_i = energy(state, init_weights)
        if i < relaxation1:
            energies1.append(energy_i)
        else:
            energies2.append(energy_i)
        
        for j in range(N):
            energy_i = energy(state, init_weights)
            k = np.random.randint(0, N)
            state[k] = update_heavyside(state, init_weights, k)
            if i >= relaxation1:
                delta_weights = learning_rate * np.outer(state[k], state)
                init_weights[k] += delta_weights.reshape(N,)
    
    # saving energy each trial
    nonlearning_history.append(energies1)
    learning_history.append(energies2)

# save energy files      
np.savetxt('learning_history.txt', learning_history)
np.savetxt('nonlearning_history.txt', nonlearning_history)


learning_avg = np.mean(learning_history, axis=0)
nonlearning_avg = np.mean(nonlearning_history, axis=0)
# Concatenate the energy lists and create a list of colors for the scatter plot
plt.figure(1)

colors = ['red']*relaxation1 + ['blue']*relaxation2

# Plot energy vs relaxations for nonlearning
plt.scatter(range(relaxation1), nonlearning_avg, s=2, label='Non-learning')

# Plot energy vs relaxations for learning
plt.scatter(range(relaxation1, relaxation1+relaxation2), learning_avg, s=2, label='Learning')

# Plot turning point
plt.axvline(x=relaxation1, linestyle='--', color='black', label='Turning Point')
plt.annotate('Non-learning', xy=(0.25, 0.85), xycoords='figure fraction', color='red')
plt.annotate('Learning', xy=(0.65, 0.85), xycoords='figure fraction', color='blue')
plt.xlabel('Relaxation')
plt.ylabel('Energy')
plt.title('Energy vs Relaxations')
plt.legend()
plt.savefig('energy_vs_relaxations.png')
plt.show()

# Plot histograms of attractor energies before and after learning
plt.figure(4)
plt.hist(energies1, bins=20, alpha=0.5, color='red', label='Non-learning')
plt.hist(energies2, bins=20, alpha=0.5, color='blue', label='Learning')
plt.xlabel('Final Energy')
plt.ylabel('Count')
plt.title('Histogram of Attractor Energies Before and After Learning')
plt.legend()
plt.savefig('histogram_of_attractor_energies.png')
plt.show()
