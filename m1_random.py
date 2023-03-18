"""
Created on Mon Mar 13 10:10:37 2023

@author: namlh
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2023)

# Define system parameters
N = 100 # number of neurons
relaxation1 = 1000 # relaxation length without learning
relaxation2 = 1000 # relaxation length with learning
beta = 1.0 # inverse temperature parameter
learning_rate = 0.0025 # learning rate parameter: 0.0025, 0.001, 0.01

# Define energy function
def energy(state, weights):
    return -0.5 * np.dot(np.dot(state, weights), state)

# Define update rule
# heavy side
def update_heavyside(state, weights, i):
    activation = np.dot(weights[:, i], state)
    new_state = np.where(activation >= 0, 1, -1)
    return new_state

# Boltzmann update rule
def update_boltzmann(state, weights, i, beta):
    energy = np.dot(weights[:, i], state)
    prob = 1 / (1 + np.exp(-beta * energy))
    new_state = np.random.choice([-1, 1], p=[1-prob, prob])
    return new_state


# Initialize weights randomly
init_weights = np.random.choice([-1, 1], size=(N, N))
init_weights = 0.5 * (init_weights + init_weights.T)

# Initialize states for both non-learning and learning
init_states = np.random.choice([-1,1], size=(relaxation1+relaxation2, N))

# Relax system first without learning, then continue with learning
energies1 = []
energies2 = []
first_relax = []
first_learning_relax = []

# Shuffle order of state updates
update_order = np.arange(N)
np.random.shuffle(update_order)

# save the dynamics of states at relaxation 1
relax_states = []

for i in range(relaxation1+relaxation2):
    state = init_states[i]
    energy_i = energy(state, init_weights)
    if i < relaxation1:
        print(f"Energy at relaxation {i+1} without learning is {energy_i:.3f}")
        energies1.append(energy_i)
    else:
        print(f"Energy at relaxation {i+1} with learning is {energy_i:.3f}")
        energies2.append(energy_i)
        
    for j in range(N):
        energy_i = energy(state, init_weights)
    
        # save energy across state updates for 5 relaxations
        if i < 5:
            first_relax.append(energy_i)
        else:
            if i >= relaxation1 and i < relaxation1+5:
                first_learning_relax.append(energy_i)
        # k = np.random.randint(0, N)
        # pick a state from a permutation
        k = update_order[j]
        state[k] = update_heavyside(state, init_weights, k)
        if i >= relaxation1:
            delta_weights = learning_rate * np.outer(state[k], state)
            init_weights[k] += delta_weights.reshape(N,)
            
    relax_states.append(state)
            
# Plot energy vs state updates for first relaxation for non learning
plt.figure(1)
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

# Plot energy vs state updates for first relaxation for learning
plt.figure(2)
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
plt.figure(3)
energies = energies1 + energies2
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

# Visualize final state
# plt.imshow(init_states[0].reshape((10,15)), cmap='gray')
# plt.show()
'''import matplotlib.animation as animation

# Define function to update heatmap
def update_heatmap(frame):
    if frame >= len(relax_states):
        return
    ax.clear()
    ax.imshow(np.reshape(relax_states[frame], (10, 10)), cmap='gray')
    ax.set_title(f'Relaxation 1 - Iteration {frame+1}')


# Initialize figure and axis
fig, ax = plt.subplots()

# Create animation object
ani = animation.FuncAnimation(fig, update_heatmap, frames=relaxation1, interval=1000)

# Show animation
plt.show()'''
