import numpy as np

# Parameters
n = 20  # Number of students
m = 4   # Number of groups
group_size = n // m  # Students per group
affinity_matrix = np.random.uniform(0, 5, (n, m))  # Affinity values

# Ensure no 0-affinity groups for random initialization
affinity_matrix[affinity_matrix < 1] = 0.1

# Parameters
pop_size = 300
generations = 100
mutation_rate = 0.5
elitism = 40
trials = 3  # Number of runs per operator
patience = 20

crossover_methods = {}