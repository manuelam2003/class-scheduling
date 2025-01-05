import numpy as np
import random
from parameters import n, m, group_size, affinity_matrix

def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        chromosome = np.zeros((n, m), dtype=int)
        for student in range(n):
            valid_groups = np.where(affinity_matrix[student] > 0)[0]
            group = random.choice(valid_groups)
            chromosome[student, group] = 1
        if not is_balanced(chromosome):
            chromosome = repair_balance(chromosome)
        population.append(chromosome)
    return population

def is_balanced(chromosome):
    return all(np.sum(chromosome, axis=0) == group_size)

def repair_balance(chromosome):
    group_counts = np.sum(chromosome, axis=0)
    excess = np.where(group_counts > group_size)[0]
    deficit = np.where(group_counts < group_size)[0]

    for e in excess:
        students = np.where(chromosome[:, e] == 1)[0]
        for s in students:
            if group_counts[e] == group_size:
                break
            valid_groups = [g for g in range(m) if g != e and group_counts[g] < group_size and affinity_matrix[s, g] > 0]
            if valid_groups:
                new_group = random.choice(valid_groups)
                chromosome[s, e] = 0
                chromosome[s, new_group] = 1
                group_counts[e] -= 1
                group_counts[new_group] += 1

    for student in range(n):
        if not np.any(chromosome[student]):
            valid_groups = np.where(affinity_matrix[student] > 0)[0]
            if valid_groups.size > 0:
                new_group = random.choice(valid_groups)
                chromosome[student, new_group] = 1
                group_counts[new_group] += 1

    return chromosome

def align_fitness_progressions(fitness_runs):
    max_length = max(len(run) for run in fitness_runs)  # Find the longest progression
    aligned_runs = []
    for run in fitness_runs:
        if len(run) < max_length:
            # Pad with the last value
            aligned_runs.append(run + [run[-1]] * (max_length - len(run)))
        else:
            aligned_runs.append(run[:max_length])  # Truncate if needed
    return np.array(aligned_runs)
