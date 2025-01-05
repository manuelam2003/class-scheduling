import numpy as np
import random
from helpers import repair_balance
from parameters import affinity_matrix, n, m

def fitness(chromosome):
    return np.sum(affinity_matrix * chromosome)

def mutate(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        student = random.randint(0, n - 1)
        current_groups = np.where(chromosome[student] == 1)[0]
        if current_groups.size == 0:
            valid_groups = np.where(affinity_matrix[student] > 0)[0]
            new_group = random.choice(valid_groups)
            chromosome[student, new_group] = 1
        else:
            current_group = current_groups[0]
            valid_groups = [g for g in range(m) if g != current_group and affinity_matrix[student, g] > 0]
            if valid_groups:
                new_group = random.choice(valid_groups)
                chromosome[student, current_group] = 0
                chromosome[student, new_group] = 1
    return repair_balance(chromosome)

def select(population, fitness_scores, k=3):
    candidates = random.sample(list(zip(population, fitness_scores)), k)
    return max(candidates, key=lambda x: x[1])[0]

