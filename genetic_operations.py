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

def tournament_selection(population, fitness_scores, k=3):
    candidates = random.sample(list(zip(population, fitness_scores)), k)
    return max(candidates, key=lambda x: x[1])[0]

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    
    pick = random.uniform(0, total_fitness)
    
    current = 0
    for individual, fitness in zip(population, fitness_scores):
        current += fitness
        if current > pick:
            return individual

def stochastic_universal_sampling(population, fitness_scores, num_to_select=1):
    # Calculate the total fitness
    total_fitness = sum(fitness_scores)
    
    # Calculate the distance between pointers
    pointer_spacing = total_fitness / num_to_select
    
    # Generate a random start point
    start_point = random.uniform(0, pointer_spacing)
    
    # Generate the pointers
    pointers = [start_point + i * pointer_spacing for i in range(num_to_select)]
    
    # Select individuals
    selected_individuals = []
    current = 0
    for pointer in pointers:
        while pointer > fitness_scores[current]:
            pointer -= fitness_scores[current]
            current = (current + 1) % len(fitness_scores)
        selected_individuals.append(population[current])
    
    # Return the selected individual(s)
    return selected_individuals if num_to_select > 1 else selected_individuals[0]
