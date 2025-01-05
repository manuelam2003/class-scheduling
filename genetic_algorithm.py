from helpers import initialize_population
from genetic_operations import *
from crossover_methods import *

def genetic_algorithm(pop_size=50, generations=100, mutation_rate=0.1, elitism=1, patience=10, crossover_method=None, select=tournament_selection):
    population = initialize_population(pop_size)
    no_improvement = 0
    best_fitness = float('-inf')
    avg_fitness_progress = []  # To store average fitness per generation


    for generation in range(generations):
        fitness_scores = [fitness(chrom) for chrom in population]
        avg_fitness = np.mean(fitness_scores)
        avg_fitness_progress.append(avg_fitness)  # Track average fitness
        max_fitness = max(fitness_scores)

        if max_fitness > best_fitness:
            best_fitness = max_fitness
            no_improvement = 0
        else:
            no_improvement += 1

        # Early stopping if no improvement for `patience` generations
        if no_improvement >= patience:
            # print(f"Early stopping at generation {generation}")
            break

        # Elitism: Select the best individuals to carry over
        elite_indices = np.argsort(fitness_scores)[-elitism:]
        elites = [population[i] for i in elite_indices]

        # Create a new population
        new_population = elites.copy()  # Start with elites
        while len(new_population) < pop_size:
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)
            child1, child2 = crossover_method(parent1, parent2)  # Use the specified crossover method
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < pop_size:  # Add second child only if space remains
                new_population.append(mutate(child2, mutation_rate))

        population = new_population
        # print(f"Generation {generation}: Best fitness = {max_fitness}, Avg fitness = {avg_fitness:.2f}")

    # Return the best solution
    fitness_scores = [fitness(chrom) for chrom in population]
    best_index = np.argmax(fitness_scores)



    return population[best_index], avg_fitness_progress