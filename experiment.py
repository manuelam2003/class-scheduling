import numpy as np
from helpers import align_fitness_progressions
from parameters import pop_size, generations, mutation_rate, elitism, patience, trials, crossover_methods
from genetic_algorithm import genetic_algorithm
from crossover_methods import *
from plotting import plot_results, plot_convergence_speed
from genetic_operations import fitness
import random

# Define crossover methods
crossover_methods.update({
    # "Single-point": single_point_crossover,
    # "Two-point": two_point_crossover,
    # "Heuristic": heuristic_crossover,
    # "Heuristic 2": heuristic_crossover2,
    # "Stochastic heuristic": stochastic_heuristic_crossover,
    "Uniform": uniform_crossover,
    "Bernouilli": bernouilli_crossover,
})

# Storage for results
results = {method: {"best_fitness": [], "avg_fitness_progress": []} for method in crossover_methods.keys()}

# Run experiments
for method_name, method in crossover_methods.items():
    print(f"Testing {method_name} crossover...")
    best_fitness_runs = []
    avg_fitness_runs = []

    for trial in range(trials):
        random.seed(trial)
        np.random.seed(trial)
        best_solution, avg_fitness_per_gen = genetic_algorithm(
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            elitism=elitism,
            patience=patience,
            crossover_method=method,
        )

        final_fitness = fitness(best_solution)
        best_fitness_runs.append(final_fitness)
        avg_fitness_runs.append(avg_fitness_per_gen)

        # Debugging: Print the fitness values for each generation
        print(f"Run {trial + 1}: {method_name} - Best fitness: {fitness(best_solution)}")
        print(f"Run {trial + 1}: {method_name} - Avg fitness per generation: {avg_fitness_per_gen[:5]+avg_fitness_per_gen[-5:]}")

    # Align and store results
    aligned_avg_fitness_runs = align_fitness_progressions(avg_fitness_runs)
    results[method_name]["best_fitness"] = best_fitness_runs
    results[method_name]["avg_fitness_progress"] = np.mean(aligned_avg_fitness_runs, axis=0)

convergence_speed = {}
for method_name, method_results in results.items():
    threshold = 0.9 * min(method_results["best_fitness"])
    generations_to_converge = []

    for generation_index, fitness_value in enumerate(method_results["avg_fitness_progress"]):
        if fitness_value >= threshold:
            generations_to_converge.append(generation_index)
            break  # Once the threshold is reached, stop checking further generations
    if generations_to_converge == []:
        generations_to_converge.append(0)
    convergence_speed[method_name] = np.mean(generations_to_converge)
    print(f"Convergence speed for {method_name}: {convergence_speed[method_name]} generations")

# Plot results
plot_results(results, "Crossover Methods")
plot_convergence_speed(convergence_speed)