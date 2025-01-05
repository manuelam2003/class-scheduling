import matplotlib.pyplot as plt
import numpy as np

def plot_results(results):
    # Plotting results
    plt.figure(figsize=(12, 8))

    # Best fitness bar plot
    plt.subplot(1, 2, 1)
    avg_best_fitness = {method: np.mean(results[method]["best_fitness"]) for method in results.keys()}
    std_best_fitness = {method: np.std(results[method]["best_fitness"]) for method in results.keys()}
    methods = list(results.keys())
    plt.bar(methods, avg_best_fitness.values(), yerr=std_best_fitness.values(), capsize=5, color='skyblue')
    plt.xlabel("Crossover Methods")
    plt.ylabel("Best Fitness (Mean ± Std)")
    plt.title("Comparison of Best Fitness")

    # Average fitness progress
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        plt.plot(data["avg_fitness_progress"], label=method)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Fitness Progression Over Generations")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_convergence_speed(convergence_speed):
    plt.figure(figsize=(12, 6))
    plt.bar(convergence_speed.keys(), convergence_speed.values(), color='lightgreen')
    plt.xlabel('Crossover Methods')
    plt.ylabel('Generations to Converge')
    plt.title('Convergence Speed of Crossover Methods')
    plt.show()
