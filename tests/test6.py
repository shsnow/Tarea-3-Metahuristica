import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo genético
num_generations = 100
elite_size = 2  # Número de individuos élite
population_size = 50
mutation_rate = 0.01
crossover_rate = 0.7

# Funciones objetivo
def f1(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def f2(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def f3(x):
    return np.sum(np.sin(x) * (np.sin((np.arange(1, len(x) + 1) * x**2) / np.pi))**20)

# Inicialización de la población
def initialize_population(size, dimension, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (size, dimension))

# Evaluación de la población
def evaluate_population(population, objective_function):
    return np.array([objective_function(individual) for individual in population])

# Selección por torneo
def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size)
        winner = participants[np.argmin(fitness[participants])]
        selected.append(population[winner])
    return np.array(selected)

# Selección elitista
def elitist_selection(population, fitness, elite_size):
    elite_indices = np.argsort(fitness)[:elite_size]
    return population[elite_indices]

# Cruce de dos puntos
def crossover(parent1, parent2, dimension, crossover_rate):
    if np.random.rand() < crossover_rate:
        point1 = np.random.randint(1, dimension-1)
        point2 = np.random.randint(point1, dimension)
        child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        return child1, child2
    else:
        return parent1, parent2

# Mutación no uniforme
def mutate(individual, lower_bound, upper_bound, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.uniform(-0.1, 0.1) * (upper_bound - lower_bound)
            individual[i] = np.clip(individual[i], lower_bound, upper_bound)
    return individual

# Algoritmo Genético con selección elitista y cruce de dos puntos
def genetic_algorithm(objective_function, dimension, lower_bound, upper_bound, population_size, mutation_rate, crossover_rate):
    # Inicializar la población
    population = initialize_population(population_size, dimension, lower_bound, upper_bound)
    
    # Evaluar la población inicial
    fitness = evaluate_population(population, objective_function)
    
    # Guardar la historia de la mejor solución
    best_fitness_history = []
    
    # Evolucionar la población
    for generation in range(num_generations):
        # Selección elitista
        elite = elitist_selection(population, fitness, elite_size)
        
        # Selección
        selected_population = tournament_selection(population, fitness)
        
        # Crear la próxima generación
        next_generation = elite.tolist()
        for i in range(0, population_size - elite_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1]
            child1, child2 = crossover(parent1, parent2, dimension, crossover_rate)
            next_generation.append(mutate(child1, lower_bound, upper_bound, mutation_rate))
            next_generation.append(mutate(child2, lower_bound, upper_bound, mutation_rate))
        
        population = np.array(next_generation)
        fitness = evaluate_population(population, objective_function)
        
        # Guardar el mejor resultado de la generación
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        best_fitness_history.append(best_fitness)
    
    # Devolver el mejor resultado y la historia de la convergencia
    return best_individual, best_fitness, best_fitness_history

# Configuración de los problemas
problems = [
    {"objective_function": f1, "dimension": 10, "lower_bound": -5.12, "upper_bound": 5.12, "name": "Problema 1"},
    {"objective_function": f2, "dimension": 20, "lower_bound": -30, "upper_bound": 30, "name": "Problema 2"},
    {"objective_function": f3, "dimension": 5, "lower_bound": 0, "upper_bound": np.pi, "name": "Problema 3"}
]

# Valores de parámetros a probar
population_sizes = [20, 50, 100]
mutation_rates = [0.01, 0.05, 0.1]
crossover_rates = [0.5, 0.7, 0.9]

num_runs = 10

# Almacenar resultados
results = []

for problem in problems:
    for pop_size in population_sizes:
        for mut_rate in mutation_rates:
            for cross_rate in crossover_rates:
                fitness_results = []
                for run in range(num_runs):
                    best_solution, best_fitness, best_fitness_history = genetic_algorithm(
                        problem["objective_function"], 
                        problem["dimension"], 
                        problem["lower_bound"], 
                        problem["upper_bound"],
                        pop_size,
                        mut_rate,
                        cross_rate
                    )
                    fitness_results.append(best_fitness)
                mean_fitness = np.mean(fitness_results)
                std_fitness = np.std(fitness_results)
                result = {
                    "problem": problem["name"],
                    "population_size": pop_size,
                    "mutation_rate": mut_rate,
                    "crossover_rate": cross_rate,
                    "fitness_results": fitness_results,
                    "mean_fitness": mean_fitness,
                    "std_fitness": std_fitness,
                    "best_fitness": min(fitness_results)  # Agregar el mejor fitness alcanzado
                }
                results.append(result)
                print(f'{problem["name"]} - PSize: {pop_size}, MRate: {mut_rate}, CRate: {cross_rate}, Mean Fitness: {mean_fitness}, Std Fitness: {std_fitness}, Best Fitness: {min(fitness_results)}')

# Análisis de resultados
for problem in problems:
    problem_results = [res for res in results if res["problem"] == problem["name"]]
    best_result = min(problem_results, key=lambda x: x["mean_fitness"])
    print(f'\n{problem["name"]} - Mejor configuración:')
    print(f'  Tamaño de población: {best_result["population_size"]}')
    print(f'  Tasa de mutación: {best_result["mutation_rate"]}')
    print(f'  Tasa de cruce: {best_result["crossover_rate"]}')
    print(f'  Fitness promedio: {best_result["mean_fitness"]}')
    print(f'  Desviación estándar del fitness: {best_result["std_fitness"]}')
    print(f'  Mejor fitness alcanzado: {best_result["best_fitness"]}')

    # Gráfico de convergencia para la mejor configuración
    best_fitness_history = None
    for run in range(num_runs):
        _, _, best_fitness_history = genetic_algorithm(
            problem["objective_function"], 
            problem["dimension"], 
            problem["lower_bound"], 
            problem["upper_bound"],
            best_result["population_size"],
            best_result["mutation_rate"],
            best_result["crossover_rate"]
        )
        if run < 2:
            plt.plot(best_fitness_history, label=f'Ejecución {run + 1}')
    
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.title(f'Convergencia del Algoritmo Genético - {problem["name"]}')
    plt.legend()
    plt.show()
