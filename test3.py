import numpy as np
import matplotlib.pyplot as plt

# Parámetros comunes del algoritmo genético
num_generations = 100
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

# Selección por torneo podria modificar este paara tomar los mas aptos o cosas asi, estoy seleccionado a lo random
def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size)
        winner = participants[np.argmin(fitness[participants])]
        selected.append(population[winner])
    return np.array(selected)

# Cruce de un punto aqui igual se puede editar, podria hacer cruza a lo random o en dos puntos
def crossover(parent1, parent2, dimension):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, dimension-1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    else:
        return parent1, parent2

# Mutación
def mutate(individual, lower_bound, upper_bound):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(lower_bound, upper_bound)
    return individual

def genetic_algorithm(objective_function, dimension, lower_bound, upper_bound):
    # Inicializar la población
    population = initialize_population(population_size, dimension, lower_bound, upper_bound)
    
    # Evaluar la población inicial
    fitness = evaluate_population(population, objective_function)
    
    # Guardar la historia de la mejor solución
    best_fitness_history = []
    
    # Evolucionar la población
    for generation in range(num_generations):
        # Selección
        selected_population = tournament_selection(population, fitness)
        
        # Crear la próxima generación
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1]
            child1, child2 = crossover(parent1, parent2, dimension)
            next_generation.append(mutate(child1, lower_bound, upper_bound))
            next_generation.append(mutate(child2, lower_bound, upper_bound))
        
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

num_runs = 10

for problem in problems:
    results = []
    convergence_histories = []
    for run in range(num_runs):
        best_solution, best_fitness, best_fitness_history = genetic_algorithm(
            problem["objective_function"], 
            problem["dimension"], 
            problem["lower_bound"], 
            problem["upper_bound"]
        )
        results.append(best_fitness)
        convergence_histories.append(best_fitness_history)
        print(f'{problem["name"]} - Ejecución {run + 1}: Mejor Fitness = {best_fitness}')
    
    # Resultados finales
    print(f'\n{problem["name"]}:')
    print(f'Mejores Fitness: {results}')
    print(f'Promedio del mejor Fitness: {np.mean(results)}')
    print(f'Desviación estándar del mejor Fitness: {np.std(results)}')
    
    # Gráficos de convergencia
    for i, history in enumerate(convergence_histories[:-1]):  # son las representaciones, con 2 se ve bonito, 
        plt.plot(history, label=f'Ejecución {i+1}')
    
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.title(f'Convergencia del Algoritmo Genético - {problem["name"]}')
    plt.legend()
    plt.show()
