import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo genético
num_generations = 100
population_size = 50
mutation_rate = 0.01
crossover_rate = 0.7
dimension = 10
lower_bound = -5.12
upper_bound = 5.12

# Definición de la función objetivo
def objective_function(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

# Inicialización de la población
def initialize_population(size, dimension, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (size, dimension))

# Evaluación de la población
def evaluate_population(population):
    return np.array([objective_function(individual) for individual in population])

# Selección por torneo
def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size)
        winner = participants[np.argmin(fitness[participants])]
        selected.append(population[winner])
    return np.array(selected)

# Cruce de un punto
def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, dimension-1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    else:
        return parent1, parent2

# Mutación
def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(lower_bound, upper_bound)
    return individual

# Algoritmo Genético
def genetic_algorithm():
    # Inicializar la población
    population = initialize_population(population_size, dimension, lower_bound, upper_bound)
    
    # Evaluar la población inicial
    fitness = evaluate_population(population)
    
    # Evolucionar la población
    for generation in range(num_generations):
        # Selección
        selected_population = tournament_selection(population, fitness)
        
        # Crear la próxima generación
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
        
        population = np.array(next_generation)
        fitness = evaluate_population(population)
        
        # Guardar el mejor resultado de la generación
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        print(f"Generación {generation}: Mejor Fitness = {best_fitness}")
    
    # Devolver el mejor resultado
    return best_individual, best_fitness

# Ejecutar el algoritmo genético
best_solution, best_fitness = genetic_algorithm()
print(f"Mejor solución encontrada: {best_solution}")
print(f"Mejor fitness: {best_fitness}")
