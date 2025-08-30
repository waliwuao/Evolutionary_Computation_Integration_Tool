import numpy as np
from numba import jit

x_data = np.linspace(0, 2*np.pi, 100)
y_data = np.sin(x_data)
"""
以下几个函数都用于定义问题与判断解的好坏
"""

@jit(nopython=True)
def polynomial(x, coeffs):
    n = len(coeffs)
    result = 0.0
    for i in range(n):
        result += coeffs[i] * (x ** i)
    return result

@jit(nopython=True)
def fitness_function(coeffs):
    y_pred = np.zeros_like(x_data)
    for i in range(len(x_data)):
        y_pred[i] = polynomial(x_data[i], coeffs)
    mse = np.sum((y_pred - y_data) ** 2)
    return -mse

@jit(nopython=True)
def cal_fitness_batch(population):
    pop_size = population.shape[0]
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = fitness_function(population[i])
    return fitness

class Pop:
    def __init__(self, pop_size, dimention, var_min, var_max):
        self.count = 0

        self.pop_size = pop_size
        self.dimention = dimention
        self.var_min = var_min
        self.var_max = var_max
        self.best_fitness_all = None
        self.best_solution_all = None

        self.best_solution = None
        self.best_fitness = None

        self.population = self.init_population()
        self.fitness = self.cal_fitness()

    def init_population(self):
        return np.random.uniform(self.var_min, self.var_max, (self.pop_size, self.dimention))

    def cal_fitness(self):
        self.fitness = cal_fitness_batch(self.population)
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        if self.best_fitness_all is None or self.best_fitness > self.best_fitness_all:
            self.best_fitness_all = self.best_fitness
            self.best_solution_all = self.best_solution

        return self.fitness
    
    def get_population(self):
        return self.population
    
    def get_fitness(self):
        return self.fitness
    
    def get_best_solution(self):
        return self.best_solution
    
    def get_best_fitness(self):
        return self.best_fitness

    def set_population(self, population):
        self.population = population
        
    def update(self):
        self.cal_fitness()

    def get_best_solution_all(self):
        return self.best_solution_all
    
    def get_best_fitness_all(self):
        return self.best_fitness_all

    def print_best(self):
        self.count += 1
        print(f"第{self.count}次迭代，Best MSE: {self.best_fitness_all:.6f}")