import numpy as np
from numba import jit

@jit(nopython=True)
def update_particles(population, velocities, pbest, gbest, w, c1, c2):
    pop_size, dim = population.shape
    for i in range(pop_size):
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        cognitive = c1 * r1 * (pbest[i] - population[i])
        social = c2 * r2 * (gbest - population[i])
        velocities[i] = w * velocities[i] + cognitive + social
        population[i] += velocities[i]
    return population, velocities

class PSO:
    def __init__(self, pop, w=0.7, c1=1.4, c2=1.4, max_stagnation=50, mutation_rate=0.05, mutation_scale=0.1):
        self.pop = pop
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_stagnation = max_stagnation
        self.mutation_rate = mutation_rate  # 变异概率
        self.mutation_scale = mutation_scale  # 变异幅度
        
        self.var_min = pop.var_min
        self.var_max = pop.var_max
        
        pop_size, dim = pop.get_population().shape
        self.velocities = np.random.uniform(
            -abs(self.var_max - self.var_min) * 0.1,
            abs(self.var_max - self.var_min) * 0.1,
            (pop_size, dim)
        )
        self.pbest = pop.get_population().copy()
        self.pbest_fitness = pop.get_fitness().copy()
        
        self.prev_best = -np.inf
        self.stagnation_count = 0

    def evolve(self, max_iter):
        """进化主循环"""
        for t in range(max_iter):
            self.pop.update()
            current_pop = self.pop.get_population().copy()
            current_fitness = self.pop.get_fitness()
            gbest = self.pop.get_best_solution()
            
            for i in range(len(current_fitness)):
                if current_fitness[i] > self.pbest_fitness[i]:
                    self.pbest[i] = current_pop[i].copy()
                    self.pbest_fitness[i] = current_fitness[i]
            
            # 非线性惯性权重，前期探索，后期开发
            w = self.w * (0.5 + 0.5 * np.cos(np.pi * t / max_iter))
            
            current_pop, self.velocities = update_particles(
                current_pop, self.velocities, self.pbest, gbest, w, self.c1, self.c2
            )
            
            current_pop = np.clip(current_pop, self.var_min, self.var_max)
            
            mutations = np.random.rand(*current_pop.shape) < self.mutation_rate
            current_pop[mutations] += np.random.normal(0, self.mutation_scale, size=current_pop[mutations].shape)
            current_pop = np.clip(current_pop, self.var_min, self.var_max)
            
            if abs(self.pop.get_best_fitness() - self.prev_best) < 1e-8:
                self.stagnation_count += 1
                if self.stagnation_count >= self.max_stagnation:
                    self.velocities *= 1.2 
                    mutations = np.random.rand(*current_pop.shape) < self.mutation_rate * 2
                    current_pop[mutations] += np.random.normal(0, self.mutation_scale * 2, size=current_pop[mutations].shape)
                    current_pop = np.clip(current_pop, self.var_min, self.var_max)
                    self.stagnation_count = 0
            else:
                self.stagnation_count = 0
            self.prev_best = self.pop.get_best_fitness()
            
            self.pop.set_population(current_pop)
            self.pop.print_best()
