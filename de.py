import numpy as np
from numba import jit
from pop import fitness_function

@jit(nopython=True)
def de_mutation(population, target_idx, f):
    pop_size, dim = population.shape
    idxs = np.array([i for i in range(pop_size) if i != target_idx], dtype=np.int64)
    a, b, c = np.random.choice(idxs, 3, replace=False)
    return population[a] + f * (population[b] - population[c])

@jit(nopython=True)
def de_crossover(target, mutant, cr):
    dim = target.shape[0]
    trial = target.copy()
    j_rand = np.random.randint(dim)
    for j in range(dim):
        if np.random.random() < cr or j == j_rand:
            trial[j] = mutant[j]
    return trial

class DE:
    def __init__(self, pop, f=0.5, cr=0.7):
        self.pop = pop
        self.f = f 
        self.cr = cr
        
        self.var_min = pop.var_min
        self.var_max = pop.var_max

    def evolve(self, max_iter):
        for t in range(max_iter):
            self.pop.update()
            current_pop = self.pop.get_population().copy()
            pop_size, dim = current_pop.shape
            new_pop = current_pop.copy()
            
            f = self.f * (0.5 + 0.5 * np.exp(-t/max_iter))
            cr = self.cr * (1 - t/max_iter)
            
            for i in range(pop_size):
                mutant = de_mutation(current_pop, i, f)
                trial = de_crossover(current_pop[i], mutant, cr)
                trial = np.clip(trial, self.var_min, self.var_max)
                
                trial_fitness = fitness_function(trial)
                if trial_fitness > self.pop.get_fitness()[i]:
                    new_pop[i] = trial
            
            self.pop.set_population(new_pop)
            self.pop.print_best()