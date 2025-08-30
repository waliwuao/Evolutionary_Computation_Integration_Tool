import numpy as np
import math
from numba import jit

@jit(nopython=True)
def crossover_enhanced(parent1, parent2, crossover_rate, var_min, var_max):
    dim = parent1.shape[0]
    child = np.zeros_like(parent1)

    if np.random.random() < crossover_rate:
        mask = np.random.rand(dim) < 0.5
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]

        alpha = 0.3
        for i in range(dim):
            if not mask[i]:
                p1_gene = parent1[i]
                p2_gene = parent2[i]
                min_gene = min(p1_gene, p2_gene)
                max_gene = max(p1_gene, p2_gene)
                diff = max_gene - min_gene

                random_offset = np.random.uniform(-0.5 * diff * alpha, 0.5 * diff * alpha)
                child[i] = (p1_gene + p2_gene) / 2.0 + random_offset

        child = np.clip(child, var_min, var_max)
    else:
        child = parent1.copy() if np.random.rand() < 0.5 else parent2.copy()

    return child

@jit(nopython=True)
def mutate_adaptive_gaussian(individual, mutation_rate, var_min, var_max, population_std, progress):
    dim = individual.shape[0]
    mutated_individual = individual.copy()

    base_mutation_scale = 0.1 * (var_max - var_min)
    current_mutation_scale = base_mutation_scale * (1.0 - progress)**0.5 * (1.0 + population_std / max(1e-6, (var_max - var_min)))

    for i in range(dim):
        if np.random.random() < mutation_rate:
            mutation_amount = np.random.normal(0, current_mutation_scale[i])
            mutated_individual[i] += mutation_amount

    return np.clip(mutated_individual, var_min, var_max)

class GA:
    def __init__(self, pop_instance, crossover_rate=0.8, mutation_rate=0.05, elitism_ratio=0.1,
                 tournament_size=3, stagnation_threshold=50, recovery_mutation_factor=2.0,
                 reinitialization_ratio=0.1, diversity_decay_rate=0.5):
        self.pop = pop_instance
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.stagnation_threshold = stagnation_threshold
        self.recovery_mutation_factor = recovery_mutation_factor
        self.reinitialization_ratio = reinitialization_ratio
        self.diversity_decay_rate = diversity_decay_rate

        self.var_min = self.pop.var_min
        self.var_max = self.pop.var_max

        self.prev_best_fitness = -np.inf
        self.stagnation_count = 0

        self.initial_diversity = self._calculate_diversity(self.pop.get_population())
        self.population_std_dev = np.std(self.pop.get_population(), axis=0)

    def _calculate_diversity(self, population):
        if population.shape[0] <= 1:
            return 0.0
        return np.mean(np.std(population, axis=0))

    def _tournament_select(self, population, fitness, k):
        pop_size = population.shape[0]
        k = min(k, pop_size)
        
        best_individual = None
        best_fitness = -np.inf

        for _ in range(k):
            idx = np.random.randint(pop_size)
            if fitness[idx] > best_fitness:
                best_fitness = fitness[idx]
                best_individual = population[idx]
        return best_individual

    def evolve(self, max_iter):
        for t in range(max_iter):
            self.pop.update()
            current_pop = self.pop.get_population().copy()
            current_fitness = self.pop.get_fitness()
            pop_size, dim = current_pop.shape

            current_best_fitness = self.pop.get_best_fitness()
            if current_best_fitness > self.pop.get_best_fitness_all():
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            progress = t / float(max_iter)
            current_diversity = self._calculate_diversity(current_pop)
            
            diversity_penalty = self.diversity_decay_rate * max(0.0, (self.initial_diversity - current_diversity) / max(self.initial_diversity, 1e-9))
            stagnation_bonus = (self.stagnation_count / self.stagnation_threshold) * 0.5

            adaptive_mutation_rate = self.mutation_rate + diversity_penalty + stagnation_bonus
            adaptive_mutation_rate = np.clip(adaptive_mutation_rate, self.mutation_rate * 0.5, self.mutation_rate * 5.0)

            adaptive_crossover_rate = self.crossover_rate * (1.0 - progress * 0.5)
            adaptive_crossover_rate = np.clip(adaptive_crossover_rate, 0.5, self.crossover_rate)

            if self.stagnation_count >= self.stagnation_threshold:
                current_mutation_rate = adaptive_mutation_rate * self.recovery_mutation_factor
                current_mutation_rate = np.clip(current_mutation_rate, 0.05, 0.5)

                reinit_num = max(1, int(pop_size * self.reinitialization_ratio))
                reinit_indices = np.random.choice(pop_size, reinit_num, replace=False)
                for idx in reinit_indices:
                    current_pop[idx] = np.random.uniform(self.var_min, self.var_max, size=dim)
                
                self.pop.set_population(current_pop)
                self.pop.update()
                current_pop = self.pop.get_population().copy()
                current_fitness = self.pop.get_fitness()

                self.stagnation_count = 0
                self.prev_best_fitness = self.pop.get_best_fitness()
            else:
                current_mutation_rate = adaptive_mutation_rate
                self.prev_best_fitness = self.pop.get_best_fitness()

            elitism_size = max(1, int(pop_size * self.elitism_ratio))
            elite_indices = np.argsort(current_fitness)[-elitism_size:]
            elites = current_pop[elite_indices]

            next_gen_pop = []
            
            self.population_std_dev = np.std(current_pop, axis=0)
            self.population_std_dev = np.maximum(self.population_std_dev, 1e-6)

            for _ in range(pop_size - elitism_size):
                parent1 = self._tournament_select(current_pop, current_fitness, self.tournament_size)
                parent2 = self._tournament_select(current_pop, current_fitness, self.tournament_size)

                child = crossover_enhanced(parent1, parent2, adaptive_crossover_rate, self.var_min, self.var_max)
                child = mutate_adaptive_gaussian(child, current_mutation_rate, self.var_min, self.var_max, self.population_std_dev, progress)

                next_gen_pop.append(child)

            offspring_array = np.vstack(next_gen_pop)
            new_pop = np.vstack([elites, offspring_array])
            self.pop.set_population(new_pop)
            self.pop.print_best()
