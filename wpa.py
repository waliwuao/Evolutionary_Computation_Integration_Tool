import numpy as np
import math
from numba import jit

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
    def __init__(self, pop_size, dimension, var_min, var_max):
        self.count = 0
        self.pop_size = pop_size
        self.dimension = dimension
        self.var_min = var_min
        self.var_max = var_max
        self.best_fitness_all = -np.inf
        self.best_solution_all = None

        self.best_solution = None
        self.best_fitness = -np.inf

        self.population = self.init_population()
        self.fitness = self.cal_fitness()

    def init_population(self):
        return np.random.uniform(self.var_min, self.var_max, (self.pop_size, self.dimension))

    def cal_fitness(self):
        self.fitness = cal_fitness_batch(self.population)
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]

        if self.best_fitness > self.best_fitness_all:
            self.best_fitness_all = self.best_fitness
            self.best_solution_all = self.best_solution.copy()
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
        print(f"第{self.count}次迭代，Best MSE: {-self.best_fitness_all:.6f}")

@jit(nopython=True)
def clip_scalar(x, a_min, a_max):
    return max(a_min, min(x, a_max))

@jit(nopython=True)
def update_beta_pop_deep_opt(current_pop, beta_num, best_solution, global_best_influences,
                             velocity_strength, step_size, chaos_prob,
                             var_min, var_max, local_mean, distance_threshold,
                             population_std_dev, progress, beta_group_mean_dist):
    pop_size, dim = current_pop.shape
    chaos_range_factor = 1.0 + 2.0 * (1.0 - progress) * (population_std_dev[0] / max(1e-6, (var_max - var_min)))

    for i in range(beta_num):
        distance_global = best_solution - current_pop[i]
        distance_local_mean = local_mean - current_pop[i]
        distance_norm_global = np.linalg.norm(distance_global)
        distance_local_beta_avg = beta_group_mean_dist

        weight_global_best = 0.4 + 0.6 * (i / beta_num)
        weight_local_mean = 1.0 - weight_global_best
        global_weight_scale = clip_scalar(distance_norm_global / (distance_threshold * 2.0), 0.1, 1.0)
        local_weight_scale = 1.0 - global_weight_scale

        influence_vector = (
            global_weight_scale * weight_global_best * distance_global +
            local_weight_scale * weight_local_mean * distance_local_mean
        )

        adaptive_step_modifier = 1.0 + 0.5 * math.exp(-distance_norm_global / (distance_threshold * 2.0))
        current_pop[i] += velocity_strength * adaptive_step_modifier * influence_vector

        if dim >= 2 and (distance_norm_global > distance_threshold * 0.5 or step_size > 0.05):
            if distance_norm_global > 1e-8:
                random_vec = np.random.randn(dim)
                dot_product = np.dot(random_vec, distance_global)
                norm_sq_global = distance_norm_global**2 + 1e-8
                perpendicular_direction = random_vec - (dot_product / norm_sq_global) * distance_global

                perpendicular_norm = np.linalg.norm(perpendicular_direction)
                if perpendicular_norm > 1e-8:
                    perpendicular_direction /= perpendicular_norm
                    perpendicular_step_scale = step_size * (0.7 + 0.3 * progress)
                    perpendicular_step = perpendicular_step_scale * (1.0 + population_std_dev[0] / max(1e-6, (var_max - var_min)))
                    current_pop[i] += np.random.uniform(-perpendicular_step, perpendicular_step) * perpendicular_direction

        if np.random.random() < chaos_prob:
            chaos_range = step_size * chaos_range_factor
            current_pop[i] += np.random.uniform(-chaos_range, chaos_range, size=dim)

        current_pop[i] = np.clip(current_pop[i], var_min, var_max)

    return current_pop

@jit(nopython=True)
def update_remaining_pop_deep_opt(current_pop, beta_num, step_size, var_min, var_max, progress, population_std_dev):
    pop_size, dim = current_pop.shape
    decay_factor = math.exp(-3.0 * progress)
    std_scale = 1.0 + (population_std_dev[0] / max(1e-6, (var_max - var_min))) * 0.5
    std_scale = clip_scalar(std_scale, 0.8, 1.5)
    adaptive_step = step_size * decay_factor * std_scale
    jitter_scale = step_size * decay_factor * 0.1 * (1.0 + population_std_dev[0] / max(1e-6, (var_max - var_min)))

    for i in range(beta_num, pop_size):
        current_pop[i] += np.random.uniform(-adaptive_step, adaptive_step, size=dim)
        current_pop[i] += np.random.uniform(-jitter_scale, jitter_scale, size=dim)
        current_pop[i] = np.clip(current_pop[i], var_min, var_max)

    return current_pop

class WPA:
    def __init__(self, pop_instance, beta=0.7, velocity_strength=0.7, step_size=0.1, chaos_prob=0.1,
                 max_stagnation=50, stagnation_recovery_factor=1.5, reinitialization_ratio=0.2,
                 distance_threshold=0.1, diversity_decay_rate=0.5,
                 chaos_std_scale_factor=0.5,
                 perpendicular_step_scale_factor=0.5,
                 stagnation_convergence_threshold=0.01,
                 consecutive_stagnation_reset_ratio=0.3
                 ):
        self.pop = pop_instance
        self.beta = beta
        self.initial_velocity_strength = velocity_strength
        self.velocity_strength = velocity_strength
        self.initial_step_size = step_size
        self.step_size = step_size
        self.initial_chaos_prob = chaos_prob
        self.chaos_prob = chaos_prob
        self.max_stagnation = max_stagnation
        self.stagnation_recovery_factor = stagnation_recovery_factor
        self.reinitialization_ratio = reinitialization_ratio
        self.distance_threshold = distance_threshold
        self.diversity_decay_rate = diversity_decay_rate

        self.chaos_std_scale_factor = chaos_std_scale_factor
        self.perpendicular_step_scale_factor = perpendicular_step_scale_factor
        self.stagnation_convergence_threshold = stagnation_convergence_threshold
        self.consecutive_stagnation_reset_ratio = reinitialization_ratio

        self.var_min = self.pop.var_min
        self.var_max = self.pop.var_max

        self.prev_best_fitness = -np.inf
        self.stagnation_count = 0

        initial_pop = self.pop.get_population()
        self.initial_diversity_std = self._calculate_diversity_std(initial_pop)
        self.initial_diversity_dist = self._calculate_diversity_dist(initial_pop)

        self.population_std_dev = np.std(initial_pop, axis=0)
        self.population_avg_dist = self._calculate_diversity_dist(initial_pop)

        self.history_best_solution = None
        self.history_best_fitness = -np.inf

        self._original_step_size = self.initial_step_size
        self._original_chaos_prob = self.initial_chaos_prob

    def _calculate_diversity_std(self, population):
        if population.shape[0] <= 1:
            return 0.0
        return np.mean(np.std(population, axis=0))

    def _calculate_diversity_dist(self, population):
        if population.shape[0] <= 1:
            return 0.0
        return 0.0

    def evolve(self, max_iter):
        for t in range(max_iter):
            self.pop.update()

            current_pop = self.pop.get_population().copy()
            pop_size, dim = current_pop.shape
            beta_num = max(1, int(pop_size * self.beta))

            best_solution = self.pop.get_best_solution()
            current_best_fitness = self.pop.get_best_fitness()

            if current_best_fitness > self.history_best_fitness + self.stagnation_convergence_threshold:
                self.history_best_fitness = current_best_fitness
                self.history_best_solution = best_solution.copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            progress = t / float(max_iter)
            self.velocity_strength = self.initial_velocity_strength * math.exp(-1.0 * progress)
            self.step_size = self._original_step_size * math.exp(-2.0 * progress)
            self.population_std_dev = np.std(current_pop, axis=0)

            current_diversity_std = self._calculate_diversity_std(current_pop)
            diversity_std_ratio = current_diversity_std / max(self.initial_diversity_std, 1e-9)
            chaos_bonus_stagnation = 0.25 * (self.stagnation_count / self.max_stagnation) if self.stagnation_count > 0 else 0.0
            chaos_penalty_diversity = self.diversity_decay_rate * max(0.0, diversity_std_ratio - 1.0)

            self.chaos_prob = self._original_chaos_prob + chaos_bonus_stagnation - chaos_penalty_diversity
            self.chaos_prob = np.clip(self.chaos_prob, 0.05, 0.4)

            if self.stagnation_count >= self.max_stagnation:
                self.step_size *= self.stagnation_recovery_factor
                self.chaos_prob = np.clip(self.chaos_prob * 1.5, 0.1, 0.5)
                reinit_num = max(1, int(pop_size * self.consecutive_stagnation_reset_ratio))
                reinit_indices = np.random.choice(pop_size, reinit_num, replace=False)
                for idx in reinit_indices:
                    current_pop[idx] = np.random.uniform(self.var_min, self.var_max, size=dim)

                self.pop.set_population(current_pop)
                self.pop.update()
                current_pop = self.pop.get_population().copy()
                self.stagnation_count = 0
                self.prev_best_fitness = self.pop.get_best_fitness()

            self.prev_best_fitness = self.pop.get_best_fitness()

            local_mean = np.mean(current_pop[:beta_num], axis=0)
            if beta_num > 1:
                beta_group_mean_dist = np.mean(np.linalg.norm(current_pop[:beta_num] - local_mean, axis=1))
            else:
                beta_group_mean_dist = 0.0

            current_pop = update_beta_pop_deep_opt(
                current_pop, beta_num, best_solution,
                global_best_influences=None,
                velocity_strength=self.velocity_strength,
                step_size=self.step_size,
                chaos_prob=self.chaos_prob,
                var_min=self.var_min,
                var_max=self.var_max,
                local_mean=local_mean,
                distance_threshold=self.distance_threshold,
                population_std_dev=self.population_std_dev,
                progress=progress,
                beta_group_mean_dist=beta_group_mean_dist
            )

            current_pop = update_remaining_pop_deep_opt(
                current_pop, beta_num, self.step_size,
                self.var_min, self.var_max,
                progress,
                population_std_dev=self.population_std_dev
            )

            self.pop.set_population(current_pop)
            self.pop.print_best()

        print(f"Final best solution found: {self.pop.get_best_solution_all()}")
        print(f"Final best fitness (MSE): {-self.pop.get_best_fitness_all():.6f}")
