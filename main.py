from pop import Pop
from bfo import BFO


pop_size = 100
dim = 8
var_min = -1
var_max = 1
max_iter = 10000

pop = Pop(pop_size, dim, var_min, var_max)

bfo = BFO(pop)
bfo.evolve(max_iter)













