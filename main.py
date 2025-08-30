from pop import Pop
from ga import GA  # 可替换为其他算法如PSO、DE、WPA

# 参数设置
pop_size = 100  # 种群大小
dim = 8         # 维度（多项式系数数量）
var_min = -1    # 变量最小值
var_max = 1     # 变量最大值
max_iter = 1000  # 最大迭代次数

# 初始化种群
pop = Pop(pop_size, dim, var_min, var_max)

# 初始化优化算法并执行
ga = GA(pop)
ga.evolve(max_iter)
