# 计算工具库说明文档

## 项目概述
该项目包含多种优化算法实现，用于求解多项式系数拟合问题（以正弦函数拟合为例）。主要实现了粒子群优化(PSO)、差分进化(DE)、遗传算法(GA)和狼群算法（WPA）等优化算法，通过种群迭代寻找最优多项式系数，最小化预测值与真实值的均方误差(MSE)。同时这些代码均使用了numba进行加速，在运行之前，请先确保你已经正确的安装了numba。
```python
pip install numba
```

## 核心文件说明

### 1. main.py
主程序入口文件，用于初始化种群和优化算法并执行优化过程。

**代码示例解析**：
```python
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

#获取最优解与最优适应度
best_solution = pop.get_best_solution_all()
best_fitness = pop.get_best_fitness_all()
```

**使用说明**：
- 替换导入的算法类（如`GA`替换为`PSO`、`DE`等）可切换不同优化算法
- 通过调整`pop_size`、`dim`等参数控制优化过程
- 执行`evolve()`方法启动优化迭代

### 2. pop.py
种群类定义及适应度计算核心实现。

#### Pop类
种群管理核心类，负责种群初始化、适应度计算和最优解跟踪。

**初始化参数**：
- `pop_size`：种群大小（个体数量）
- `dimention`：问题维度（多项式系数数量）
- `var_min`：变量取值下限
- `var_max`：变量取值上限

**主要方法**：
- `init_population()`：初始化种群，随机生成在[var_min, var_max]范围内的个体
- `cal_fitness()`：计算种群中所有个体的适应度，更新当前最优解和全局最优解
- `get_population()`：返回当前种群
- `set_population(population)`：设置新种群
- `update()`：更新种群适应度和最优解
- `print_best()`：打印当前迭代的最优MSE

#### 适应度计算函数
- `polynomial(x, coeffs)`：计算多项式值，`coeffs`为多项式系数
- `fitness_function(coeffs)`：计算适应度（-MSE，因为优化算法默认最大化适应度）
- `cal_fitness_batch(population)`：批量计算种群中所有个体的适应度

## 优化算法核心参数说明

### 1. PSO（粒子群优化，pso.py）
```python
PSO(pop, w=0.7, c1=1.4, c2=1.4, max_stagnation=50, mutation_rate=0.05, mutation_scale=0.1)
```
- `pop`：Pop类实例
- `w`：惯性权重
- `c1`：个体认知系数
- `c2`：社会学习系数
- `max_stagnation`：最大停滞代数（超过此值则增强变异）
- `mutation_rate`：变异概率
- `mutation_scale`：变异幅度

### 2. DE（差分进化，de.py）
```python
DE(pop, f=0.5, cr=0.7)
```
- `pop`：Pop类实例
- `f`：差分变异系数
- `cr`：交叉概率

### 3. GA（遗传算法，ga.py）
```python
GA(pop_instance, crossover_rate=0.8, mutation_rate=0.05, elitism_ratio=0.1,
   tournament_size=3, stagnation_threshold=50, recovery_mutation_factor=2.0,
   reinitialization_ratio=0.1, diversity_decay_rate=0.5)
```
- `pop_instance`：Pop类实例
- `crossover_rate`：交叉概率
- `mutation_rate`：变异概率
- `elitism_ratio`：精英保留比例
- `tournament_size`：锦标赛选择规模
- `stagnation_threshold`：停滞阈值
- `recovery_mutation_factor`：恢复阶段变异放大因子
- `reinitialization_ratio`：种群重新初始化比例
- `diversity_decay_rate`：多样性衰减率

### 4. WPA（wpa.py）
```python
WPA(pop_instance, beta=0.7, velocity_strength=0.7, step_size=0.1, chaos_prob=0.1,
    max_stagnation=50, stagnation_recovery_factor=1.5, reinitialization_ratio=0.2,
    distance_threshold=0.1, diversity_decay_rate=0.5,
    chaos_std_scale_factor=0.5,
    perpendicular_step_scale_factor=0.5,
    stagnation_convergence_threshold=0.01,
    consecutive_stagnation_reset_ratio=0.3)
```
- `pop_instance`：Pop类实例
- `beta`：β群体比例
- `velocity_strength`：速度强度
- `step_size`：步长
- `chaos_prob`：混沌扰动概率
- `max_stagnation`：最大停滞代数
- `stagnation_recovery_factor`：停滞恢复因子
- `reinitialization_ratio`：重新初始化比例
- `distance_threshold`：距离阈值
- `diversity_decay_rate`：多样性衰减率

## 使用流程
1. 导入所需的种群类(Pop)和优化算法类（如PSO）
2. 初始化种群参数（大小、维度、变量范围）
3. 创建种群实例
4. 初始化优化算法实例，传入种群对象和算法参数
5. 调用优化算法的`evolve(max_iter)`方法执行优化
6. 算法会在迭代过程中打印每次迭代的最优MSE

## 注意事项
- 所有算法均使用`numba`的`@jit`装饰器加速计算
- 目标函数默认为拟合正弦函数（`y_data = np.sin(x_data)`），可在`pop.py`中修改
- 多项式阶数由`dim`参数控制（`dim`=8表示8阶多项式）
- 适应度函数使用负MSE（-MSE），因此算法均以最大化适应度为目标
