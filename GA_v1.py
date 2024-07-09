# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 适应度函数,求取最大值
def fitness(x):
    return x + 16 * np.sin(5 * x) + 10 * np.cos(4 * x)

# 个体类
class indivdual:
    def __init__(self):
        self.x = 0  # 染色体编码
        self.fitness = 0  # 个体适应度值

    def __eq__(self, other):
        self.x = other.x
        self.fitness = other.fitness

# 初始化种群
#pop为种群适应度存储数组，N为个体数
def initPopulation(pop, N):
    for i in range(N):
        ind = indivdual()#个体初始化
        ind.x = np.random.uniform(-10, 10)#  个体编码。-10,10的正态分布，可以自己设定限定边界
        ind.fitness = fitness(ind.x)#计算个体适应度函数值
        pop.append(ind)#将个体适应度函数值添加进种群适应度数组pop

# 选择过程
def selection(N):
    # 种群中随机选择2个个体进行变异（这里没有用轮盘赌，直接用的随机选择）
    return np.random.choice(N, 2)

# 结合/交叉过程
def crossover(parent1, parent2):
    child1, child2 = indivdual(), indivdual()#父亲，母亲初始化
    child1.x = 0.9 * parent1.x + 0.1 * parent2.x #交叉0.9,0.1，可以设置其他系数
    child2.x = 0.1 * parent1.x + 0.9 * parent2.x
    child1.fitness = fitness(child1.x)#子1适应度函数值
    child2.fitness = fitness(child2.x)#子2适应度函数值
    return child1, child2
# 变异过程
def mutation(pop):
    # 种群中随机选择一个进行变异
    ind = np.random.choice(pop)
    # 用随机赋值的方式进行变异
    ind.x = np.random.uniform(-10, 10)
    ind.fitness = fitness(ind.x)


# 最终执行
def implement():
    # 种群中个体数量
    N = 40
    # 种群
    POP = []
    # 迭代次数
    iter_N = 400
    # 初始化种群
    initPopulation(POP, N)

# 进化过程
    for it in range(iter_N):#遍历每一代
        a, b = selection(N)#随机选择两个个体
        if np.random.random() < 0.65:  # 以0.65的概率进行交叉结合
            child1, child2 = crossover(POP[a], POP[b])
            new = sorted([POP[a], POP[b], child1, child2], key=lambda ind: ind.fitness, reverse=True)#将父母亲和子代进行比较，保留最好的两个
            POP[a], POP[b] = new[0], new[1]

        if np.random.random() < 0.1:  # 以0.1的概率进行变异
            mutation(POP)

        POP.sort(key=lambda ind: ind.fitness, reverse=True)

    return POP

if __name__ =='__main__':

   pop = implement()



   # 绘图代码
   def func(x):
       return x + 16 * np.sin(5 * x) + 10 * np.cos(4 * x)


   x = np.linspace(-10, 10, 100000)
   y = func(x)
   scatter_x = np.array([ind.x for ind in pop])
   scatter_y = np.array([ind.fitness for ind in pop])
   best=sorted(pop,key=lambda pop:pop.fitness,reverse=True)[0]#最佳点

   print('best_x',best.x)
   print('best_y',best.fitness)
   plt.plot(x, y)
   #plt.scatter(scatter_x, scatter_y, c='r')
   plt.scatter(best.x,best.fitness,c='g',label='best point')
   plt.legend()
   plt.show()

