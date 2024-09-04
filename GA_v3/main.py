from pymoo.operators.crossover.sbx import SBX
from pymoo.core.individual import Individual
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from Tool.LHS import lhs_sample
from Tool.Benchmark_fun import result_calculation
import numpy as np


class GA(object):
    def __init__(self, pop, lb, ub, pc=1.0, pvc=1.0, eta_c=15, pm=1.0, eta_m=15, pvm=None):
        """
        :param pop: 种群
        :param lb: 下边界
        :param ub: 上边界
        :param pc: 一对父代进行交叉操作发生的概率
        :param pvc:父代内每个变量发生交叉的概率
        :param eta_c:控制交叉操作的分布指数
        :param pm:父代进行变异操作发生的概率
        :param eta_m:父代内每个变量发生变异的概率
        :param pvm:控制变异操作的分布指数
        """
        self.pop = pop
        self.lb = lb
        self.ub = ub
        self.pc = pc
        self.pvc = pvc
        self.eta_c = eta_c
        self.pm = pm
        self.eta_m = eta_m
        self.pop_sbx = None
        self.pop_mut = None
        self.pop_all = None
        self.best_ind = []
        if pvm is not None:
            self.pvm = pvm
        else:
            self.pvm = 1/pop.shape[1]

    def sbx(self, pop_sbx=None):
        if pop_sbx is None:
            pop_sbx = self.pop
        n = pop_sbx.shape[0]
        d = pop_sbx.shape[1]
        parents = []
        problem = Problem(n_var=d, xl=self.lb, xu=self.ub)
        for i in range(n):
            p1 = Individual(X=self.pop[i])
            idx = np.delete(np.arange(n), i)
            j = np.random.choice(idx)
            p2 = Individual(X=self.pop[j])
            parents.append([p1, p2])
        off = SBX(prob=self.pc, prob_var=self.pvc, eta=self.eta_c).do(problem, parents)
        self.pop_sbx = off.get("X")
        return self.pop_sbx

    def mutation(self, pop_mut=None):
        if pop_mut is None:
            if self.pop_sbx is not None:
                pop_mut = self.pop_sbx.copy()
            else:
                raise Exception("GA.sbx() don't run")
        d = pop_mut.shape[1]
        problem = Problem(n_var=d, xl=self.lb, xu=self.ub)
        pop_temp = Population.new(X=pop_mut)
        mutation = PolynomialMutation(prob=self.pm, prob_var=self.pvm, eta=self.eta_m)
        off = mutation(problem, pop_temp)
        self.pop_mut = off.get("X")
        self.pop_all = np.vstack((self.pop, self.pop_sbx))
        self.pop_all = np.vstack((self.pop_all, self.pop_mut))
        self.pop_all = np.unique(self.pop_all, axis=0)
        return self.pop_mut

    def select(self, y_sel, pop_sel=None):
        if pop_sel is None:
            if self.pop_all is not None:
                pop_sel = self.pop_all
            else:
                raise Exception("GA.mutation() don't run")
        n = self.pop.shape[0]
        sort_idx = np.argsort(y_sel)
        self.pop = pop_sel[sort_idx[0:n]]
        self.best_ind.append(self.pop[0])
        return self.pop


"""
# 用法示例：
pop_0 = lhs_sample(100, 100, -5.12, 5.12)
lb_ = -5.12
ub_ = 5.12
ga = GA(pop_0, lb_, ub_)
for item in range(100):
    # pop_1 = ga.sbx(pop_0)
    # pop_2 = ga.mutation(pop_1)
    # pop_0 = np.vstack((pop_1, pop_2))
    # fit = result_calculation('ellipsoid', pop_0)
    # pop_0 = ga.select(pop_0, fit)
    ga.sbx()
    ga.mutation()
    fit = result_calculation('ellipsoid', ga.pop_all)
    ga.select(fit)

best_record = np.array(ga.best_ind)
print(best_record[-1])
print(result_calculation('ellipsoid', best_record[-1]))
"""