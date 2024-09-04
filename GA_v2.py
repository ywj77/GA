import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class GA(object):
    def __init__(self, gap):
        self.pop = None
        self.max_iter = gap['max_iter']
        self.pc = 0.8
        self.pm = 0.005
        self.pop_n = gap['pop_n']
        self.pop_d = gap['pop_d']
        self.len = gap['len']
        self.lb = gap['lb']
        self.ub = gap['ub']
        self.mm = gap['mm']
        self.fan = gap['gf']
        self.sf = gap['sf']
        self.best_i = []
        self.best_if = []

    def init(self):
        n = self.pop_n
        d = self.pop_d
        l = self.len
        pop = np.ones((n, d, l))
        for n_ in range(n):
            for d_ in range(d):
                pop[n_, d_] = np.random.randint(0, 2, l, dtype=int)
        self.pop = pop
        return self.pop

    @staticmethod
    def b2d(pop_b, lb, ub):
        if np.ndim(pop_b) == 2:
            pop_b = pop_b[np.newaxis, :]
        elif np.ndim(pop_b) == 1:
            pop_b = pop_b[np.newaxis, :]
            pop_b = pop_b[np.newaxis, :]

        n = pop_b.shape[0]
        d = pop_b.shape[1]
        l = pop_b.shape[2]
        temp_1 = np.zeros((n, d))
        for n_ in range(n):
            for d_ in range(d):
                for l_ in range(l):
                    temp_1[n_, d_] += pop_b[n_, d_, l_]*np.power(2, l_)
        temp_2 = (ub - lb)/(np.power(2, l) - 1)
        temp_1 = lb + temp_1*temp_2
        return temp_1

    def roulette(self, pop, fitness):
        sort_idx = None
        sort_pop = None
        sort_fit = None
        n = self.pop_n
        if self.mm == 'max':
            sort_idx = np.argsort(fitness)
            sort_pop = pop[sort_idx]
            sort_fit = fitness[sort_idx]
        elif self.mm == 'min':
            sort_idx = np.argsort(-fitness)
            sort_pop = pop[sort_idx]
            sort_fit = 1/fitness[sort_idx]
        fit_sum = np.sum(sort_fit)
        rou = np.zeros(pop.shape[0])
        rou[0] = sort_fit[0]/fit_sum
        for i in range(1, pop.shape[0]):
            rou[i] = rou[i-1] + sort_fit[i]/fit_sum
        rou[-1] = 1
        new_idx = []
        for i in range(n):
            rand = np.random.uniform()
            for j in range(pop.shape[0]):
                if rou[j] >= rand:
                    new_idx.append(j)
                    break

        new_pop = sort_pop[new_idx]
        new_fit = fitness[sort_idx][new_idx]
        new_idx = None
        if self.mm == 'max':
            new_idx = np.argsort(new_fit)
        elif self.mm == 'min':
            new_idx = np.argsort(-new_fit)
        new_pop = new_pop[new_idx]
        new_fit = new_fit[new_idx]
        self.best_i.append(self.b2d(new_pop[-1], self.lb, self.ub))
        self.best_if.append(new_fit[-1])
        return new_pop, new_fit

    def best(self, pop, fitness):
        new_pop = None
        new_fit = None
        if self.mm == 'min':
            idx = np.argsort(-fitness)
            new_pop = pop[idx][:(self.pop_n+1):-1]
            new_fit = fitness[idx][:(self.pop_n+1):-1]
        if self.mm == 'max':
            idx = np.argsort(fitness)
            new_pop = pop[idx][:(self.pop_n+1):-1]
            new_fit = fitness[idx][:(self.pop_n+1):-1]
        self.best_i.append(self.b2d(new_pop[-1], self.lb, self.ub))
        self.best_if.append(new_fit[-1])
        return new_pop, new_fit

    def select(self, pop, fitness):
        if self.sf == 'roulette':
            return self.roulette(pop, fitness)
        elif self.sf == 'best':
            return self.best(pop, fitness)

    def crossover(self, pop):
        n = pop.shape[0]
        d = pop.shape[1]
        new_pop = []
        for i in range(n):
            idx_1 = np.arange(n)
            rand = np.random.uniform()
            if rand <= self.pc:
                idx_1 = np.delete(idx_1, i)
                k = np.random.choice(idx_1)
                idx_2 = np.arange(1, self.len)
                ran_p = np.random.choice(idx_2, 2)
                cp1, cp2 = ran_p[0], ran_p[1]
                cp1, cp2 = np.minimum(cp1, cp2),  np.maximum(cp1, cp2)
                new_pop1, new_pop2 = [], []
                for j in range(d):
                    temp1, temp2 = [], []
                    temp1.extend(pop[i, j][0:cp1])
                    temp1.extend(pop[k, j][cp1:cp2])
                    temp1.extend(pop[i, j][cp2:])
                    temp2.extend(pop[k, j][0:cp1])
                    temp2.extend(pop[i, j][cp1:cp2])
                    temp2.extend(pop[k, j][cp2:])
                    new_pop1.append(temp1)
                    new_pop2.append(temp2)
                new_pop.append(new_pop1)
                new_pop.append(new_pop2)
        new_pop = np.array(new_pop)
        return new_pop

    def mutation(self, pop):
        n = pop.shape[0]
        d = pop.shape[1]
        new_pop = pop.copy()
        for i in range(n):
            for j in range(d):
                rand = np.random.uniform()
                if rand <= self.pm:
                    cp = np.random.randint(0, self.len)
                    new_pop[i, j, cp] = 1 - pop[i, j, cp]
        return new_pop

    def run(self):
        pop = self.init()
        for g in tqdm(range(self.max_iter),  desc="GA_Processing"):
            pop1 = self.crossover(pop)
            pop2 = self.mutation(pop1)
            pop2 = np.unique(pop2, axis=0)
            pop3 = self.b2d(pop2, self.lb, self.ub)
            fit = self.fan(pop3)
            pop, fit = self.select(pop2, fit)
        return self.best_i, self.best_if


def get_fitness(pop):
    res = np.sum(pop**2, axis=1)
    return res


if __name__ == "__main__":
    dim = 10
    ga_init = {
        'max_iter': 100,
        'pop_n': 100,
        'pop_d': dim,
        'len': 10,
        'lb': -5.12 * np.ones(dim),
        'ub': 5.12 * np.ones(dim),
        'mm': ['min', 'max'][0],
        'gf': get_fitness,
        'sf': ['roulette', 'best'][0],
    }
    ga = GA(ga_init)
    ga.run()
    x = np.arange(ga_init['max_iter'])
    y = ga.best_if
    plt.figure()
    plt.plot(x, y)
    plt.show()
