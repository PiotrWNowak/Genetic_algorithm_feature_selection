#!/usr/bin/python3
import numpy as np
import random

class Genetic:
    def __init__(self, features, parents, children, mutation_scale, regulation_type = None, constant_factor=10**-4, percent_factor=1, epsilon=10**-8):
        self.features = features
        self.parents = parents
        self.children = children
        self.mutation_scale = mutation_scale
        self.regulation_type = regulation_type
        self.constant_factor = constant_factor
        self.percent_factor = percent_factor
        self.epsilon = epsilon

        self.tab = np.random.random_integers( 0, 1, ( parents+children, features ) )
        self.results = np.zeros((parents+children))
        self.accuracy = np.zeros((parents+children))
        self.best = np.array([[0,0]])
        self.mean_ = np.array([[0,0]])

    def crossover(self):
        for i in range(self.parents, len(self.tab)):
            j = (i - self.parents) % self.parents
            change = np.random.rand(self.features) < 0.5
            self.tab[i][change] = self.tab[j][change]

    def mutation(self):
        for i in range(self.parents, len(self.tab)):
            for j in range(len(self.tab[i])):
                if random.random() < self.mutation_scale:
                    self.tab[i][j] = 0

    def fit(self, x, y, generations, modelCV):
        parents = 0
        for j in range(generations):
            for i in range(parents, len(self.tab)):
                if np.count_nonzero(self.tab[i]) == 0:
                    self.tab[i][random.randint(0,self.features-1)] = 1
                self.results[i], self.accuracy[i] = modelCV(x, y, self.tab[i])
            temp_sort = self.regulation()
            self.tab = self.tab[temp_sort]
            self.accuracy = self.accuracy[temp_sort]
            self.results = self.results[temp_sort]
            self.best = np.append(self.best, [[self.results[0],self.accuracy[0]]], axis=0)
            self.mean_ = np.append(self.mean_, [[np.mean(self.results),np.mean(self.accuracy)]], axis=0)
            parents = self.parents
            self.crossover()
            self.mutation()
        self.best = np.delete(self.best, 0, 0)
        self.mean_ = np.delete(self.mean_, 0, 0)

    def regulation(self):
        return {
            None: self.results.argsort(),
            'constant': (self.results + self.constant_factor * np.count_nonzero(self.tab, axis=1)).argsort(),
            'percent': (self.results + 0.01 * self.percent_factor * (self.results + self.epsilon) * np.count_nonzero(self.tab, axis=1)).argsort()
        }.get(self.regulation_type)
