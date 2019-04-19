import time
import pickle
import numpy as np
import pandas as pd
import pyswarms as ps
from scipy import optimize
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from Scripts.OptimizeFunctions import OptimizeFunctions as OptimizeFunctions
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface

class OptimizePSOKFold(object):

    def __init__(self, variables, swarm_size, dim, epsilon, iters, options, constraints = (np.array([0]), np.array([1])), sigma_mean_params = -1):
        self.path = variables['backup_folder']
        self.swarm_size = swarm_size
        self.dim = dim
        self.epsilon = epsilon
        self.iters = iters
        self.options = options
        self.constraints = constraints
        self.sigma_mean_params = sigma_mean_params
        self.optimizeFunctions = OptimizeFunctions()

    def worker(self, variables):
        sorted_decision = pickle.load(open(self.path + "sorted_decision.p", "rb"))

        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=23)

        X = sorted_decision.drop('Decision', axis=1)
        y = sorted_decision.Decision

        optimizer = ps.single.GlobalBestPSO(n_particles = self.swarm_size, dimensions = self.dim, options = self.options, bounds = self.constraints)

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            start = time.time()
            train_data = sorted_decision.iloc[train_index]
            test_data = sorted_decision.iloc[test_index]
            cost, global_min = optimizer.optimize(self.optimizeFunctions.optFunc, iters=self.iters, sorted_decision = sorted_decision, variables = variables)
            self.result_table = self.optimizeFunctions.setDecisions([global_min], test_data, variables)
            test_accuracy = self.optimizeFunctions.getAccuracy(self.result_table)
            end = time.time()

            train_accuracy = 1 - cost
            measured_time = end - start

            print("------------------------------------- Fold " + str(idx) + " --------------------------------------")
            print("Global minimum: {}".format(global_min))
            print("Train Accuracy: {}".format(train_accuracy))
            print("Test Accuracy: {}".format(test_accuracy))
            print("Time: {}".format(measured_time))
            print("-----------------------------------------------------------------------------------")

            self.optimizeFunctions.saveResults(variables['results_folder'], ["PSO K-Fold {}".format(idx), test_accuracy, "---", "---", "---", "---", "---", "---", "---", "---", global_min[0], measured_time])
        
