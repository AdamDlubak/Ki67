import time
import pickle
import numpy as np
import pyswarms as ps
from scipy import optimize
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from Scripts.OptimizeFunctions import OptimizeFunctions as OptimizeFunctions
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface

class OptimizePSO(object):

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

        optimizer = ps.single.GlobalBestPSO(n_particles = self.swarm_size, dimensions = self.dim, options = self.options, bounds = self.constraints)

        start = time.time()
        cost, global_min = optimizer.optimize(self.optimizeFunctions.optFunc, iters = self.iters, sorted_decision = sorted_decision, variables = variables)
        end = time.time()

        accuracy = 1 - cost
        measured_time = end - start

        print("Global minimum: {}".format(global_min))
        print("Function value at global minimum: {}".format(cost))
        print("Accuracy: {}".format(accuracy))
        print("Time: {}".format(measured_time))

        self.optimizeFunctions.saveResults(variables['results_folder'], ["PSO", accuracy, "---", "---", "---", "---", "---", "---", "---", "---", global_min[0], measured_time])

        plot_cost_history(optimizer.cost_history)
        plt.show()

        return global_min