import time
import pickle
from scipy import optimize
from Scripts.OptimizeFunctions import OptimizeFunctions as OptimizeFunctions
from Scripts.ValueTrain import OptimizeFunctions as OptimizeFunctions

class OptimizeBruteForce(object):

    def __init__(self, variables, constraints = ((0, 1, 0.01),), sigma_mean_params = -1):
        self.path = variables['backup_folder']
        self.constraints = constraints
        self.sigma_mean_params = sigma_mean_params
        self.optimizeFunctions = OptimizeFunctions()

    def worker(self, variables):
        sorted_decision = pickle.load(open(self.path + "sorted_decision.p", "rb"))
        
        
        start = time.time()
        results = optimize.brute(self.optimizeFunctions.makeJob, self.constraints, full_output=True, finish=optimize.fmin, args=(sorted_decision, variables))
        end = time.time()

        accuracy = 1 - results[1]
        global_min  = results[0]
        measured_time = end - start

        print("Global minimum: {}".format(global_min))
        print("Function value at global minimum: {}".format(results[1]))
        print("Accuracy: {}".format(accuracy))
        print("Time: {}".format(measured_time))

        self.optimizeFunctions.saveResults(variables['results_folder'], ["BruteForce", accuracy, "---", "---", "---", "---", "---", "---", "---", "---", global_min[0], measured_time])
        
        return global_min
