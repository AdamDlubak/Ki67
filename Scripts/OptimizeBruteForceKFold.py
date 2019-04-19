import time
import pickle
from scipy import optimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from Scripts.OptimizeFunctions import OptimizeFunctions as OptimizeFunctions

class OptimizeBruteForceKFold(object):

    def __init__(self, variables, constraints = ((0, 1, 0.01),), sigma_mean_params = -1):
        self.path = variables['backup_folder']
        self.constraints = constraints
        self.sigma_mean_params = sigma_mean_params
        self.optimizeFunctions = OptimizeFunctions()

    def worker(self, variables):
        sorted_decision = pickle.load(open(self.path + "sorted_decision.p", "rb"))

        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=23)

        X = sorted_decision.drop('Decision', axis=1)
        y = sorted_decision.Decision

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            start = time.time()
            train_data = sorted_decision.iloc[train_index]
            test_data = sorted_decision.iloc[test_index]
            resbrute = optimize.brute(self.optimizeFunctions.makeJob, self.constraints, full_output=True, finish=optimize.fmin, args=(train_data, variables)) 
            self.result_table = self.optimizeFunctions.setDecisions([resbrute[0]], test_data, variables)
            test_accuracy = self.optimizeFunctions.getAccuracy(self.result_table)
            end = time.time()

            train_accuracy = 1 - resbrute[1]
            global_min  = resbrute[0]
            measured_time = end - start

            print("------------------------------------- Fold " + str(idx) + " --------------------------------------")
            print("Global minimum: {}".format(global_min))
            print("Train Accuracy: {}".format(train_accuracy))
            print("Test Accuracy: {}".format(test_accuracy))
            print("Time: {}".format(measured_time))
            print("-----------------------------------------------------------------------------------")

            self.optimizeFunctions.saveResults(variables['results_folder'], ["BruteForce K-Fold {}".format(idx), test_accuracy, "---", "---", "---", "---", "---", "---", "---", "---", global_min, measured_time])
        
