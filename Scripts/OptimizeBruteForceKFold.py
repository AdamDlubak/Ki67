import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy import optimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from Functions.RulesExtractor import RulesExtractor as RulesExtractor
from Scripts.OptimizeFunctions import OptimizeFunctions as OptimizeFunctions

class OptimizeBruteForceKFold(object):

    def __init__(self, variables, constraints, sigma_mean_params = -1):
        self.path = variables['backup_folder']
        self.constraints = constraints
        self.sigma_mean_params = sigma_mean_params
        self.optimizeFunctions = OptimizeFunctions()
        self.d_results = [variables["class_1"], variables["class_2"]]
        self.x_range = np.arange(variables["set_min"], variables["set_max"], variables["fuzzy_sets_precision"])


    def worker(self, variables, width):
        decision = pickle.load(open(self.path + "decision.p", "rb"))
        decision_table_with_reduct = pickle.load(open(self.path + "decision_table_with_reduct.p", "rb"))
        reductor = pickle.load(open(self.path + "reductor.p", "rb"))
        features = pickle.load(open(self.path + "features.p", "rb"))
        normalized_features_table = pickle.load(open(self.path + "normalized_features_table.p", "rb"))

        rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, variables)
        rule_antecedents, feature_names = rules_extractor.worker(decision_table_with_reduct, features, self.d_results, decision)

        skf = StratifiedKFold(n_splits=3, shuffle = True, random_state=23)

        X = normalized_features_table.drop('Decision', axis=1)
        y = normalized_features_table.Decision

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            start = time.time()
            train_data = normalized_features_table.iloc[train_index]
            test_data = normalized_features_table.iloc[test_index]
            accuracy_table = []
            global_min_table = []
            for center_point in tqdm(np.arange(0, 1.01, 0.20)):
                accuracy, test_features_table = self.optimizeFunctions.makeJob(center_point, width, train_data, variables, self.x_range, rules_extractor, rule_antecedents, self.d_results, decision)
                accuracy_table.append(accuracy)
                global_min_table.append(center_point)
                display(test_features_table.sort_values(by=["Predicted Value"]))

            min_index = np.argmax(accuracy_table)

            test_accuracy, self.result_table = self.optimizeFunctions.makeJob(global_min_table[min_index], width, test_data, variables, self.x_range, rules_extractor, rule_antecedents, self.d_results, decision)
            end = time.time()

            train_accuracy = accuracy_table[min_index]
            global_min  = global_min_table[min_index]
            measured_time = end - start

            print(accuracy_table)
            print(global_min_table)
            print("------------------------------------- Fold " + str(idx) + " --------------------------------------")
            print("Global minimum: {}".format(global_min))
            print("Train Accuracy: {}".format(train_accuracy))
            print("Test Accuracy: {}".format(test_accuracy))
            print("Time: {}".format(measured_time))
            print("-----------------------------------------------------------------------------------")

            self.optimizeFunctions.saveResults(variables['results_folder'], ["BruteForce K-Fold {}".format(idx), test_accuracy, "---", "---", "---", "---", "---", "---", "---", "---", global_min, measured_time])
        
