import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import optimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from Class.FuzzyHelper import FuzzyHelper as FuzzyHelper
from Class.RulesExtractor import RulesExtractor as RulesExtractor

class OptimizeBruteForceKFold(object):

    def __init__(self, variables, s_function_width):
        self.path = variables['backup_folder']
        self.d_results = [variables["class_1"], variables["class_2"]]
        self.x_range = np.arange(variables["set_min"], variables["set_max"], variables["fuzzy_sets_precision"])
        self.s_function_width = s_function_width
        self.fuzzyHelper = FuzzyHelper(variables)
        self.loadData(variables)

    def loadData(self, variables):
        self.decision = pickle.load(open(self.path + "decision.p", "rb"))
        reductor = pickle.load(open(self.path + "reductor.p", "rb"))
        features = pickle.load(open(self.path + "features.p", "rb"))
        decision_table_with_reduct = pickle.load(open(self.path + "decision_table_with_reduct.p", "rb"))
        self.rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, variables)
        self.rule_antecedents = self.rules_extractor.worker(decision_table_with_reduct, features, self.d_results, self.decision)
        self.df = pickle.load(open(self.path + "train_features_df.p", "rb"))

    def worker(self, variables, constraints, s_function_width, n_folds = 10):

        skf = StratifiedKFold(n_splits= n_folds, shuffle = True, random_state=23)
        X = self.df.drop('Decision', axis=1)
        y = self.df.Decision
        s_function_centers = []
        best_accuracy_score = 0
        best_s_function_center = 0
        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            
            start = time.time()
            train_data = self.df.iloc[train_index]
            test_data = self.df.iloc[test_index]
            
            params = (s_function_width, train_data, variables, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            optimization_result = optimize.brute(self.fuzzyHelper.sFunctionsOptBrute, constraints, args=params, full_output=True, finish=optimize.fmin)
            end = time.time()

            self.fuzzyHelper.prepareRules(True, self.x_range, optimization_result[0], s_function_width, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision, self.df)

            accuracy = 1 - optimization_result[1]
            measured_time = end - start
            s_function_center = optimization_result[0][0]
            _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, test_data, variables, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            test_accuracy, test_precision, test_recall, test_fscore, test_support = self.fuzzyHelper.getScores(df, False)
            if idx == 0:
                display(df.sort_values(by=["Predicted Value"]).head(10))
                display(df.sort_values(by=["Predicted Value"]).tail(10))
            s_function_centers.append(s_function_center)
            if test_accuracy > best_accuracy_score:
                best_s_function_center = s_function_center
                best_accuracy_score = test_accuracy
            
            print("------------------------------------- Fold " + str(idx) + " --------------------------------------")
            print("Center Point: {}".format(s_function_center))
            print("Train Accuracy: {}".format(accuracy))
            print("Test Accuracy: {}".format(test_accuracy))
            print("Test Precision: {}".format(test_precision))
            print("Test Recall: {}".format(test_recall))
            print("Test F-Score: {}".format(test_fscore))
            print("Test Support: {}".format(test_support))
            print("Time: {}".format(measured_time))
            print("-----------------------------------------------------------------------------------")

            self.fuzzyHelper.saveResults(variables['results_folder'] + variables["results_file"], ["Train: BruteForce S-Functions K-Fold {}".format(idx), test_accuracy, test_precision[0], test_precision[1], test_recall[0], test_recall[1], test_fscore[0], test_fscore[1], test_support[0], test_support[1], s_function_center, s_function_width, "---", measured_time])

        mean_s_function_center = sum(s_function_centers) / len(s_function_centers) 
        return best_s_function_center, mean_s_function_center