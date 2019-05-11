import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import optimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from Class.FuzzyHelper import FuzzyHelper as FuzzyHelper
from Class.RulesExtractor import RulesExtractor as RulesExtractor

class OptimizeBruteForce(object):

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


    def sFunctionsWorker(self, variables, constraints, s_function_width):
        start = time.time()
        params = (s_function_width, self.df, variables, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        optimization_result = optimize.brute(self.fuzzyHelper.sFunctionsOptBrute, constraints, args=params, full_output=True, finish=optimize.fmin)
        end = time.time()

        # Used to save to pickle file
        self.fuzzyHelper.prepareRules(True, self.x_range, optimization_result[0], s_function_width, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision, self.df)

        accuracy = 1 - optimization_result[1]
        measured_time = end - start
        s_function_center = optimization_result[0][0]
        _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, self.df, variables, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)
  
        print("-----------------------------------------------------------------------------------")
        print("Center Point: {}".format(s_function_center))
        print("Train Accuracy: {}".format(accuracy))
        print("Time: {}".format(measured_time))
        print("-----------------------------------------------------------------------------------")

        display(df.sort_values(by=["Predicted Value"]))
        self.fuzzyHelper.saveResults(variables['results_folder'] + variables["results_file"], [variables["test_type"], variables["dataset_name"], variables["gausses"], "Train", "BruteForce S-Functions", accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, s_function_width, "---", measured_time])

        return s_function_center

    def thresholdWorker(self, variables, s_function_center, s_function_width, precision = 0.001):
        start = time.time()
        accuracy, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, self.df, variables, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        threshold = (slice(df["Predicted Value"].min(), df["Predicted Value"].max(), precision), )
        params = (df, start)
        optimization_result = optimize.brute(self.fuzzyHelper.thresholdOptBrute, threshold, args=params, full_output=True, finish=optimize.fmin)
        end = time.time()

        accuracy = 1 - optimization_result[1]
        threshold = optimization_result[0][0]
        measured_time = end - start

        df = df.apply(self.fuzzyHelper.setDecisions, threshold = threshold, axis=1)
        accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)

        print("-----------------------------------------------------------------------------------")
        print("Center Point: {}".format(s_function_center))
        print("Threshold: {}".format(threshold))
        print("Train Accuracy: {}".format(accuracy))
        print("Time: {}".format(measured_time))
        print("-----------------------------------------------------------------------------------")

        display(df.sort_values(by=["Predicted Value"]))
        self.fuzzyHelper.saveResults(variables['results_folder'] + variables["results_file"], [variables["test_type"], variables["dataset_name"], variables["gausses"], "Train", "BruteForce Threshold", accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, s_function_width, threshold, measured_time])

        return threshold