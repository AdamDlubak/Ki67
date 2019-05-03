import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from Class.FuzzyHelper import FuzzyHelper as FuzzyHelper
from Class.RulesExtractor import RulesExtractor as RulesExtractor

class ValueTest(object):

    def __init__(self, variables, s_function_width, is_train_df = True, s_function_center = 0.5, threshold = -1):
        self.path = variables['backup_folder']
        self.d_results = [variables["class_1"], variables["class_2"]]
        self.x_range = np.arange(variables["set_min"], variables["set_max"], variables["fuzzy_sets_precision"])
        self.s_function_width = s_function_width
        self.fuzzyHelper = FuzzyHelper(variables)
        self.loadData(variables, is_train_df)

    def loadData(self, variables, is_train_df):
        self.decision = pickle.load(open(self.path + "decision.p", "rb"))
        reductor = pickle.load(open(self.path + "reductor.p", "rb"))
        features = pickle.load(open(self.path + "features.p", "rb"))
        decision_table_with_reduct = pickle.load(open(self.path + "decision_table_with_reduct.p", "rb"))
        self.rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, variables)
        self.rule_antecedents = self.rules_extractor.worker(decision_table_with_reduct, features, self.d_results, self.decision)
        if is_train_df:
            self.df = pickle.load(open(self.path + "train_features_df.p", "rb"))
            self.data_type = "Train"
        else:
            self.df = pickle.load(open(self.path + "test_features_df.p", "rb"))
            self.data_type = "Test"

    def sFunction(self, variables, s_function_center, title):
        start = time.time()
        _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, self.s_function_width, self.df, variables, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        end = time.time()

        measured_time = end - start
        accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)
        print("-----------------------------------------------------------------------------------")
        print("Center Point: {}".format(s_function_center))
        print(self.data_type + " Accuracy: {}".format(accuracy))
        print("Time: {}".format(measured_time))
        print("-----------------------------------------------------------------------------------")

        self.fuzzyHelper.saveResults(variables['results_folder'] + variables["results_file"], [self.data_type + ": " + title, accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, self.s_function_width, "---", measured_time])

    def noOptymalizationWorker(self, variables):
            self.sFunction(variables, 0.5, "No Optymalization")
    
    def sOptymalizationWorker(self, variables, center_point, description = "S Optymalization"):
            self.sFunction(variables, center_point, description)

    def thresholdWorker(self, variables, s_function_center, threshold, precision = 0.001):   
        start = time.time()
        _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, self.s_function_width, self.df, variables, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        _, df = self.fuzzyHelper.thresholdOptValue(threshold, df)
        end = time.time()
        measured_time = end - start
        accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)

        print("-----------------------------------------------------------------------------------")
        print("Center Point: {}".format(s_function_center))
        print("Threshold: {}".format(threshold))
        print(self.data_type + " Accuracy: {}".format(accuracy))
        print("Time: {}".format(measured_time))
        print("-----------------------------------------------------------------------------------")

        self.fuzzyHelper.saveResults(variables['results_folder'] + variables["results_file"], [self.data_type + ": " + "Threshold Optymalization", accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, self.s_function_width, threshold, measured_time])

    def printAllDataframe(self):
        self.fuzzyHelper.printAllDataframe(self.df.sort_values(by=['Predicted Value']))

    def plotHistogram(self, bins = 100):
        self.fuzzyHelper.plotHistogram(self.df, bins)