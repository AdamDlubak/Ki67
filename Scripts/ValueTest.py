import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from Scripts.OptimizeFunctions import OptimizeFunctions as OptimizeFunctions

class ValueTest(object):

    def __init__(self, variables, threshold_value = -1):
        self.path = variables['backup_folder']
        self.threshold_value = threshold_value
        self.optimizeFunctions = OptimizeFunctions()
        self.accuracy = -1
        self.precision = -1
        self.recall = -1
        self.fscore = -1
        self.support = -1
        self.result_table = []

    def worker(self, variables):
        start = time.time()
        test_normalized_features_table = pickle.load(open(self.path + "test_normalized_features_table.p", "rb"))
        rules = pickle.load(open(self.path + "rules.p", "rb"))

        test_sorted_decision = self.optimizeFunctions.setRules(rules, test_normalized_features_table)
        self.result_table = self.optimizeFunctions.setDecisions([self.threshold_value], test_sorted_decision, variables)
        self.accuracy, self.precision, self.recall, self.fscore, self.support = self.optimizeFunctions.getScores(self.result_table)
        end = time.time()
        measured_time = end - start

        if self.threshold_value == -1:
            print_threshold = self.threshold_value
        else:
            print_threshold = self.threshold_value[0]

        self.optimizeFunctions.saveResults(variables['results_folder'], ["ValueTest", self.accuracy, self.precision[0], self.precision[1], self.recall[0], self.recall[1], self.fscore[0], self.fscore[1], self.support[0], self.support[1], print_threshold, measured_time])
        
    def printAllDataframe(self):
        self.optimizeFunctions.printAllDataframe(self.result_table)

    def plotHistogram(self, bins = 100):
        self.optimizeFunctions.plotHistogram(self.result_table, bins)