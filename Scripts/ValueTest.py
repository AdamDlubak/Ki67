import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from Class.FuzzyHelper import FuzzyHelper as FuzzyHelper
from Class.RulesExtractor import RulesExtractor as RulesExtractor

class ValueTest(object):

    def __init__(self, settings, s_function_width, is_train_df = True, s_function_center = 0.5, threshold = -1):
        self.path = settings.backup_folder
        self.settings = settings
        self.d_results = [settings.class_2, settings.class_1]
        self.x_range = np.arange(settings.set_min, settings.set_max, settings.fuzzy_sets_precision)
        self.s_function_width = s_function_width
        self.fuzzyHelper = FuzzyHelper(settings)
        self.loadData(settings, is_train_df)

    def highlightClassOne(self, row):
        return ['background-color: green' if self.settings.class_1 == x else "" for x in row]

    def loadData(self, settings, is_train_df):
        self.decision = pickle.load(open(self.path + "decision.p", "rb"))
        reductor = pickle.load(open(self.path + "reductor.p", "rb"))
        features = pickle.load(open(self.path + "features.p", "rb"))
        decision_table_with_reduct = pickle.load(open(self.path + "decision_table_with_reduct.p", "rb"))
        self.rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, settings)
        self.rule_antecedents = self.rules_extractor.worker(decision_table_with_reduct, features, self.d_results, self.decision)
        if is_train_df:
            self.df = pickle.load(open(self.path + "train_features_df.p", "rb"))
            self.data_type = "Train"
        else:
            self.df = pickle.load(open(self.path + "test_features_df.p", "rb"))
            self.data_type = "Test"

    def sFunction(self, settings, s_function_center, title, show_results = True):
        print("-----------------------------------------------------------------------------------")
        start = time.time()
        _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, self.s_function_width, self.df, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        end = time.time()

        measured_time = end - start
        accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)
        if show_results:
        
            print("-----------------------------------------------------------------------------------")
            print("Center Point: {}".format(s_function_center))
            print("Time: {}".format(measured_time))
            print("-----------------------------------------------------------------------------------")

            # df = df.sort_values(by=["Predicted Value"])
            # self.decision.view()
            # display(df.style.apply(self.highlightClassOne, axis = 1))
        pickle.dump(df, open(settings.backup_folder + self.data_type + "_df_results.p", "wb"))


        self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, self.data_type, title, accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, self.s_function_width, "---", measured_time])

        return fscore

    def noOptymalizationWorker(self, settings, show_results = True):
            return self.sFunction(settings, 0.5, "No Optymalization", show_results)
    
    def sOptymalizationWorker(self, settings, center_point, show_results = True):
            self.sFunction(settings, center_point, "Value S-Functions", show_results)

    def thresholdWorker(self, settings, s_function_center, threshold, precision = 0.001, show_results = True):   
        print("-----------------------------------------------------------------------------------")
        start = time.time()
        _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, self.s_function_width, self.df, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
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
       
        df = df.sort_values(by=["Predicted Value"])
        if show_results:
            display(df.style.apply(self.highlightClassOne, axis = 1))
            self.decision.view()
        pickle.dump(df, open(settings.backup_folder + self.data_type + "_df_results.p", "wb"))

        self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, self.data_type, "Value Threshold", accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, self.s_function_width, threshold, measured_time])

    def printAllDataframe(self):
        self.fuzzyHelper.printAllDataframe(self.df.sort_values(by=['Predicted Value']))

    def plotHistogram(self, bins = 100):
        self.fuzzyHelper.plotHistogram(self.df, bins)