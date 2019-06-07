import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from Class.FuzzyHelper import FuzzyHelper as FuzzyHelper
from Class.RulesExtractor import RulesExtractor as RulesExtractor

class Ki67(object):

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
        start = time.time()
        df = self.fuzzyHelper.sFunctionsValueKi67(s_function_center, self.s_function_width, self.df, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        end = time.time()
        measured_time = end - start
        print("-----------------------------------------------------------------------------------")
        print("Time: {}".format(measured_time))
        print("-----------------------------------------------------------------------------------")

        if not os.path.exists(settings.backup_folder + "Images/"):
            os.makedirs(settings.backup_folder + "Images/")   

        pickle.dump(df, open(settings.backup_folder + "Images/" + self.data_type + "_" + settings.file_name + "_" + settings.class_1 + "_df_results.p", "wb"))
        self.fuzzyHelper.saveResultsKi67(settings.results_folder + settings.results_file, [settings.test_type, "Ki67", settings.file_name, settings.style, settings.gausses, settings.adjustment, settings.class_1, self.data_type, title, s_function_center, self.s_function_width, "---", measured_time])

    def noOptymalizationWorker(self, settings, show_results = True):
            return self.sFunction(settings, 0.5, "Choose Features group", show_results)
    
    def sOptymalizationWorker(self, settings, center_point, show_results = True):
            self.sFunction(settings, center_point, "Value S-Functions", show_results)

    def printAllDataframe(self):
        self.fuzzyHelper.printAllDataframe(self.df.sort_values(by=['Predicted Value']))

    def plotHistogram(self, bins = 100):
        self.fuzzyHelper.plotHistogram(self.df, bins)