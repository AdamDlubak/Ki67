import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import optimize
from Class.Helper import Helper as Helper
from sklearn.metrics import accuracy_score
from Scripts.Fuzzify import Fuzzify as Fuzzify
from Scripts.LoadCSV import LoadCSV as LoadCSV
from sklearn.model_selection import StratifiedKFold
from Class.FuzzyHelper import FuzzyHelper as FuzzyHelper
from Class.RulesExtractor import RulesExtractor as RulesExtractor

class OptimizeBruteForce(object):

    def __init__(self, settings, s_function_width):
        self.path = settings.backup_folder
        self.d_results = [settings.class_2, settings.class_1]
        self.x_range = np.arange(settings.set_min, settings.set_max, settings.fuzzy_sets_precision)
        self.s_function_width = s_function_width
        self.fuzzyHelper = FuzzyHelper(settings)
        self.loadData(settings)
        self.settings = settings

    def highlightClassOne(self, row):
        return ['background-color: green' if self.settings.class_1 == x else "" for x in row]

    def loadData(self, settings):
        self.decision = pickle.load(open(self.path + "decision.p", "rb"))
        reductor = pickle.load(open(self.path + "reductor.p", "rb"))
        features = pickle.load(open(self.path + "features.p", "rb"))
        decision_table_with_reduct = pickle.load(open(self.path + "decision_table_with_reduct.p", "rb"))
        self.rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, settings)
        self.rule_antecedents = self.rules_extractor.worker(decision_table_with_reduct, features, self.d_results, self.decision)
        self.df = pickle.load(open(self.path + "train_features_df.p", "rb"))

    def adjustmentsWorker(self, settings, constraints, s_function_width, show_results = True):
        helper = Helper()
        loadCSV = LoadCSV()
        samples_stats, train_stats, test_stats, train_samples = loadCSV.worker(settings)
        fuzzify = Fuzzify()

        params = (settings, fuzzify)
        optimization_result = optimize.brute(self.fuzzyHelper.adjustmentsOptBrute, constraints, args=params, full_output=True, finish=optimize.fmin)
        optymized_mean = optimization_result[0][0]
        changed_decisions, features_number_after_reduct, implicants_number, _ = fuzzify.worker(settings, settings.optymized_mean)

        fuzzification_data = [settings.dataset_name, settings.style, settings.gausses, settings.adjustment, samples_stats, train_stats, test_stats, changed_decisions, round(changed_decisions / train_samples, 2), implicants_number, settings.feature_numbers, features_number_after_reduct]
        helper.saveFuzzificationStats(fuzzification_data)
        print("-----------------------------------------------------------------------------------")
        print("Optymized Mean: {}".format(optymized_mean))
        print("-----------------------------------------------------------------------------------")
        valueTest = ValueTest(settings, settings.s_function_width, settings.is_training)
        valueTest.sOptymalizationWorker(settings, settings.default_s_value, settings.show_results)

        valueTest = ValueTest(settings, settings.s_function_width, not settings.is_training)
        valueTest.sOptymalizationWorker(settings, settings.default_s_value, settings.show_results)

    def sFunctionsWorker(self, settings, constraints, s_function_width, show_results = True):
        start = time.time()
        params = (s_function_width, self.df, self.df, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        optimization_result = optimize.brute(self.fuzzyHelper.sFunctionsOptBrute, constraints, args=params, full_output=True, finish=optimize.fmin)
        end = time.time()

        # Used to save to pickle file
        self.fuzzyHelper.prepareRules(True, self.x_range, optimization_result[0], s_function_width, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision, self.df)

        accuracy = 1 - optimization_result[1]
        measured_time = end - start
        s_function_center = optimization_result[0][0]
        _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, self.df, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)
  
        print("-----------------------------------------------------------------------------------")
        print("Center Point: {}".format(s_function_center))
        print("Time: {}".format(measured_time))
        print("-----------------------------------------------------------------------------------")

        df = df.sort_values(by=["Predicted Value"])
        if show_results:
            self.decision.view()
            display(df.style.apply(self.highlightClassOne, axis = 1))
            
        self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, "Train", "BruteForce S-Functions", accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, s_function_width, "---", measured_time])

        return s_function_center

    def thresholdWorker(self, settings, s_function_center, s_function_width, precision = 0.001, show_results = True):
        start = time.time()
        accuracy, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, self.df, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
        threshold = (slice(df["Predicted Value"].min(), df["Predicted Value"].max(), precision), )
        params = (df, start)
        optimization_result = optimize.brute(self.fuzzyHelper.thresholdOptBrute, threshold, args=params, full_output=True, finish=optimize.fmin)
        end = time.time()
        accuracy = 1 - optimization_result[1]
        threshold = optimization_result[0][0]
        measured_time = end - start
        
        df = df.apply(self.fuzzyHelper.setDecisions, threshold = threshold, axis=1)
        self.df = df
        accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)

        print("-----------------------------------------------------------------------------------")
        print("Center Point: {}".format(s_function_center))
        print("Threshold: {}".format(threshold))
        print("Train Accuracy: {}".format(accuracy))
        print("Time: {}".format(measured_time))
        print("-----------------------------------------------------------------------------------")

        df = df.sort_values(by=["Predicted Value"])

        if show_results:
            self.decision.view()
            display(df.style.apply(self.highlightClassOne, axis = 1))

        self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, "Train", "BruteForce Threshold", accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, s_function_width, threshold, measured_time])

        return threshold