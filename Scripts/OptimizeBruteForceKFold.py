import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import optimize
from Class.Helper import Helper as Helper
from sklearn.metrics import accuracy_score
from Scripts.LoadCSV import LoadCSV as LoadCSV
from Scripts.Fuzzify import Fuzzify as Fuzzify
from sklearn.model_selection import StratifiedKFold
from Scripts.ValueTest import ValueTest as ValueTest
from Class.FuzzyHelper import FuzzyHelper as FuzzyHelper
from Class.RulesExtractor import RulesExtractor as RulesExtractor

import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface


class OptimizeBruteForceKFold(object):

    def __init__(self, settings, s_function_width):
        self.path = settings.backup_folder
        self.d_results = [settings.class_2, settings.class_1]
        self.x_range = np.arange(settings.set_min, settings.set_max, settings.fuzzy_sets_precision)
        self.s_function_width = s_function_width
        self.fuzzyHelper = FuzzyHelper(settings)
        self.loadData(settings)

    def loadData(self, settings):
        self.decision = pickle.load(open(self.path + "decision.p", "rb"))
        self.df = pickle.load(open(self.path + "train_features_df.p", "rb"))
        
    def adjustmentsWorker(self, settings, constraints, s_function_width, n_folds = 10):
        skf = StratifiedKFold(n_splits= n_folds, shuffle = True, random_state=23)
        X = self.df.drop('Decision', axis=1)
        y = self.df.Decision
        adjustment_value = []
        best_accuracy_score = 0
        best_adjustment = 0

        
        helper = Helper()
        loadCSV = LoadCSV()
        samples_stats, train_stats, test_stats, train_samples = loadCSV.worker(settings)
        
        fuzzify = Fuzzify()


        swarm_size = 5
        dim = 7
        epsilon = 1.0
        iters = 20
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        x_max = 0.2 * np.ones(7)
        x_min = 0.8 * np.ones(7)
        bounds = (x_min, x_max)
        optimizer = ps.single.GlobalBestPSO(n_particles = 10, dimensions = dim, options = options, bounds = bounds)


        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            
            start = time.time()
            train_data = self.df.iloc[train_index]
            train_data_for_worker = train_data.copy()
            test_data = self.df.iloc[test_index]
            constraints = settings.constraints_adj
            valueTest = ValueTest(settings, settings.s_function_width, not settings.is_training)
            params = (settings, fuzzify, train_data_for_worker, valueTest)




            # optimization_result = optimize.brute(self.fuzzyHelper.adjustmentsOptBrute, constraints, args=params, full_output=True, finish=optimize.fmin)
            cost, global_min = optimizer.optimize(self.fuzzyHelper.adjustmentsOptBrute, iters=iters, settings= settings, fuzzify = fuzzify, train_data_for_worker = train_data_for_worker, valueTest = valueTest)
            

            print(global_min)
            changed_decisions, features_number_after_reduct, implicants_number, features, decision_table_with_reduct, reductor = fuzzify.worker(settings, global_min)

            self.rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, settings)
            self.rule_antecedents = self.rules_extractor.worker(decision_table_with_reduct, features, self.d_results, self.decision)
            end = time.time()

            self.fuzzyHelper.prepareRules(True, self.x_range, global_min, s_function_width, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision, self.df)
            _, df = self.fuzzyHelper.sFunctionsValue(0.5, s_function_width, train_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            display(df.sort_values(by=["Predicted Value"]))

            accuracy = 1 - cost
            measured_time = end - start
            s_function_center = global_min
            _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, test_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            test_accuracy, test_precision, test_recall, test_fscore, test_support = self.fuzzyHelper.getScores(df, False)
            display(df.sort_values(by=["Predicted Value"]))
            adjustment_value.append(s_function_center)
            if test_accuracy > best_accuracy_score:
                best_adjustment = adjustment_value
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

            self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, "Train", "BruteForce S-Functions K-Fold {}".format(idx), test_accuracy, test_precision[0], test_precision[1], test_recall[0], test_recall[1], test_fscore[0], test_fscore[1], test_support[0], test_support[1], s_function_center, s_function_width, "---", measured_time])

        mean_s_function_center = sum(adjustment_value) / len(adjustment_value) 
        return best_adjustment, mean_s_function_center, changed_decisions, features_number_after_reduct, implicants_number

    def thresholdWorker(self, settings, s_function_center, s_function_width, precision_value = 0.001, show_results = True, n_folds = 10):
        
        skf = StratifiedKFold(n_splits= n_folds, shuffle = True, random_state=23)
        X = self.df.drop('Decision', axis=1)
        y = self.df.Decision
        thresholds = []
        best_accuracy_score = 0
        best_threshold = 0
        fuzzify = Fuzzify()

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            
            start = time.time()
            train_data = self.df.iloc[train_index]
            train_data_for_worker = train_data.copy()
            test_data = self.df.iloc[test_index]       

            changed_decisions, features_number_after_reduct, implicants_number, features, decision_table_with_reduct, reductor = fuzzify.workerKFold(settings, train_data_for_worker, settings.adjustment_value)
            self.rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, settings)
            self.rule_antecedents = self.rules_extractor.worker(decision_table_with_reduct, features, self.d_results, self.decision)

            self.fuzzyHelper.prepareRules(True, self.x_range, s_function_center, s_function_width, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision, self.df)

            accuracy, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, train_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            threshold = (slice(df["Predicted Value"].min(), df["Predicted Value"].max(), precision_value), )
            params = (df, start)
            optimization_result = optimize.brute(self.fuzzyHelper.thresholdOptBrute, threshold, args=params, full_output=True, finish=optimize.fmin)
            end = time.time()
            accuracy = 1 - optimization_result[1]
            threshold = optimization_result[0][0]
            measured_time = end - start
            
            df = df.apply(self.fuzzyHelper.setDecisions, threshold = threshold, axis=1)
            accuracy, precision, recall, fscore, support = self.fuzzyHelper.getScores(df)

            accuracy, test_df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, test_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            test_df = test_df.apply(self.fuzzyHelper.setDecisions, threshold = threshold, axis=1)
            test_accuracy, test_precision, test_recall, test_fscore, test_support = self.fuzzyHelper.getScores(test_df)

            thresholds.append(s_function_center)
            if test_accuracy > best_accuracy_score:
                best_threshold = threshold
                best_accuracy_score = test_accuracy

            print("-----------------------------------------------------------------------------------")
            print("Center Point: {}".format(s_function_center))
            print("Threshold: {}".format(threshold))
            print("Train Accuracy: {}".format(accuracy))
            print("Test Accuracy: {}".format(test_accuracy))
            print("Time: {}".format(measured_time))
            print("-----------------------------------------------------------------------------------")

            df = df.sort_values(by=["Predicted Value"])

            if show_results:
                self.decision.view()
                display(df)

            self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, "Train", "BruteForce K-Fold Threshold", accuracy, precision[0], precision[1], recall[0], recall[1], fscore[0], fscore[1], support[0], support[1], s_function_center, s_function_width, threshold, measured_time])
            self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, "Test", "BruteForce K-Fold Threshold", test_accuracy, test_precision[0], test_precision[1], test_recall[0], test_recall[1], test_fscore[0], test_fscore[1], test_support[0], test_support[1], s_function_center, s_function_width, threshold, measured_time])

        return threshold

    def worker(self, settings, constraints, s_function_width, n_folds = 10):

        skf = StratifiedKFold(n_splits= n_folds, shuffle = True, random_state=23)
        X = self.df.drop('Decision', axis=1)
        y = self.df.Decision
        s_function_centers = []
        best_accuracy_score = 0
        best_s_function_center = 0
        fuzzify = Fuzzify()

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            
            start = time.time()
            train_data = self.df.iloc[train_index]
            train_data_for_worker = train_data.copy()

            test_data = self.df.iloc[test_index]
            changed_decisions, features_number_after_reduct, implicants_number, features, decision_table_with_reduct, reductor = fuzzify.workerKFold(settings, train_data_for_worker, settings.adjustment_value)
            self.rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, settings)
            self.rule_antecedents = self.rules_extractor.worker(decision_table_with_reduct, features, self.d_results, self.decision)
            params = (s_function_width, train_data, test_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
          
            optimization_result = optimize.brute(self.fuzzyHelper.sFunctionsOptBrute, (constraints), args=params, full_output=True, finish=optimize.fmin)
            end = time.time()

            self.fuzzyHelper.prepareRules(True, self.x_range, optimization_result[0], s_function_width, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision, self.df)
            s_function_center = optimization_result[0][0]
            _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, train_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            # display(df.sort_values(by=["Predicted Value"]))

            accuracy = 1 - optimization_result[1]
            measured_time = end - start
            s_function_center = optimization_result[0][0]
            _, df = self.fuzzyHelper.sFunctionsValue(s_function_center, s_function_width, test_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            test_accuracy, test_precision, test_recall, test_fscore, test_support = self.fuzzyHelper.getScores(df, False)
            # if idx == 0:
            # display(df.sort_values(by=["Predicted Value"]))
                # display(df.sort_values(by=["Predicted Value"]).tail(10))
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
            _, df = self.fuzzyHelper.sFunctionsValue(0.5, s_function_width, test_data, settings, self.x_range, self.rules_extractor, self.rule_antecedents, self.d_results, self.decision)
            test_accuracy, test_precision, test_recall, test_fscore, test_support = self.fuzzyHelper.getScores(df, False)
            print("Test Accuracy: {}".format(test_accuracy))
            print("Test F-Score: {}".format(test_fscore))



            self.fuzzyHelper.saveResults(settings.results_folder + settings.results_file, [settings.test_type, settings.dataset_name, settings.style, settings.gausses, settings.adjustment, "Train", "BruteForce S-Functions K-Fold {}".format(idx), test_accuracy, test_precision[0], test_precision[1], test_recall[0], test_recall[1], test_fscore[0], test_fscore[1], test_support[0], test_support[1], s_function_center, s_function_width, "---", measured_time])

        mean_s_function_center = sum(s_function_centers) / len(s_function_centers) 
        return best_s_function_center, mean_s_function_center, changed_decisions, features_number_after_reduct, implicants_number
