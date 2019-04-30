import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from scipy.stats import norm

class Fuzzifier(object):

    def __init__(self, variables, levels, d_results):
        self.features = []
        self.feature_labels = []
        self.x_range = np.arange(variables["set_min"], variables["set_max"], variables["fuzzy_sets_precision"])
        self.levels = levels
        self.d_results = d_results
        self.variables = variables
        self.fuzzify_parameters = []

        for i in range(0, self.variables["feature_numbers"]):
            feature_label = "F" + str(i)
            self.features.append(
                ctrl.Antecedent(self.x_range, feature_label))
            self.feature_labels.append(feature_label)
            self.features[i].automf(names=self.levels)

        self.decision = ctrl.Consequent(self.x_range, 'Decision')
        # self.decision.automf(names=self.d_results)
        # self.decision['negatif'] = fuzz.trimf(np.arange(self.variables["set_min"], self.variables["set_max"], self.variables["fuzzy_sets_precision"]), [0, 0, 0.2])
        # self.decision['positif'] = fuzz.trimf(np.arange(self.variables["set_min"], self.variables["set_max"], self.variables["fuzzy_sets_precision"]), [0, 0.7, 1])

    def gaussMiddleLeft(self, x, mean, sigma):
        mean = mean * 0.5
        y =  fuzz.gaussmf(x, mean, sigma)
        return y

    def gaussLeft(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x <= mean
        y[idx] = 1 - fuzz.gaussmf(x[idx], mean, sigma)
        return y


    def gaussRight(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x >= mean
        y[idx] = 1 - fuzz.gaussmf(x[idx], mean, sigma)
        return y

    def gaussMiddleRight(self, x, mean, sigma):
        mean = mean * 1.5
        y = fuzz.gaussmf(x, mean, sigma)
        return y


    def numbersToFiveRowSets(self, idx, values, sigma, mean):

        values = values.values
        return_array = []

        if sigma == -1 or mean == -1:
            mean, sigma = norm.fit(values)

        if self.variables["show_results"]:
            print("Feature " + str(idx) + ":")
            print("\tMean: " + str(mean))
            print("\tSigma: " + str(sigma))

        self.fuzzify_parameters.append(mean)
        self.fuzzify_parameters.append(sigma)
        self.features[idx][self.variables["low"]] = self.gaussLeft(self.x_range, mean, sigma)
        self.features[idx][self.variables["middlelow"]] = self.gaussMiddleLeft(self.x_range, mean, sigma)
        self.features[idx][self.variables["middle"]] = fuzz.gaussmf(self.x_range, mean, sigma)
        self.features[idx][self.variables["middlehigh"]] = self.gaussMiddleRight(self.x_range, mean, sigma)
        self.features[idx][self.variables["high"]] = self.gaussRight(self.x_range, mean, sigma)

        for x in values:
            middle_value = fuzz.gaussmf(x, mean, sigma)
            middlelow_value = fuzz.gaussmf(x, mean * 0.5, sigma)
            middlehigh_value = fuzz.gaussmf(x, mean * 1.5, sigma)
            
            if x <= mean:
                low_value = 1 - fuzz.gaussmf(x, mean, sigma)
            else:
                low_value = 0

            if x >= mean:
                high_value = 1 - fuzz.gaussmf(x, mean, sigma)
            else:
                high_value = 0

            max_value = max([middlelow_value, low_value, middle_value, high_value, middlehigh_value])
            if max_value == middlelow_value:
                return_value = self.features[idx][self.variables["middlelow"]].label    
            elif max_value == low_value:
                 return_value = self.features[idx][self.variables["low"]].label                   
            elif max_value == middle_value:
                 return_value = self.features[idx][self.variables["middle"]].label  
            elif max_value == high_value:
                 return_value = self.features[idx][self.variables["high"]].label  
            elif max_value == middlehigh_value:
                 return_value = self.features[idx][self.variables["middlehigh"]].label                                                     
            
            return_array.append(return_value)

        return return_array


    def numbersToRowSets(self, idx, values, sigma, mean):

        values = values.values
        return_array = []

        if sigma == -1 or mean == -1:
            mean, sigma = norm.fit(values)

        if self.variables["show_results"]:
            print("Feature " + str(idx) + ":")
            print("\tMean: " + str(mean))
            print("\tSigma: " + str(sigma))

        self.fuzzify_parameters.append(mean)
        self.fuzzify_parameters.append(sigma)
        self.features[idx][self.variables["low"]] = self.gaussLeft(self.x_range, mean, sigma)
        self.features[idx][self.variables["middle"]] = fuzz.gaussmf(self.x_range, mean, sigma)
        self.features[idx][self.variables["high"]] = self.gaussRight(self.x_range, mean, sigma)

        for x in values:
            middle_value = fuzz.gaussmf(x, mean, sigma)
            if x <= mean:
                low_value = 1 - fuzz.gaussmf(x, mean, sigma)
            else:
                low_value = 0

            if x >= mean:
                high_value = 1 - fuzz.gaussmf(x, mean, sigma)
            else:
                high_value = 0

            max_value = high_value
            return_value = self.features[idx][self.variables["high"]].label

            if middle_value > max_value:
                max_value = middle_value
                return_value = self.features[idx][self.variables["middle"]].label

            if low_value > max_value:
                max_value = low_value
                return_value = self.features[idx][self.variables["low"]].label

            return_array.append(return_value)

        return return_array

    def presentFuzzyFeature_Charts(self):
        for x in self.features:
            x.view()


    def fuzzify(self, features_table, sigma_mean_params):
        if isinstance(sigma_mean_params, (int, np.integer)):
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToRowSets(idx, features_table[x.label], sigma_mean_params, sigma_mean_params)
        else:
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToRowSets(idx, features_table[x.label], sigma_mean_params[idx*2], sigma_mean_params[idx*2 + 1])
         
        if self.variables["show_results"]:
            self.presentFuzzyFeature_Charts()
            display(features_table)

        return features_table, self.feature_labels, self.features, self.decision, self.fuzzify_parameters

    def fuzzifyFive(self, features_table, sigma_mean_params):
        if isinstance(sigma_mean_params, (int, np.integer)):
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToFiveRowSets(idx, features_table[x.label], sigma_mean_params, sigma_mean_params)
        else:
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToFiveRowSets(idx, features_table[x.label], sigma_mean_params[idx*2], sigma_mean_params[idx*2 + 1])
         
        # if self.variables["show_results"]:
        self.presentFuzzyFeature_Charts()
        display(features_table)

        return features_table, self.feature_labels, self.features, self.decision, self.fuzzify_parameters