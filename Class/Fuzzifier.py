import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from scipy.stats import norm

class Fuzzifier(object):

    def __init__(self, variables, d_results):
        self.features = []
        self.feature_labels = []
        self.x_range = np.arange(variables["set_min"], variables["set_max"], variables["fuzzy_sets_precision"])
        self.d_results = d_results
        self.variables = variables
        self.fuzzify_parameters = []

        for i in range(0, self.variables["feature_numbers"]):
            feature_label = "F" + str(i)
            self.features.append(
                ctrl.Antecedent(self.x_range, feature_label))
            self.feature_labels.append(feature_label)

        self.decision = ctrl.Consequent(self.x_range, 'Decision')

   
    def gaussLeft(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x <= mean
        y[idx] = 1 - fuzz.gaussmf(x[idx], mean, sigma)
        return y
        
    def gaussMiddleLeft(self, x, mean, sigma):
        y =  fuzz.gaussmf(x, mean, sigma)
        return y

    def gaussMiddleRight(self, x, mean, sigma):
        y = fuzz.gaussmf(x, mean, sigma)
        return y

    def gaussRight(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x >= mean
        y[idx] = 1 - fuzz.gaussmf(x[idx], mean, sigma)
        return y

 
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


        self.features[idx][self.variables["verylow"]] = self.gaussLeft(self.x_range, mean, sigma)
        self.features[idx][self.variables["middle"]] = fuzz.gaussmf(self.x_range, mean, sigma)
        self.features[idx][self.variables["veryhigh"]] = self.gaussRight(self.x_range, mean, sigma)

        if self.variables["gausses"] == 5:
            self.features[idx][self.variables["middlelow"]] = self.gaussMiddleLeft(self.x_range, mean * 0.5, sigma)
            self.features[idx][self.variables["middlehigh"]] = self.gaussMiddleRight(self.x_range, mean * 1.5, sigma)

        elif self.variables["gausses"] == 7:
            self.features[idx][self.variables["low"]] = self.gaussMiddleLeft(self.x_range, mean * 0.33, sigma)
            self.features[idx][self.variables["middlelow"]] = self.gaussMiddleLeft(self.x_range, mean * 0.66, sigma)
            self.features[idx][self.variables["middlehigh"]] = self.gaussMiddleRight(self.x_range, mean * 1.33, sigma)
            self.features[idx][self.variables["high"]] = self.gaussMiddleRight(self.x_range, mean * 1.66, sigma)
            
        elif self.variables["gausses"] == 9:
            self.features[idx][self.variables["low"]] = self.gaussMiddleLeft(self.x_range, mean * 0.25, sigma)
            self.features[idx][self.variables["middlelowminus"]] = self.gaussMiddleLeft(self.x_range, mean * 0.5, sigma)
            self.features[idx][self.variables["middlelow"]] = self.gaussMiddleLeft(self.x_range, mean * 0.75, sigma)
            self.features[idx][self.variables["middlehigh"]] = self.gaussMiddleRight(self.x_range, mean * 1.25, sigma)
            self.features[idx][self.variables["middlehighplus"]] = self.gaussMiddleRight(self.x_range, mean * 1.5, sigma)
            self.features[idx][self.variables["high"]] = self.gaussMiddleRight(self.x_range, mean * 1.75, sigma)
                        
        elif self.variables["gausses"] == 11:
            self.features[idx][self.variables["low"]] = self.gaussMiddleLeft(self.x_range, mean * 0.20, sigma)
            self.features[idx][self.variables["middlelowminus"]] = self.gaussMiddleLeft(self.x_range, mean * 0.4, sigma)
            self.features[idx][self.variables["middlelow"]] = self.gaussMiddleLeft(self.x_range, mean * 0.6, sigma)
            self.features[idx][self.variables["middlelowplus"]] = self.gaussMiddleLeft(self.x_range, mean * 0.8, sigma)
            self.features[idx][self.variables["middlehighminus"]] = self.gaussMiddleRight(self.x_range, mean * 1.2, sigma)
            self.features[idx][self.variables["middlehigh"]] = self.gaussMiddleRight(self.x_range, mean * 1.4, sigma)
            self.features[idx][self.variables["middlehighplus"]] = self.gaussMiddleRight(self.x_range, mean * 1.6, sigma)
            self.features[idx][self.variables["high"]] = self.gaussMiddleRight(self.x_range, mean * 1.8, sigma)
                              

        for x in values:
            verylow_value = low_value = middlelowminus_value = middlelow_value = middlelowplus_value = middle_value = middlehighminus_value = middlehigh_value = middlehighplus_value = high_value = veryhigh_value = 0

            middle_value = fuzz.gaussmf(x, mean, sigma)
            if x <= mean:
                verylow_value = 1 - fuzz.gaussmf(x, mean, sigma)
            else:
                verylow_value = 0

            if x >= mean:
                veryhigh_value = 1 - fuzz.gaussmf(x, mean, sigma)
            else:
                veryhigh_value = 0

            if self.variables["gausses"] == 5:
                middlelow_value = fuzz.gaussmf(x, mean * 0.5, sigma)
                middlehigh_value = fuzz.gaussmf(x, mean * 1.5, sigma)


            elif self.variables["gausses"] == 7:
                low_value = fuzz.gaussmf(x, mean * 0.33, sigma)
                middlelow_value = fuzz.gaussmf(x, mean * 0.66, sigma)
                middlehigh_value = fuzz.gaussmf(x, mean * 1.33, sigma)
                high_value = fuzz.gaussmf(x, mean * 1.66, sigma)

            elif self.variables["gausses"] == 9:
                low_value = fuzz.gaussmf(x, mean * 0.25, sigma)
                middlelowminus_value = fuzz.gaussmf(x, mean * 0.5, sigma)
                middlelow_value = fuzz.gaussmf(x, mean * 0.75, sigma)
                middlehigh_value = fuzz.gaussmf(x, mean * 1.25, sigma)
                middlehighplus_value = fuzz.gaussmf(x, mean * 1.5, sigma)
                high_value = fuzz.gaussmf(x, mean * 1.75, sigma)
                            
            elif self.variables["gausses"] == 11:
                low_value = fuzz.gaussmf(x, mean * 0.2, sigma)
                middlelowminus_value = fuzz.gaussmf(x, mean * 0.4, sigma)
                middlelow_value = fuzz.gaussmf(x, mean * 0.6, sigma)
                middlelowplus_value = fuzz.gaussmf(x, mean * 0.8, sigma)
                middlehighminus_value = fuzz.gaussmf(x, mean * 1.2, sigma)
                middlehigh_value = fuzz.gaussmf(x, mean * 1.4, sigma)
                middlehighplus_value = fuzz.gaussmf(x, mean * 1.6, sigma)
                high_value = fuzz.gaussmf(x, mean * 1.8, sigma)
            
            max_value = max([verylow_value, low_value, middlelowminus_value, middlelow_value, middlelowplus_value, middle_value, middlehighminus_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value])
            
            if max_value == verylow_value:
                return_value = self.features[idx][self.variables["verylow"]].label    
            elif max_value == low_value:
                 return_value = self.features[idx][self.variables["low"]].label                   
            elif max_value == middlelowminus_value:
                 return_value = self.features[idx][self.variables["middlelowminus"]].label  
            elif max_value == middlelow_value:
                 return_value = self.features[idx][self.variables["middlelow"]].label  
            elif max_value == middlelowplus_value:
                 return_value = self.features[idx][self.variables["middlelowplus"]].label                                                     
            elif max_value == middle_value:
                 return_value = self.features[idx][self.variables["middle"]].label      
            elif max_value == middlehighminus_value:
                 return_value = self.features[idx][self.variables["middlehighminus"]].label      
            elif max_value == middlehigh_value:
                 return_value = self.features[idx][self.variables["middlehigh"]].label      
            elif max_value == middlehighplus_value:
                 return_value = self.features[idx][self.variables["middlehighplus"]].label      
            elif max_value == high_value:
                 return_value = self.features[idx][self.variables["high"]].label                                                                                      
            elif max_value == veryhigh_value:
                 return_value = self.features[idx][self.variables["veryhigh"]].label                                                                                      

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
