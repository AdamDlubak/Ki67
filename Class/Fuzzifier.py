import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from scipy.stats import norm
import sympy as sy
from sympy import symbols, Eq, solve

class Fuzzifier(object):

    def __init__(self, settings, d_results):
        self.features = []
        self.feature_labels = []
        self.x_range = np.arange(settings.set_min, settings.set_max, settings.fuzzy_sets_precision)
        self.d_results = d_results
        self.settings = settings
        self.fuzzify_parameters = []

        for i in range(0, self.settings.feature_numbers):
            feature_label = "F" + str(i)
            self.features.append(
                ctrl.Antecedent(self.x_range, feature_label))
            self.feature_labels.append(feature_label)

        self.decision = ctrl.Consequent(self.x_range, 'Decision')

    def gaussLeft(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x <= mean
        y[idx] = 1 - np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
        return y

    def gaussianFunction(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r):
        y = np.zeros(len(x))

        if (center_l == -1) and (center_r == -1):
            idx = (x < bottom_r) & (x >= bottom_l)
            y[idx] =  np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        elif center_l == -1:  
            idx = (x >= bottom_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / sigma ** 2.)
        elif center_r == -1:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / sigma ** 2.)
            idx = (x >= center_l) & (x < bottom_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
        else:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / sigma ** 2.)

            idx = (x >= center_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
          
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / sigma ** 2.)
        return y

    def gaussRight(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x >= mean
        y[idx] = 1 - np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
        return y

    def leftGaussianValue(self, x, mean, sigma):
        if x <= mean:
            verylow_value = 1 - np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        else:
            verylow_value = 0

        return verylow_value

    def gaussianValue(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r):

        if (center_l == -1) and (center_r == -1):       
            if (x < bottom_r) and (x >= bottom_l):
                y =  np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            else:
                y = 0
        elif center_l == -1:
            if (x < center_r) and (x >= bottom_l):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x < bottom_r) and (x >= center_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / sigma ** 2.)
            else:
                y = 0
        elif center_r == -1:
            if (x < bottom_r) and (x >= center_l):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x < center_l) and (x >= bottom_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / sigma ** 2.)
            else:
                y = 0
        else:
            if (x >= bottom_l) and (x < center_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / sigma ** 2.)
            elif (x >= center_l) and (x < center_r):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x >= center_r) and (x < bottom_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / sigma ** 2.)
            else:
                y = 0
        return y

    def rightGaussianValue(self, x, mean, sigma):
        if x >= mean:
            veryhigh_value = 1 - np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        else:
            veryhigh_value = 0
        
        return veryhigh_value

    def calculateSigma(self, mean_1, mean_2):
        x, sigma = symbols('x sigma')
        eq1 = Eq(sy.exp(-((x - mean_1) ** 2.) / sigma ** 2.) - 0.5)
        eq2 = Eq(sy.exp(-((x - mean_2) ** 2.) / sigma ** 2.) - 0.5)

        res = solve((eq1,eq2), (x, sigma))
        for x in res:
            if x[1] >= 0:
                x_value = x[0]
                sigma_value = x[1]
                break

        return np.float64(x_value), np.float64(sigma_value)

    def numbersToRowSets(self, idx, values, mean):

        values = values.values
        return_array = []

        if mean == -1:
            mean, _ = norm.fit(values)
        if mean == -2:
            mean = 0.5
        
        _, sigma = norm.fit(values) 

        if self.settings.show_results:
            print("Feature " + str(idx) + ":")
            print("\tMean: " + str(mean))
            print("\tSigma: " + str(sigma))

        if self.settings.gausses == 3:
            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean, sigma)
            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, sigma, -1, -1, 0, 1.01)
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean, sigma)

        elif self.settings.gausses == 5:
            cr_low, sigma = self.calculateSigma(mean * (1/2), mean)
            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean * (1/2), sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean * (1/2), sigma, -1, cr_low, 0, mean)
 
            cr_middle, _ = self.calculateSigma(mean, mean * (3/2))
            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, sigma, cr_low, cr_middle, mean * (1/2), mean * (3/2))

            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean * (3/2), sigma, cr_middle, -1, mean, 1.01)  
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean * (3/2), sigma)

        elif self.settings.gausses == 7:
            cr_low, sigma = self.calculateSigma(mean * (1/3), mean * (2/3))
            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean * (1/3), sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean * (1/3), sigma, -1, cr_low, 0, mean * (2/3))
            
            cr_middlelow, sigma = self.calculateSigma(mean * (2/3), mean)
            self.features[idx][self.settings.middlelow] = self.gaussianFunction(self.x_range, mean * (2/3), sigma, cr_low, cr_middlelow, mean * (1/3), mean)
            
            cr_middle, sigma = self.calculateSigma(mean, mean * (4/3))
            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, sigma, cr_middlelow, cr_middle, mean * (2/3), mean * (4/3))
            
            cr_middlehigh, sigma = self.calculateSigma(mean * (4/3), mean * (5/3))
            self.features[idx][self.settings.middlehigh] = self.gaussianFunction(self.x_range, mean * (4/3), sigma, cr_middle, cr_middlehigh, mean, mean * (5/3))
            
            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean * (5/3), sigma, cr_middlehigh, -1, mean * (4/3), 1.01)  
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean * (5/3), sigma)
            
        elif self.settings.gausses == 9:
   
            cr_low, sigma = self.calculateSigma(mean * (1/4), mean * (2/4))
            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean * (1/4), sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean * (1/4), sigma, -1, cr_low, 0, mean * (2/4))

            cr_middlelowminus, sigma = self.calculateSigma(mean * (2/4), mean * (3/4))
            self.features[idx][self.settings.middlelowminus] = self.gaussianFunction(self.x_range, mean * (2/4), sigma, cr_low, cr_middlelowminus, mean * (1/4), mean * (3/4))

            cr_middlelow, sigma = self.calculateSigma(mean * (3/4), mean)
            self.features[idx][self.settings.middlelow] = self.gaussianFunction(self.x_range, mean * (3/4), sigma, cr_middlelowminus, cr_middlelow, mean * (2/4), mean)

            cr_middle, sigma = self.calculateSigma(mean, mean * (5/4))
            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, sigma, cr_middlelow, cr_middle, mean * (3/4), mean * (5/4))
            
            cr_middlehigh, sigma = self.calculateSigma(mean * (5/4), mean * (6/4))
            self.features[idx][self.settings.middlehigh] = self.gaussianFunction(self.x_range, mean * (5/4), sigma, cr_middle, cr_middlehigh, mean, mean * (6/4))
            
            cr_middlehighplus, sigma = self.calculateSigma(mean * (6/4), mean * (7/4))
            self.features[idx][self.settings.middlehighplus] = self.gaussianFunction(self.x_range, mean * (6/4), sigma, cr_middlehigh, cr_middlehighplus, mean * (5/4), mean * (7/4))

            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean * (7/4), sigma, cr_middlehighplus, -1, mean * (6/4), 1.01)  
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean * (7/4), sigma)

        elif self.settings.gausses == 11:
            cr_low, sigma = self.calculateSigma(mean * (1/5), mean * (2/5))
            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean * (1/5), sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean * (1/5), sigma, -1, cr_low, 0, mean * (2/5))

            cr_middlelowminus, sigma = self.calculateSigma(mean * (2/5), mean * (3/5))
            self.features[idx][self.settings.middlelowminus] = self.gaussianFunction(self.x_range, mean * (2/5), sigma, cr_low, cr_middlelowminus, mean * (1/5), mean * (3/5))
          
            cr_middlelow, sigma = self.calculateSigma(mean * (3/5), mean * (4/5))
            self.features[idx][self.settings.middlelow] = self.gaussianFunction(self.x_range, mean * (3/5), sigma, cr_middlelowminus, cr_middlelow, mean * (2/5), mean * (4/5))
   
            cr_middlelowplus, sigma = self.calculateSigma(mean * (4/5), mean * (5/5))
            self.features[idx][self.settings.middlelowplus] = self.gaussianFunction(self.x_range, mean * (4/5), sigma, cr_middlelow, cr_middlelowplus, mean * (3/5), mean * (5/5))

            cr_middle, sigma = self.calculateSigma(mean, mean * (6/5))
            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, sigma, cr_middlelowplus, cr_middle, mean * (4/5), mean * (6/5))

            cr_middlehighminus, sigma = self.calculateSigma(mean * (6/5), mean * (7/5))
            self.features[idx][self.settings.middlehighminus] = self.gaussianFunction(self.x_range, mean * (6/5), sigma, cr_middle, cr_middlehighminus, mean * (5/5), mean * (7/5))

            cr_middlehigh, sigma = self.calculateSigma(mean * (7/5), mean * (8/5))
            self.features[idx][self.settings.middlehigh] = self.gaussianFunction(self.x_range, mean * (7/5), sigma, cr_middlehighminus, cr_middlehigh, mean * (6/5), mean * (8/5))
            
            cr_middlehighplus, sigma = self.calculateSigma(mean * (8/5), mean * (9/5))
            self.features[idx][self.settings.middlehighplus] = self.gaussianFunction(self.x_range, mean * (8/5), sigma, cr_middlehigh, cr_middlehighplus, mean * (7/5), mean * (9/5))
            
            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean * (9/5), sigma, cr_middlehighplus, -1, mean * (8/5), 1.01)  
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean * (9/5), sigma)

        self.fuzzify_parameters.append(mean)
        self.fuzzify_parameters.append(sigma)
                              
        for x in values:
            verylow_value = low_value = middlelowminus_value = middlelow_value = middlelowplus_value = middle_value = middlehighminus_value = middlehigh_value = middlehighplus_value = high_value = veryhigh_value = 0

            if self.settings.gausses == 3:
                verylow_value = self.leftGaussianValue(x, mean, sigma)
                middle_value = self.gaussianValue(x, mean, sigma, -1, -1, 0, 1.01)
                veryhigh_value = self.rightGaussianValue(x, mean, sigma)
                
            if self.settings.gausses == 5:
                verylow_value = self.leftGaussianValue(x, mean * (1/2), sigma)
                low_value = self.gaussianValue(x, mean * (1/2), sigma, -1, cr_low, 0, mean)
                middle_value = self.gaussianValue(x, mean, sigma, cr_low, cr_middle, mean * (1/2), mean * (3/2))
                high_value = self.gaussianValue(x, mean * (3/2), sigma, cr_middle, -1, mean, 1.01)  
                veryhigh_value = self.rightGaussianValue(x, mean * (3/2), sigma)
    
            elif self.settings.gausses == 7:
                verylow_value = self.leftGaussianValue(x, mean * (1/3), sigma)
                low_value = self.gaussianValue(x, mean * (1/3), sigma, -1, cr_low, 0, mean * (2/3))
                middlelow_value = self.gaussianValue(x, mean * (2/3), sigma, cr_low, cr_middlelow, mean * (1/3), mean)
                middle_value = self.gaussianValue(x, mean, sigma, cr_middlelow, cr_middle, mean * (2/3), mean * (4/3))
                middlehigh_value = self.gaussianValue(x, mean * (4/3), sigma, cr_middle, cr_middlehigh, mean, mean * (5/3))
                high_value = self.gaussianValue(x, mean * (5/3), sigma, cr_middlehigh, -1, mean * (4/3), 1.01)  
                veryhigh_value = self.rightGaussianValue(x, mean * (5/3), sigma)

            elif self.settings.gausses == 9:
                verylow_value = self.leftGaussianValue(x, mean * (1/4), sigma)
                low_value = self.gaussianValue(x, mean * (1/4), sigma, -1, cr_low, 0, mean * (2/4))
                middlelowminus_value = self.gaussianValue(x, mean * (2/4), sigma, cr_low, cr_middlelowminus, mean * (1/4), mean * (3/4))
                middlelow_value = self.gaussianValue(x, mean * (3/4), sigma, cr_middlelowminus, cr_middlelow, mean * (2/4), mean)
                middle_value = self.gaussianValue(x, mean, sigma, cr_middlelow, cr_middle, mean * (3/4), mean * (5/4))
                middlehigh_value = self.gaussianValue(x, mean * (5/4), sigma, cr_middle, cr_middlehigh, mean, mean * (6/4))
                middlehighplus_value = self.gaussianValue(x, mean * (6/4), sigma, cr_middlehigh, cr_middlehighplus, mean * (5/4), mean * (7/4))
                high_value = self.gaussianValue(x, mean * (7/4), sigma, cr_middlehighplus, -1, mean * (6/4), 1.01)  
                veryhigh_value = self.rightGaussianValue(x, mean * (7/4), sigma)

            elif self.settings.gausses == 11:
                verylow_value = self.leftGaussianValue(x, mean * (1/5), sigma)
                low_value = self.gaussianValue(x, mean * (1/5), sigma, -1, cr_low, 0, mean * (2/5))
                middlelowminus_value = self.gaussianValue(x, mean * (2/5), sigma, cr_low, cr_middlelowminus, mean * (1/5), mean * (3/5))
                middlelow_value = self.gaussianValue(x, mean * (3/5), sigma, cr_middlelowminus, cr_middlelow, mean * (2/5), mean * (4/5))
                middlelowplus_value = self.gaussianValue(x, mean * (4/5), sigma, cr_middlelow, cr_middlelowplus, mean * (3/5), mean * (5/5))
                middle_value = self.gaussianValue(x, mean, sigma, cr_middlelowplus, cr_middle, mean * (4/5), mean * (6/5))
                middlehighminus_value = self.gaussianValue(x, mean * (6/5), sigma, cr_middle, cr_middlehighminus, mean * (5/5), mean * (7/5))
                middlehigh_value = self.gaussianValue(x, mean * (7/5), sigma, cr_middlehighminus, cr_middlehigh, mean * (6/5), mean * (8/5))
                middlehighplus_value = self.gaussianValue(x, mean * (8/5), sigma, cr_middlehigh, cr_middlehighplus, mean * (7/5), mean * (9/5))
                high_value = self.gaussianValue(x, mean * (9/5), sigma, cr_middlehighplus, -1, mean * (8/5), 1.01)  
                veryhigh_value = self.rightGaussianValue(x, mean * (9/5), sigma)


            max_value = max([verylow_value, low_value, middlelowminus_value, middlelow_value, middlelowplus_value, middle_value, middlehighminus_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value])
            if max_value == verylow_value:
                return_value = self.features[idx][self.settings.verylow].label    
            elif max_value == low_value:
                 return_value = self.features[idx][self.settings.low].label                   
            elif max_value == middlelowminus_value:
                 return_value = self.features[idx][self.settings.middlelowminus].label  
            elif max_value == middlelow_value:
                 return_value = self.features[idx][self.settings.middlelow].label  
            elif max_value == middlelowplus_value:
                 return_value = self.features[idx][self.settings.middlelowplus].label                                                     
            elif max_value == middle_value:
                 return_value = self.features[idx][self.settings.middle].label      
            elif max_value == middlehighminus_value:
                 return_value = self.features[idx][self.settings.middlehighminus].label      
            elif max_value == middlehigh_value:
                 return_value = self.features[idx][self.settings.middlehigh].label      
            elif max_value == middlehighplus_value:
                 return_value = self.features[idx][self.settings.middlehighplus].label      
            elif max_value == high_value:
                 return_value = self.features[idx][self.settings.high].label                                                                                      
            else:
                 return_value = self.features[idx][self.settings.veryhigh].label                                                                                      

            return_array.append(return_value)

        return return_array
 
    def presentFuzzyFeature_Charts(self):
        for x in self.features:
            x.view()

    def fuzzify(self, features_table, mean_param):

        if isinstance(mean_param, (int, np.integer)):
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToRowSets(idx, features_table[x.label], mean_param)
        
        else:
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToRowSets(idx, features_table[x.label], mean_param[idx])
        
        if self.settings.show_results:
            self.presentFuzzyFeature_Charts()
            display(features_table)

        return features_table, self.feature_labels, self.features, self.decision, self.fuzzify_parameters
