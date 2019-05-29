import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from scipy.stats import norm
import sympy as sy
from sympy import symbols, Eq, solve

class FuzzifierProgressive(object):

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

    def gaussianFunction(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r, sigma_l, sigma_r):
        y = np.zeros(len(x))
        if (center_l == -1) and (center_r == -1):
            idx = (x < bottom_r) & (x >= bottom_l)
            y[idx] =  np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        elif center_l == -1:  
            idx = (x >= bottom_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / sigma_r ** 2.)
        elif center_r == -1:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / sigma_l ** 2.)
            idx = (x >= center_l) & (x < bottom_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
        else:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / sigma_l ** 2.)

            idx = (x >= center_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
          
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / sigma_r ** 2.)
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

    def gaussianValue(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r, sigma_l, sigma_r):
        
        if (center_l == -1) and (center_r == -1):       
            if (x < bottom_r) and (x >= bottom_l):
                y =  np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            else:
                y = 0
        elif center_l == -1:
            if (x < center_r) and (x >= bottom_l):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x < bottom_r) and (x >= center_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / sigma_r ** 2.)
            else:
                y = 0
        elif center_r == -1:
            if (x < bottom_r) and (x >= center_l):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x < center_l) and (x >= bottom_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / sigma_l ** 2.)
            else:
                y = 0
        else:
            if (x >= bottom_l) and (x < center_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / sigma_l ** 2.)
            elif (x >= center_l) and (x < center_r):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x >= center_r) and (x < bottom_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / sigma_r ** 2.)
            else:
                y = 0
        return y

    def rightGaussianValue(self, x, mean, sigma):
        if x >= mean:
            veryhigh_value = 1 - np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        else:
            veryhigh_value = 0
        
        return veryhigh_value

    def calculateBothSigma(self, mean_1, mean_2):
        x, sigma = symbols('x sigma')
        eq1 = Eq(sy.exp(-((x - mean_1) ** 2.) / sigma ** 2.) - 0.5)
        eq2 = Eq(sy.exp(-((x - mean_2) ** 2.) / sigma ** 2.) - 0.5)

        res = solve((eq1,eq2), (x, sigma))
        x_value = 1
        sigma_value = 1

        for x in res:
            if x[1] >= 0:
                if x[1] < sigma_value:
                    x_value = x[0]
                    sigma_value = x[1]

        return np.float64(x_value), np.float64(sigma_value)

    def calculateSigma(self, mean_1, mean_2, sigma_value):
        x, sigma = symbols('x sigma')
        eq1 = Eq(sy.exp(-((x - mean_1) ** 2.) / sigma ** 2.) - 0.5)
        eq2 = Eq(sy.exp(-((x - mean_2) ** 2.) / sigma_value ** 2.) - 0.5)

        res = solve((eq1,eq2), (x, sigma))
        x_value = 1
        sigma_value = 1

        for x in res:
            if x[1] >= 0:
                if x[1] < sigma_value:
                    x_value = x[0]
                    sigma_value = x[1]

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

        minus_mean = 1 - mean
        if self.settings.gausses == 3:
            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean, sigma)
            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, sigma, -1, -1, 0, 1.01, -1, -1)
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean, sigma)

        elif self.settings.gausses == 5:
            if (minus_mean * (1/4) >= (mean * (1/4))):
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/4)))
            else:
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/4)))

            cr_low, left_sigma = self.calculateSigma(mean - (mean * (1/4)), mean, center_sigma)
            _, right_sigma     = self.calculateSigma(mean, mean + (minus_mean * (1/4)), center_sigma)

            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean - (mean * (1/4)), left_sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean - (mean * (1/4)), left_sigma, -1, cr_low, 0, mean, -1, center_sigma)
            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, center_sigma, cr_low, cr_middle, mean - (mean * (1/4)), mean + (minus_mean * (1/4)), left_sigma, right_sigma)
            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/4)), right_sigma, cr_middle, -1, mean, 1.01, center_sigma, -1)  
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean + (minus_mean * (1/4)), right_sigma)

        elif self.settings.gausses == 7:
            if (minus_mean * (1/10) >= (mean * (1/10))):
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/10)))
            else:
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/10)))
            
            cr_middlelow, middle_left_sigma = self.calculateSigma(mean - (mean * (1/10)), mean, center_sigma)
            cr_low, left_sigma = self.calculateSigma(mean - (mean * (4/10)), mean - (mean * (1/10)), middle_left_sigma)

            _, middle_right_sigma = self.calculateSigma(mean + (minus_mean * (1/10)), mean, center_sigma)
            cr_high, right_sigma = self.calculateSigma(mean + (minus_mean * (4/10)), mean + (minus_mean * (1/10)), middle_right_sigma)

            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean - (mean * (4/10)), left_sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean - (mean * (4/10)), left_sigma, -1, cr_low, 0, mean - (mean * (1/10)), -1, middle_left_sigma)
            self.features[idx][self.settings.middlelow] = self.gaussianFunction(self.x_range, mean - (mean * (1/10)), middle_left_sigma, cr_low, cr_middlelow, mean - (mean * (4/10)), mean, left_sigma, center_sigma)

            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, center_sigma, cr_middlelow, cr_middle, mean - (mean * (1/10)), mean + (minus_mean * (1/10)), middle_left_sigma, middle_right_sigma)

            self.features[idx][self.settings.middlehigh] = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/10)), middle_right_sigma, cr_middle, cr_high, mean, mean + (minus_mean * (4/10)), center_sigma, right_sigma)
            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean + (minus_mean * (4/10)), right_sigma, cr_high, -1, mean + (minus_mean * (1/10)), 1.01, middle_right_sigma, -1)  
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean + (minus_mean * (4/10)), right_sigma)
            
        elif self.settings.gausses == 9:
            if (minus_mean * (1/20) >= (mean * (1/20))):
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/20)))
            else:
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/20)))
            
            cr_middlelow, middle_left_sigma = self.calculateSigma(mean - (mean * (1/20)), mean, center_sigma)
            cr_middlelowminus, middle_left_minus_sigma = self.calculateSigma(mean - (mean * (4/20)), mean - (mean * (1/20)), middle_left_sigma)
            cr_low, left_sigma = self.calculateSigma(mean - (mean * (10/20)), mean - (mean * (4/20)), middle_left_minus_sigma)

            _, middle_right_sigma = self.calculateSigma(mean + (minus_mean * (1/20)), mean, center_sigma)
            cr_middlehighplus, middle_right_plus_sigma = self.calculateSigma(mean + (minus_mean * (4/20)), mean + (minus_mean * (1/20)), middle_right_sigma)
            cr_high, right_sigma = self.calculateSigma(mean + (minus_mean * (10/20)), mean + (minus_mean * (4/20)), middle_right_plus_sigma)

            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean - (mean * (10/20)), left_sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean - (mean * (10/20)), left_sigma, -1, cr_low, 0, mean - (mean * (4/20)), -1, middle_left_minus_sigma)
            self.features[idx][self.settings.middlelowminus] = self.gaussianFunction(self.x_range, mean - (mean * (4/20)), middle_left_minus_sigma, cr_low, cr_middlelowminus, mean - (mean * (10/20)), mean - (mean * (1/20)), left_sigma, middle_left_sigma)
            self.features[idx][self.settings.middlelow] = self.gaussianFunction(self.x_range, mean - (mean * (1/20)), middle_left_sigma, cr_middlelowminus, cr_middlelow, mean - (mean * (4/20)), mean, middle_left_minus_sigma, center_sigma)

            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, center_sigma, cr_middlelow, cr_middle, mean - (mean * (1/20)), mean + (minus_mean * (1/20)), middle_left_sigma, middle_right_sigma)

            self.features[idx][self.settings.middlehigh] = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/20)), middle_right_sigma, cr_middle, cr_high, mean, mean + (minus_mean * (4/20)), center_sigma, middle_right_plus_sigma)
            self.features[idx][self.settings.middlehighplus] = self.gaussianFunction(self.x_range, mean + (minus_mean *(4/20)), middle_right_plus_sigma, cr_middlehighplus, cr_high, mean  + (minus_mean * (1/20)), mean  + (minus_mean * (10/20)), middle_right_sigma, right_sigma)
            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean + (minus_mean * (10/20)), right_sigma, cr_high, -1, mean + (minus_mean * (4/20)), 1.01, middle_right_plus_sigma, -1)  
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean + (minus_mean * (10/20)), right_sigma)
            
        elif self.settings.gausses == 11:
            if (minus_mean * (1/35) >= (mean * (1/35))):
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/35)))
            else:
                cr_middle, center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/35)))
            
            cr_middlelowplus, middle_left_plus_sigma = self.calculateSigma(mean - (mean * (1/35)), mean, center_sigma)
            cr_middlelow, middle_left_sigma = self.calculateSigma(mean - (mean * (4/35)), mean - (mean * (1/35)), middle_left_plus_sigma)
            cr_middlelowminus, middle_left_minus_sigma = self.calculateSigma(mean - (mean * (10/35)), mean - (mean * (4/35)), middle_left_sigma)
            cr_low, left_sigma = self.calculateSigma(mean - (mean * (20/35)), mean - (mean * (10/35)), middle_left_minus_sigma)
            
            _, middle_right_sigma = self.calculateSigma(mean + (minus_mean * (1/35)), mean, center_sigma)
            cr_middlehighplus, middle_right_plus_sigma = self.calculateSigma(mean + (minus_mean * (4/35)), mean + (minus_mean * (1/35)), middle_right_sigma)
            cr_high, right_sigma = self.calculateSigma(mean + (minus_mean * (10/35)), mean + (minus_mean * (4/35)), middle_right_plus_sigma)
            cr_middlehighminus, middle_right_minus_sigma = self.calculateSigma(mean + (minus_mean * (20/35)), mean + (minus_mean * (10/35)), right_sigma)

            self.features[idx][self.settings.verylow] = self.gaussLeft(self.x_range, mean - (mean * (20/35)), left_sigma)
            self.features[idx][self.settings.low] = self.gaussianFunction(self.x_range, mean - (mean * (20/35)), left_sigma, -1, cr_low, 0, mean - (mean * (10/35)), -1, middle_left_minus_sigma)
            self.features[idx][self.settings.middlelowminus] = self.gaussianFunction(self.x_range, mean - (mean * (10/35)), middle_left_minus_sigma, cr_low, cr_middlelowminus, mean - (mean * (20/35)), mean - (mean * (4/35)), left_sigma, middle_left_sigma)
            self.features[idx][self.settings.middlelow] = self.gaussianFunction(self.x_range, mean - (mean * (4/35)), middle_left_sigma, cr_middlelowminus, cr_middlelow, mean - (mean * (10/35)), mean - (mean * (1/35)), middle_left_minus_sigma, middle_left_plus_sigma)
            self.features[idx][self.settings.middlelowplus] = self.gaussianFunction(self.x_range, mean - (mean * (1/35)), middle_left_plus_sigma, cr_middlelow, cr_middlelowplus, mean - (mean * (4/35)), mean, middle_left_sigma, center_sigma)

            self.features[idx][self.settings.middle] = self.gaussianFunction(self.x_range, mean, center_sigma, cr_middlelow, cr_middle, mean - (mean * (1/35)), mean + (minus_mean * (1/35)), middle_left_sigma, middle_right_sigma)

            self.features[idx][self.settings.middlehigh] = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/35)), middle_right_sigma, cr_middle, cr_high, mean, mean + (minus_mean * (4/35)), center_sigma, middle_right_plus_sigma)
            self.features[idx][self.settings.middlehighplus] = self.gaussianFunction(self.x_range, mean + (minus_mean * (4/35)), middle_right_plus_sigma, cr_middlehighplus, cr_high, mean  + (minus_mean * (1/35)), mean  + (minus_mean * (10/35)), middle_right_sigma, right_sigma)
            self.features[idx][self.settings.high] = self.gaussianFunction(self.x_range, mean + (minus_mean * (10/35)), right_sigma, cr_high, cr_middlehighminus, mean + (minus_mean * (4/35)), mean + (minus_mean * (20/35)), middle_right_plus_sigma, middle_right_minus_sigma)  
            self.features[idx][self.settings.middlehighminus] = self.gaussianFunction(self.x_range, mean + (minus_mean * (20/35)), middle_right_minus_sigma, cr_middlehighminus, -1, mean  + (minus_mean * (10/35)), 1.01, right_sigma, -1)
            self.features[idx][self.settings.veryhigh] = self.gaussRight(self.x_range, mean + (minus_mean * (20/35)), middle_right_minus_sigma)
            
        self.fuzzify_parameters.append(mean)
        self.fuzzify_parameters.append(sigma)
                              

        for x in values:
            verylow_value = low_value = middlelowminus_value = middlelow_value = middlelowplus_value = middle_value = middlehighminus_value = middlehigh_value = middlehighplus_value = high_value = veryhigh_value = 0

            if self.settings.gausses == 3:
                verylow_value = self.leftGaussianValue(x, mean, sigma)
                middle_value = self.gaussianValue(x, mean, sigma, -1, -1, 0, 1.01, -1, -1)
                veryhigh_value = self.rightGaussianValue(x, mean, sigma)
                
            if self.settings.gausses == 5:
                middle_value = self.gaussianValue(x, mean, center_sigma, cr_low, cr_middle, mean - (mean * (1/4)), mean + (minus_mean * (1/4)), left_sigma, right_sigma)
                verylow_value = self.leftGaussianValue(x, mean - (mean * (1/4)), left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (1/4)), left_sigma, -1, cr_low, 0, mean, -1, center_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (1/4)), right_sigma, cr_middle, -1, mean, 1.01, center_sigma, -1)  
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (1/4)), right_sigma)

            elif self.settings.gausses == 7:
                middle_value = self.gaussianValue(x, mean, center_sigma, cr_middlelow, cr_middle, mean - (mean * (1/10)), mean + (minus_mean * (1/10)), middle_left_sigma, middle_right_sigma)
                verylow_value = self.leftGaussianValue(x, mean - (mean * (4/10)), left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (4/10)), left_sigma, -1, cr_low, 0, mean - (mean * (1/10)), -1, middle_left_sigma)
                middlelow_value = self.gaussianValue(x, mean - (mean * (1/10)), middle_left_sigma, cr_low, cr_middlelow, mean - (mean * (4/10)), mean, left_sigma, center_sigma)
                middlehigh_value = self.gaussianValue(x, mean + (minus_mean * (1/10)), middle_right_sigma, cr_middle, cr_high, mean, mean + (minus_mean * (4/10)), center_sigma, right_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (4/10)), right_sigma, cr_high, -1, mean + (minus_mean * (1/10)), 1.01, middle_right_sigma, -1)  
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (4/10)), right_sigma)
                
            elif self.settings.gausses == 9:
                verylow_value = self.leftGaussianValue(x, mean - (mean * (10/20)), left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (10/20)), left_sigma, -1, cr_low, 0, mean - (mean * (4/20)), -1, middle_left_minus_sigma)
                middlelowminus_value = self.gaussianValue(x, mean - (mean * (4/20)), middle_left_minus_sigma, cr_low, cr_middlelowminus, mean - (mean * (10/20)), mean - (mean * (1/20)), left_sigma, middle_left_sigma)
                middlelow_value = self.gaussianValue(x, mean - (mean * (1/20)), middle_left_sigma, cr_middlelowminus, cr_middlelow, mean - (mean * (4/20)), mean, middle_left_minus_sigma, center_sigma)

                middle_value = self.gaussianValue(x, mean, center_sigma, cr_middlelow, cr_middle, mean - (mean * (1/20)), mean + (minus_mean * (1/20)), middle_left_sigma, middle_right_sigma)

                middlehigh_value = self.gaussianValue(x, mean + (minus_mean * (1/20)), middle_right_sigma, cr_middle, cr_high, mean, mean + (minus_mean * (4/20)), center_sigma, middle_right_plus_sigma)
                middlehighplus_value = self.gaussianValue(x, mean + (minus_mean *(4/20)), middle_right_plus_sigma, cr_middlehighplus, cr_high, mean  + (minus_mean * (1/20)), mean  + (minus_mean * (10/20)), middle_right_sigma, right_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (10/20)), right_sigma, cr_high, -1, mean + (minus_mean * (4/20)), 1.01, middle_right_plus_sigma, -1)  
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (10/20)), right_sigma)
                
            elif self.settings.gausses == 11:
                verylow_value = self.leftGaussianValue(x, mean - (mean * (20/35)), left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (20/35)), left_sigma, -1, cr_low, 0, mean - (mean * (10/35)), -1, middle_left_minus_sigma)
                middlelowminus_value = self.gaussianValue(x, mean - (mean * (10/35)), middle_left_minus_sigma, cr_low, cr_middlelowminus, mean - (mean * (20/35)), mean - (mean * (4/35)), left_sigma, middle_left_sigma)
                middlelow_value = self.gaussianValue(x, mean - (mean * (4/35)), middle_left_sigma, cr_middlelowminus, cr_middlelow, mean - (mean * (10/35)), mean - (mean * (1/35)), middle_left_minus_sigma, middle_left_plus_sigma)
                middlelowplus_value = self.gaussianValue(x, mean - (mean * (1/35)), middle_left_plus_sigma, cr_middlelow, cr_middlelowplus, mean - (mean * (4/35)), mean, middle_left_sigma, center_sigma)

                middle_value = self.gaussianValue(x, mean, center_sigma, cr_middlelow, cr_middle, mean - (mean * (1/35)), mean + (minus_mean * (1/35)), middle_left_sigma, middle_right_sigma)

                middlehigh_value = self.gaussianValue(x, mean + (minus_mean * (1/35)), middle_right_sigma, cr_middle, cr_high, mean, mean + (minus_mean * (4/35)), center_sigma, middle_right_plus_sigma)
                middlehighplus_value = self.gaussianValue(x, mean + (minus_mean * (4/35)), middle_right_plus_sigma, cr_middlehighplus, cr_high, mean  + (minus_mean * (1/35)), mean  + (minus_mean * (10/35)), middle_right_sigma, right_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (10/35)), right_sigma, cr_high, cr_middlehighminus, mean + (minus_mean * (4/35)), mean + (minus_mean * (20/35)), middle_right_plus_sigma, middle_right_minus_sigma)  
                middlehighminus_value = self.gaussianValue(x, mean + (minus_mean * (20/35)), middle_right_minus_sigma, cr_middlehighminus, -1, mean  + (minus_mean * (10/35)), 1.01, right_sigma, -1)
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (20/35)), middle_right_minus_sigma)

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
            elif max_value == veryhigh_value:
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
