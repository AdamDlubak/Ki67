import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from Class.RulesSetter import RulesSetter as RulesSetter

class FuzzyHelper(object):

    def __init__(self, variables):
        self.variables = variables
        
    def setDecisions(self, row, threshold):
        if row['Predicted Value'] > threshold:
            row['Decision Fuzzy'] = self.variables["class_1"]
        else:
            row['Decision Fuzzy'] = self.variables["class_2"]
        return row

    def makePrediction(self, row, classing, rules_feature_names, decision):
        input_values = {}
        for x in rules_feature_names:
            input_values[x] = row[x]

        classing.inputs(input_values)
        classing.compute()

        if self.variables["show_results"]:
            decision.view(sim=classing)
            print(classing.output['Decision'])

        row['Predicted Value'] = classing.output['Decision']
        
        return row

    def defineClassOne(self, x, center_point, width):
        y = np.zeros(len(x))
        idx = x <= center_point
        y[idx] = fuzz.sigmf(x[idx], center_point, width)
        return y

    def defineClassTwo(self, x, center_point, width):
        y = np.zeros(len(x))
        idx = x >= center_point
        y[idx] = fuzz.sigmf(x[idx], center_point, width)
        return y

    def prepareRules(self, save, x_range, center_point, width, rules_extractor, rule_antecedents, d_results, decision, df = None):
        
        decision[self.variables["class_1"]] = self.defineClassOne(x_range, center_point, -width)
        decision[self.variables["class_2"]] = self.defineClassTwo(x_range, center_point, width)     
        
        rules = rules_extractor.generateRules(rule_antecedents, d_results, decision)
        class_ctrl = ctrl.ControlSystem(rules)
        gen = class_ctrl.fuzzy_variables
        classing = ctrl.ControlSystemSimulation(class_ctrl)
        rules_feature_names = []
        for x in gen:
            if str(x).startswith('Antecedent'):
                rules_feature_names.append(str(x).split(': ')[1])
        if save:
            pickle.dump(classing, open(self.variables["backup_folder"] + "classing.p", "wb"))
            pickle.dump(rules_feature_names, open(self.variables["backup_folder"] + "rules_feature_names.p", "wb"))    
            pickle.dump(rules, open(self.variables["backup_folder"] + "rules.p", "wb"))
            pickle.dump(decision, open(self.variables["backup_folder"] + "decision.p", "wb"))
            _, df_to_save = self.sFunctionsWorker(df, x_range, center_point, width, rules_extractor, rule_antecedents, d_results, decision)
            pickle.dump(df_to_save, open(self.variables["backup_folder"] + "df.p", "wb"))
            
        return decision, classing, rules_feature_names

    def prepareDataFrame(self, oryginal_df, rules_feature_names):
        df = oryginal_df[rules_feature_names].copy()
        df['Decision'] = oryginal_df.Decision
        df['Decision Fuzzy'] = ""
        df['Predicted Value'] = ""
        return df

    def getAccuracy(self, df):
        accuracy = accuracy_score(df['Decision'], df['Decision Fuzzy'])
        return accuracy

    def getScores(self, df, show = True):
        accuracy = accuracy_score(df['Decision'], df['Decision Fuzzy'])
        precision, recall, fscore, support = score(df['Decision'], df['Decision Fuzzy'])

        if show:
            print("Accuracy: {}".format(accuracy))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F-Score: {}".format(fscore))
            print("Support: {}".format(support))

        return accuracy, precision, recall, fscore, support

    def sFunctionsWorker(self, df, x_range, center_point, width, rules_extractor, rule_antecedents, d_results, decision):
        decision, classing, rules_feature_names = self.prepareRules(False, x_range, center_point, width, rules_extractor, rule_antecedents, d_results, decision)
        df = self.prepareDataFrame(df, rules_feature_names)
        df = df.apply(self.makePrediction, classing = classing, rules_feature_names = rules_feature_names, decision = decision, axis=1)
        accuracy, df = self.thresholdOptValue(center_point, df)
        return accuracy, df

    def sFunctionsOptBrute(self, center_point, *params):
        width, df, self.variables, x_range, rules_extractor, rule_antecedents, d_results, decision = params
        accuracy, _ = self.sFunctionsWorker(df, x_range, center_point, width, rules_extractor, rule_antecedents, d_results, decision)
        return 1 - accuracy

    def sFunctionsValue(self, center_point, width, df, variables, x_range, rules_extractor, rule_antecedents, d_results, decision):
        accuracy, df = self.sFunctionsWorker(df, x_range, center_point, width, rules_extractor, rule_antecedents, d_results, decision)
        return accuracy, df

    def thresholdWorker(self, threshold, df):
        df = df.apply(self.setDecisions, threshold = threshold, axis=1)
        accuracy = self.getAccuracy(df)
        return accuracy, df

    def thresholdOptBrute(self, threshold, *params):
        df, _ = params
        accuracy, _ = self.thresholdWorker(threshold, df)
        return 1 - accuracy 

    def thresholdOptValue(self, threshold, df):
        accuracy, df = self.thresholdWorker(threshold, df)
        return accuracy, df   

    def setRules(self, rules, features_table, rules_feature_names, classing, decision):
        rulesSetter = RulesSetter()
        sorted_decision, rules_feature_names, classing = rulesSetter.setRules(rules, features_table) 
        sorted_decision = sorted_decision.apply(self.makePrediction, classing = classing, rules_feature_names = rules_feature_names, decision = decision, axis=1)
        return sorted_decision

    def printAllDataframe(self, df):
        pd.set_option('display.max_rows', len(df))
        display(df)
        pd.reset_option('display.max_rows')

    def plotHistogram(self, df, bins = 100):
        df['Predicted Value'].hist(bins=bins)
 
    def saveResults(self, results_path, series):
        path = results_path
        columns = ["Type", "Accuracy", "Precision A", "Precision B", "Recall A", "Recall B", "F-Score A", "F-Score B", "Support A", "Support B", "S-Functions Center", "S-Functions Width", "Threshold", "Time (s)"]
        df = pd.DataFrame(columns=columns)
        s = pd.Series(series, index=columns)
        df = df.append(s, ignore_index = True)
        df.to_csv(path, index = False, header=(not os.path.exists(path)), mode="a")