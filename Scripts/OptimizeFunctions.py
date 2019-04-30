import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from Functions.RulesSetter import RulesSetter as RulesSetter

class OptimizeFunctions(object):

    def setDecisions(self, params, sorted_decision, variables):
        decision_table = sorted_decision.copy()
        
        if params[0] == -1:
            params[0] = np.mean(decision_table['Predicted Value'] )
        for index in decision_table.index:
            if sorted_decision.loc[index]['Predicted Value'] > params[0]:
                sorted_decision.loc[index, 'Decision Fuzzy'] = variables["class_1"]
            else:
                sorted_decision.loc[index, 'Decision Fuzzy'] = variables["class_2"]
        
        return decision_table

    def getAccuracy(self, df):
        accuracy = accuracy_score(df['Decision'], df['Decision Fuzzy'])
        return accuracy

    def getScores(self, df):
        accuracy = accuracy_score(df['Decision'], df['Decision Fuzzy'])
        precision, recall, fscore, support = score(df['Decision'], df['Decision Fuzzy'])

        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F-Score: {}".format(fscore))
        print("Support: {}".format(support))

        return accuracy, precision, recall, fscore, support

    def makeJob(self, center_point, width, normalized_features_table, variables, x_range, rules_extractor, rule_antecedents, d_results, decision):
        decision[variables["class_1"]] = fuzz.sigmf(x_range, center_point, -width)
        decision[variables["class_2"]] = fuzz.sigmf(x_range, center_point, width)
        decision.view()
        rules = rules_extractor.generateRules(rule_antecedents, d_results, decision)
        class_ctrl = ctrl.ControlSystem(rules)
        gen = class_ctrl.fuzzy_variables
        classing = ctrl.ControlSystemSimulation(class_ctrl)
        rules_feature_names = []
        for x in gen:
            if str(x).startswith('Antecedent'):
                rules_feature_names.append(str(x).split(': ')[1])
        test_features_table = normalized_features_table[rules_feature_names].copy()
        test_features_table['Decision'] = normalized_features_table.Decision
        test_features_table['Decision Fuzzy'] = ""

        for index, row in test_features_table.iterrows():
            new_dict = {}
            for x in rules_feature_names:
                new_dict[x] = row[x]

            classing.inputs(new_dict)
            classing.compute()
            test_features_table.loc[index, 'Predicted Value'] = classing.output['Decision']
            if classing.output['Decision'] > center_point:
                test_features_table.loc[index, 'Decision Fuzzy'] = variables["class_1"]
            else:
                test_features_table.loc[index, 'Decision Fuzzy'] = variables["class_2"]
        
        accuracy = self.getAccuracy(test_features_table)
        return accuracy, test_features_table

    def optFunc(self, x, sorted_decision, variables):
        n_particles = x.shape[0]
        j = [self.makeJob(x[i], sorted_decision, variables) for i in range(n_particles)]
        return np.array(j)

    def setRules(self, rules, features_table):
        rulesSetter = RulesSetter()
        sorted_decision = rulesSetter.setRules(rules, features_table) 
        return sorted_decision

    def printAllDataframe(self, df):
        pd.set_option('display.max_rows', len(df))
        display(df)
        pd.reset_option('display.max_rows')

    def plotHistogram(self, df, bins = 100):
        hist = df['Predicted Value'].hist(bins=bins)
 
    def saveResults(self, results_path, series, filename = "Results.csv"):
        path = results_path + filename
        columns = ["Type", "Accuracy", "Precision A", "Precision B", "Recall A", "Recall B", "F-Score A", "F-Score B", "Support A", "Support B", "Threshold", "Time (s)"]
        df = pd.DataFrame(columns=columns)
        s = pd.Series(series, index=columns)
        df = df.append(s, ignore_index = True)
        df.to_csv(path, index = False, header=(not os.path.exists(path)), mode="a")