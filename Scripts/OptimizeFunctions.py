import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from Functions.RulesSetter import RulesSetter as RulesSetter

class OptimizeFunctions(object):

    def setDecisions(self, params, sorted_decision, variables):
        decision_table = sorted_decision.copy()
        
        if params[0] == -1:
            params[0] = np.mean(decision_table['Predicted Value'] )
        for index in decision_table.index:
            if decision_table.loc[index]['Predicted Value'] >= params[0]:
                decision_table.loc[index, 'Decision Fuzzy'] = variables['d_low']
            elif decision_table.loc[index]['Predicted Value'] < params[0]:
                decision_table.loc[index, 'Decision Fuzzy'] = variables['d_high']
        
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

    def makeJob(self, params, sorted_decision, variables):
        result_table = self.setDecisions(params, sorted_decision, variables)
        accuracy = self.getAccuracy(result_table)
        accuracy_reverse = 1 - accuracy
        return accuracy_reverse

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