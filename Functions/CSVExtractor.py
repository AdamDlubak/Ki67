import pandas as pd

class CSVExtractor(object):
    
    def __init__(self, variables, fuzzifier):
        self.features_table = []
        self.path = variables['data_folder']
        self.variables = variables
        self.fuzzifier = fuzzifier

    def showResults(self, table):
        if self.variables["show_results"]:
            display(table)

    def extractFeatures(self):
        self.features_table = pd.read_csv(self.path, sep=";")
        self.features_table["Image"] = ""

        self.showResults(self.features_table)

        return self.features_table

    def normalizeFeatures(self):
        for x in self.fuzzifier.features:
            self.features_table[x.label] = (
                self.features_table[x.label] - self.features_table[x.label].min()) / (
                    self.features_table[x.label].max() - self.features_table[x.label].min())

        self.showResults(self.features_table)

    def getFeaturesTable(self):
        return self.features_table

    def worker(self):
        all_features_table = self.extractFeatures()
        self.normalizeFeatures()
        return self.getFeaturesTable(), all_features_table
