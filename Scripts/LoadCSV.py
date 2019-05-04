import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.CSVExtractor import CSVExtractor as CSVExtractor

class LoadCSV(object):

    def saveVariables(self, variables):
        d_results = [variables["class_1"], variables["class_2"]]
        if not os.path.exists(variables["backup_folder"]):
            os.makedirs(variables["backup_folder"])
        if not os.path.exists(variables["results_folder"]):
            os.makedirs(variables["results_folder"])   

        return variables, d_results

    def prepareData(self, variables, d_results):
        if variables['load_previous_data']:
            fuzzifier = pickle.load(open(variables["backup_folder"] + "fuzzifier.p", "rb"))
            features_df = pickle.load(open(variables["backup_folder"] + "features_df.p", "rb"))
        else:
            fuzzifier = Fuzzifier(variables, d_results)
            csvExtractor = CSVExtractor(variables, fuzzifier)
            features_df = csvExtractor.worker()      
            pickle.dump(fuzzifier, open(variables["backup_folder"] + "fuzzifier.p", "wb"))
            pickle.dump(features_df, open(variables["backup_folder"] + "features_df.p", "wb"))
        
        return features_df

    def splitDataForTrainingTest(self, features_df, variables, test_size = 0.2):
        train_features_df, test_features_df = train_test_split(features_df, test_size=test_size, stratify=features_df.Decision)
        pickle.dump(train_features_df, open(variables["backup_folder"] + "train_features_df.p", "wb"))
        pickle.dump(test_features_df, open(variables["backup_folder"] + "test_features_df.p", "wb"))

    def worker(self, variables):
        variables, d_results = self.saveVariables(variables)
        features_df = self.prepareData(variables, d_results)
        self.splitDataForTrainingTest(features_df, variables)