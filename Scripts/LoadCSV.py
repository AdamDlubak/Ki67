import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.CSVExtractor import CSVExtractor as CSVExtractor

class LoadCSV(object):

    def __init__(self):
        self.variables = []
        
    def saveVariables(self):
        d_results = [self.variables["class_2"], self.variables["class_1"]]
        if not os.path.exists(self.variables["backup_folder"]):
            os.makedirs(self.variables["backup_folder"])
        if not os.path.exists(self.variables["results_folder"]):
            os.makedirs(self.variables["results_folder"])   

        return d_results

    def prepareData(self, d_results):
        fuzzifier = Fuzzifier(self.variables, d_results)
        if self.variables['load_previous_data']:
            features_df = pickle.load(open(self.variables["backup_folder"] + "features_df.p", "rb"))
        else:
            csvExtractor = CSVExtractor(self.variables)
            features_df = csvExtractor.worker(fuzzifier)   
    
        pickle.dump(fuzzifier, open(self.variables["backup_folder"] + "fuzzifier.p", "wb"))
        pickle.dump(features_df, open(self.variables["backup_folder"] + "features_df.p", "wb"))
        
        return features_df

    def splitDataForTrainingTest(self, features_df, test_size = 0.2):
        train_features_df, test_features_df = train_test_split(features_df, test_size=test_size, stratify=features_df.Decision)
        pickle.dump(train_features_df, open(self.variables["backup_folder"] + "train_features_df.p", "wb"))
        pickle.dump(test_features_df, open(self.variables["backup_folder"] + "test_features_df.p", "wb"))


        class_1_features_occurence = features_df.loc[features_df.Decision == self.variables["class_1"]]["Decision"].count()
        class_2_features_occurence = features_df.loc[features_df.Decision == self.variables["class_2"]]["Decision"].count()

        class_1_train_occurence = train_features_df.loc[train_features_df.Decision == self.variables["class_1"]]["Decision"].count()
        class_2_train_occurence = train_features_df.loc[train_features_df.Decision == self.variables["class_2"]]["Decision"].count()

        class_1_test_occurence = test_features_df.loc[test_features_df.Decision == self.variables["class_1"]]["Decision"].count()
        class_2_test_occurence = test_features_df.loc[test_features_df.Decision == self.variables["class_2"]]["Decision"].count()

        features_result = "{} ({}/{})".format(features_df.shape[0], class_1_features_occurence, class_2_features_occurence)
        train_result = "{} ({}/{})".format(train_features_df.shape[0], class_1_train_occurence, class_2_train_occurence)
        test_result = "{} ({}/{})".format(test_features_df.shape[0], class_1_test_occurence, class_2_test_occurence)
        return features_result, train_result, test_result, train_features_df.shape[0]

    def worker(self, variables):
        self.variables= variables

        d_results = self.saveVariables()
        features_df = self.prepareData(d_results)
        samples_stats, train_stats, test_stats, train_samples = self.splitDataForTrainingTest(features_df)

        return samples_stats, train_stats, test_stats, train_samples