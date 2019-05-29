import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.ImageReader import ImageReader as ImageReader
from Class.PixelFeatureExtractor import PixelFeatureExtractor as PixelFeatureExtractor
class LoadImage(object):
    
    def __init__(self):
        self.settings = []

    def saveVariables(self):
        d_results = [self.settings.class_2, self.settings.class_1]
        if not os.path.exists(self.settings.backup_folder):
            os.makedirs(self.settings.backup_folder)
        if not os.path.exists(self.settings.results_folder):
            os.makedirs(self.settings.results_folder)   

        return d_results

    def prepareData(self, d_results, test_mode = False):
        fuzzifier = Fuzzifier(self.settings, d_results)
        if self.settings.load_previous_data:
                features_df = pickle.load(open(self.settings.backup_folder + "features_df.p", "rb"))
        else:
            imageReader = ImageReader(self.settings.data_folder)
            imageReader.loadImages(self.settings)

            pixelFeatureExtractor = PixelFeatureExtractor(self.settings, fuzzifier)
            features_df = pixelFeatureExtractor.worker(imageReader, test_mode)
            
        pickle.dump(fuzzifier, open(self.settings.backup_folder + "fuzzifier.p", "wb"))
        pickle.dump(features_df, open(self.settings.backup_folder + "features_df.p", "wb"))
        return features_df, fuzzifier
    
    def setDecision(self, features_df):
        return features_df.apply(self.makeDecision, axis=1)
        
    def prepareTest(self, features_df):
        features_result = "{} (--/--)".format(features_df.shape[0])
        train_result = "-- (--/--)"
        test_result = features_result
        return features_result, train_result, test_result, features_df.shape[0], features_df
  
    def prepareTraining(self, features_df, fuzzifier):
        features_df = features_df.append({
            fuzzifier.features[0].label: 0, 
            fuzzifier.features[1].label: 0, 
            fuzzifier.features[2].label: 0, 
            fuzzifier.features[3].label: -0.364297, 
            fuzzifier.features[4].label: 0.106373, 
            fuzzifier.features[5].label: -0.265494, 
            "Image": "Black",
            "Decision": self.settings.class_2,
            "Predicted Value": ""
        }, ignore_index=True)

        class_1_features_occurence = features_df.loc[features_df.Decision == self.settings.class_1"]]["Decision.count()
        class_2_features_occurence = features_df.loc[features_df.Decision == self.settings.class_2"]]["Decision.count()

        features_result = "{} ({}/{})".format(features_df.shape[0], class_1_features_occurence, class_2_features_occurence)
        train_result = features_result
        test_result = "-- (--/--)"
        return features_result, train_result, test_result, features_df.shape[0], features_df

    def makeDecision(self, row):
        if row["Image"] == self.settings.class_1:
            row.Decision = self.settings.class_1
        else:
            row.Decision = self.settings.class_2
        return row

    def normalizeFeatures(self, df):

        df["F0"] = (df["F0"] - 0) / (255 - 0)
        df["F1"] = (df["F1"] - 0) / (255 - 0)
        df["F2"] = (df["F2"] - 0) / (255 - 0)
        df["F3"] = (abs(df["F3"]) - 0.15) / (0.70 - 0.15)
        df["F4"] = (abs(df["F4"]) - 0.00) / (0.25 - 0.00)
        df["F5"] = (abs(df["F5"]) - 0.10) / (0.60 - 0.10)

        return df
         
    def worker(self, settings, test_mode = False):
        self.settings = settings

        d_results = self.saveVariables()
        features_df, fuzzifier = self.prepareData(d_results, test_mode)
        features_df = self.setDecision(features_df)

        if test_mode:
            samples_stats, train_stats, test_stats, train_samples, test_features_df = self.prepareTest(features_df)
            test_features_df = self.normalizeFeatures(test_features_df)
            pickle.dump(test_features_df, open(self.settings.backup_folder + "test_features_df.p", "wb"))

        else:
            samples_stats, train_stats, test_stats, train_samples, train_features_df = self.prepareTraining(features_df, fuzzifier)
            train_features_df = self.normalizeFeatures(train_features_df)
            pickle.dump(train_features_df, open(self.settings.backup_folder + "train_features_df.p", "wb"))
        
        return samples_stats, train_stats, test_stats, train_samples