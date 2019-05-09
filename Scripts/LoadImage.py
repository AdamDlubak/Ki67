import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.ImageReader import ImageReader as ImageReader
from Class.PixelFeatureExtractor import PixelFeatureExtractor as PixelFeatureExtractor
class LoadImage(object):
    
    def __init__(self):
        self.variables = []

    def saveVariables(self):
        d_results = [self.variables["class_1"], self.variables["class_2"]]
        if not os.path.exists(self.variables["backup_folder"]):
            os.makedirs(self.variables["backup_folder"])
        if not os.path.exists(self.variables["results_folder"]):
            os.makedirs(self.variables["results_folder"])   

        return d_results

    def prepareData(self, d_results, test_mode = False):
        fuzzifier = Fuzzifier(self.variables, d_results)
        if self.variables['load_previous_data']:
                features_df = pickle.load(open(self.variables["backup_folder"] + "features_df.p", "rb"))
        else:
            imageReader = ImageReader(self.variables["data_folder"])
            imageReader.loadImages(self.variables)

            pixelFeatureExtractor = PixelFeatureExtractor(self.variables, fuzzifier)
            features_df = pixelFeatureExtractor.worker(imageReader, test_mode)
            
        pickle.dump(fuzzifier, open(self.variables["backup_folder"] + "fuzzifier.p", "wb"))
        pickle.dump(features_df, open(self.variables["backup_folder"] + "features_df.p", "wb"))
        
        return features_df, fuzzifier
    
    def setDecision(self, features_df, searched_class):
        return features_df.apply(self.makeDecision, self.variables = self.variables, searched_class = searched_class, axis=1)
        
    def splitDataForTrainingTest(self, features_df, fuzzifier, test_size = 0.2):
        train_features_df, test_features_df = train_test_split(features_df, test_size=test_size, stratify=features_df.Decision)

        train_features_df = train_features_df.append({
            fuzzifier.features[0].label: 0, 
            fuzzifier.features[1].label: 0, 
            fuzzifier.features[2].label: 0, 
            fuzzifier.features[3].label: 0, 
            fuzzifier.features[4].label: 0, 
            fuzzifier.features[5].label: 0, 
            fuzzifier.features[6].label: 0, 
            "Image": "Black",
            "Decision": self.variables["class_other"],
            "Predicted Value": ""
        }, ignore_index=True)

        test_features_df = test_features_df.append({
            fuzzifier.features[0].label: 0, 
            fuzzifier.features[1].label: 0, 
            fuzzifier.features[2].label: 0, 
            fuzzifier.features[3].label: 0, 
            fuzzifier.features[4].label: 0, 
            fuzzifier.features[5].label: 0, 
            fuzzifier.features[6].label: 0, 
            "Image": "Black",
            "Decision": self.variables["class_other"],
            "Predicted Value": ""
        }, ignore_index=True)

        pickle.dump(train_features_df, open(self.variables["backup_folder"] + "train_features_df.p", "wb"))
        pickle.dump(test_features_df, open(self.variables["backup_folder"] + "test_features_df.p", "wb"))

        return features_df.shape[0], train_features_df.shape[0], test_features_df.shape[0]

    def makeDecision(self, row, searched_class):
        if row["Image"] == searched_class:
            row.Decision = searched_class
        else:
            row.Decision = self.variables["class_other"]
        return row

    def worker(self, variables, searched_class, test_mode = False):
        self.variables = variables

        d_results = self.saveVariables()
        features_df, fuzzifier = self.prepareData(d_results, test_mode)
        features_df = self.setDecision(features_df, searched_class)
        samples, train_samples, test_samples =  = self.splitDataForTrainingTest(features_df, fuzzifier)
        
        return samples, train_samples, test_samples