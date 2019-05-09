import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.ImageReader import ImageReader as ImageReader
from Class.PixelFeatureExtractor import PixelFeatureExtractor as PixelFeatureExtractor
class LoadImage(object):

    def saveVariables(self, variables):
        d_results = [variables["class_1"], variables["class_2"]]
        if not os.path.exists(variables["backup_folder"]):
            os.makedirs(variables["backup_folder"])
        if not os.path.exists(variables["results_folder"]):
            os.makedirs(variables["results_folder"])   

        return variables, d_results

    def prepareData(self, variables, d_results):
        fuzzifier = Fuzzifier(variables, d_results)
        if variables['load_previous_data']:
                features_df = pickle.load(open(variables["backup_folder"] + "features_df.p", "rb"))
        else:
            imageReader = ImageReader("Data/Ki67-Example/test")
            imageReader.loadImages(variables)

            pixelFeatureExtractor = PixelFeatureExtractor(fuzzifier, variables)
            features_df = pixelFeatureExtractor.worker(imageReader)
            
        pickle.dump(fuzzifier, open(variables["backup_folder"] + "fuzzifier.p", "wb"))
        pickle.dump(features_df, open(variables["backup_folder"] + "features_df.p", "wb"))
        
        return features_df


    def worker(self, variables):
        variables, d_results = self.saveVariables(variables)
        features_df = self.prepareData(variables, d_results)
        # self.splitDataForTrainingTest(features_df, variables)