import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.ImageReader import ImageReader as ImageReader
from Class.PixelFeatureExtractor import PixelFeatureExtractor as PixelFeatureExtractor
class LoadImage(object):

    def saveVariables(self, settings):
        d_results = [settings.class_1, settings.class_2]
        if not os.path.exists(settings.backup_folder):
            os.makedirs(settings.backup_folder)
        if not os.path.exists(settings.results_folder):
            os.makedirs(settings.results_folder)   

        return settings, d_results

    def prepareData(self, settings, d_results):
        fuzzifier = Fuzzifier(settings, d_results)
        if settings.load_previous_data:
                features_df = pickle.load(open(settings.backup_folder + "features_df.p", "rb"))
        else:
            imageReader = ImageReader("Data/Ki67-Example/test")
            imageReader.loadImages(settings)

            pixelFeatureExtractor = PixelFeatureExtractor(fuzzifier, settings)
            features_df = pixelFeatureExtractor.worker(imageReader)
            
        pickle.dump(fuzzifier, open(settings.backup_folder + "fuzzifier.p", "wb"))
        pickle.dump(features_df, open(settings.backup_folder + "features_df.p", "wb"))
        
        return features_df


    def worker(self, settings):
        settings, d_results = self.saveVariables(settings)
        features_df = self.prepareData(settings, d_results)
        # self.splitDataForTrainingTest(features_df, settings)