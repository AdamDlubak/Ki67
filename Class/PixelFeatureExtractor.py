import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

class PixelFeatureExtractor(object):
    def __init__(self, variables, fuzzifier):
        self.features_table = []
        self.fuzzifier = fuzzifier
        self.variables = variables
        
    def showResults(self, table):
        if self.variables["show_results"]:
           display(table)


    def getFeatureTableFromImages(self, image_reader, test_mode = False):
        self.features_table = pd.DataFrame()
        for idx, image in enumerate(image_reader.images):

            r_image, g_image, b_image = self.splitIntoRgbChannels(image)
            h_image, s_image, v_image = self.splitIntoHsvChannels(image)
            intensity_mean = (r_image + g_image + b_image) / 3

            data_columns = {
                self.fuzzifier.features[0].label: r_image.ravel(),
                self.fuzzifier.features[1].label: g_image.ravel(),
                self.fuzzifier.features[2].label: b_image.ravel(),
                self.fuzzifier.features[3].label: h_image.ravel(),
                self.fuzzifier.features[4].label: s_image.ravel(),
                self.fuzzifier.features[5].label: v_image.ravel(),
                self.fuzzifier.features[6].label: intensity_mean.ravel()
            }

            tmp_features_table = pd.DataFrame(data = data_columns)
            tmp_features_table["Image"] = image_reader.image_names[idx]
            tmp_features_table["Decision"] = image_reader.image_decisions[idx]
            tmp_features_table["Predicted Value"] = ""
        
            if test_mode == False:
        
                tmp_features_table = tmp_features_table[(tmp_features_table[[
                    self.fuzzifier.features[0].label,   
                    self.fuzzifier.features[1].label,
                    self.fuzzifier.features[2].label,
                    self.fuzzifier.features[3].label,
                    self.fuzzifier.features[4].label,
                    self.fuzzifier.features[5].label,
                    self.fuzzifier.features[6].label
                ]] != 0).all(axis=1)]

            if idx == 0:
                self.features_table = tmp_features_table
            else:
                frames = [self.features_table, tmp_features_table]
                self.features_table = pd.concat(frames).reset_index(drop=True)

        self.showResults(self.features_table)

    def splitIntoRgbChannels(self, image):
        b, g, r = cv2.split(image)
        return r, g, b

    def splitIntoHsvChannels(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        return h, s, v

    def normalizeFeatures(self):
        for x in self.fuzzifier.features:
            self.features_table[x.label] = (
                self.features_table[x.label] - self.features_table[x.label].min()) / (
                    self.features_table[x.label].max() - self.features_table[x.label].min())

        self.showResults(self.features_table)

    def getFeaturesTable(self):
        return self.features_table

    def worker(self, image_reader, test_mode):
        self.getFeatureTableFromImages(image_reader, test_mode)
        self.normalizeFeatures()
        return self.getFeaturesTable()