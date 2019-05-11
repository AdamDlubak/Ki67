import pandas as pd
import numpy as np
from skimage import io
from skimage.color import rgb2hsv, rgb2hed
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
            h_hed_image, e_hed_image, d_hed_image = self.splitIntoHedChannels(image)

            data_columns = {
                self.fuzzifier.features[0].label: r_image.ravel(),
                self.fuzzifier.features[1].label: g_image.ravel(),
                self.fuzzifier.features[2].label: b_image.ravel(),
                self.fuzzifier.features[3].label: h_image.ravel(),
                self.fuzzifier.features[4].label: s_image.ravel(),
                self.fuzzifier.features[5].label: v_image.ravel(),
                self.fuzzifier.features[6].label: h_hed_image.ravel(),
                self.fuzzifier.features[7].label: e_hed_image.ravel(),
                self.fuzzifier.features[8].label: d_hed_image.ravel(),
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
                ]] != 0).all(axis=1)]

            if idx == 0:
                self.features_table = tmp_features_table
            else:
                frames = [self.features_table, tmp_features_table]
                self.features_table = pd.concat(frames).reset_index(drop=True)

        self.showResults(self.features_table)

    def splitIntoRgbChannels(self, image):
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        return r, g, b

    def splitIntoHsvChannels(self, image):
        hsv_image = rgb2hsv(image)
        h = hsv_image[:, :, 0]
        s = hsv_image[:, :, 1]
        v = hsv_image[:, :, 2]
        return h, s, v


    def splitIntoHedChannels(self, image):
        hed_image = rgb2hed(image)
        h = hed_image[:, :, 0]
        e = hed_image[:, :, 1]
        d = hed_image[:, :, 2]
        return h, e, d

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