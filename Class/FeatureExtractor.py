import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

class FeatureExtractor(object):
    def __init__(self, fuzzifier, settings):
        self.features_table = []
        self.fuzzifier = fuzzifier
        self.settings = settings
        
    def showResults(self, table):
        if self.settings.show_results:
           display(table)

    def thresholdImage(self, image):

        thresholded_image = []
        if image.ndim == 1:
            for x in image:
                if x != 0:
                    thresholded_image.append(x)
        else:
            
            for idx, x in enumerate(image):
                for idy, y in enumerate(x):
                    if y != 0:
                        thresholded_image.append(y)

        return [np.mean(thresholded_image)]

    def thresholdImageSTD(self, image):

        thresholded_image = []
        if image.ndim == 1:
            for x in image:
                if x != 0:
                    thresholded_image.append(x)
        else:
            
            for idx, x in enumerate(image):
                for idy, y in enumerate(x):
                    if y != 0:
                        thresholded_image.append(y)

        return [np.std(thresholded_image)]

    def  extractFeaturesFromImage(self, image, image_decision, image_name):
        
        idx = 0
        self.decision_table = pd.DataFrame({self.fuzzifier.features[idx].label: [""]})
        results = self.prepareImage(image, "r")
        self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        idx += 1

        # results = self.prepareImage(image, "r-std")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImageSTD(results)
        # idx += 1

        results = self.prepareImage(image, "g")
        self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        idx += 1

        # results = self.prepareImage(image, "g-std")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImageSTD(results)
        # idx += 1

        results = self.prepareImage(image, "b")
        self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        idx += 1

        # results = self.prepareImage(image, "b-std")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImageSTD(results)
        # idx += 1

        results = self.prepareImage(image, "s")
        self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        idx += 1

        # results = self.prepareImage(image, "s-std")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImageSTD(results)
        # idx += 1

        results = self.prepareImage(image, "h")
        self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        idx += 1

        # results = self.prepareImage(image, "h-std")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImageSTD(results)
        # idx += 1

        results = self.prepareImage(image, "v")
        self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        idx += 1

        # results = self.prepareImage(image, "v-std")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImageSTD(results)
        # idx += 1

        results = self.prepareImage(image, "intensity-mean")
        self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        idx += 1

        # results = self.prepareImage(image, "exred-mean")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        # idx += 1

        # results = self.prepareImage(image, "exblue-mean")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)

        # results = self.prepareImage(image, "exgreen-mean")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        # idx += 1

        # results = self.prepareImage(image, "region-centroid-col")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        # idx += 1

        # results = self.prepareImage(image, "region-centroid-row")
        # self.decision_table[self.fuzzifier.features[idx].label] = self.thresholdImage(results)
        # idx += 1

        # results = self.prepareImage(image, "aspect-ratio")
        # self.decision_table[self.fuzzifier.features[idx].label] = results
        # idx += 1

        # results = self.prepareImage(image, "extent")
        # self.decision_table[self.fuzzifier.features[idx].label] = results
        # idx += 1

        # results = self.prepareImage(image, "solidity")
        # self.decision_table[self.fuzzifier.features[idx].label] = results
        # idx += 1

        # results = self.prepareImage(image, "equi-diamter")
        # self.decision_table[self.fuzzifier.features[idx].label] = results
        # idx += 1

        image_column = np.full(self.decision_table.shape[0], image_name)
        self.decision_table['Image'] = image_column

        decision_column = np.full(self.decision_table.shape[0], image_decision)
        self.decision_table['Decision'] = decision_column

        self.decision_table['Predicted Value'] = ""

        return self.decision_table

    def getFeatureTableFromImages(self, image_reader):
        features_array = []
        thresholded_images = []
        self.features_table = pd.DataFrame()
        for idx, image in enumerate(image_reader.images):
            for pixels_row in tqdm(image):
                for pixel in pixels_row:
                    if pixel.all() != 0:
                        features_array.append(self.extractFeaturesFromImage(image, image_reader.image_decisions[idx], image_reader.image_names[idx]))
        self.features_table = pd.concat(features_array).reset_index(drop=True)

        self.showResults(self.features_table)

    def splitIntoRgbChannels(self, image):
        b, g, r = cv2.split(image)
        return r, g, b


    def splitIntoHsvChannels(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        return h, s, v

    def prepareImage(self, resized_image, image_type, width=224, height=224):

        # resized_image = cv2.resize(image, (width, height)) 

        r_image, g_image, b_image = self.splitIntoRgbChannels(resized_image)
        h_image, s_image, v_image = self.splitIntoHsvChannels(resized_image)

        imgray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 5, 255, 0)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = max(contours, key = cv2.contourArea)

        if image_type == "r" or image_type == "r-std":
            color_image = r_image
        if image_type == "g" or image_type == "g-std":
            color_image = g_image
        if image_type == "b" or image_type == "b-std":
            color_image = b_image
        if image_type == "h" or image_type == "h-std":
            color_image = h_image
        if image_type == "s" or image_type == "s-std":
            color_image = s_image
        if image_type == "v" or image_type == "v-std":
            color_image = v_image
        elif image_type == "intensity-mean":
            color_image = (r_image + b_image + g_image) / 3
        elif image_type == "exred-mean":
            color_image = (2 * r_image - (g_image + b_image))
        elif image_type == "exblue-mean":
            color_image = (2 * b_image - (r_image + g_image))
        elif image_type == "exgreen-mean":
            color_image = (2 * g_image - (r_image + b_image))
        elif image_type == "region-centroid-col":
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray_image, 127, 255, 0)
            M = cv2.moments(thresh)
            cX = int(M["m10"] / M["m00"])
            color_image = gray_image[:,cX]
        elif image_type == "region-centroid-row":
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray_image, 127, 255, 0)
            M = cv2.moments(thresh)
            cY = int(M["m01"] / M["m00"])
            color_image = gray_image[cY,:]
        elif image_type == "aspect-ratio":
            x, y, w, h = cv2.boundingRect(cnt)
            color_image = float(w) / h
        elif image_type == "extent":
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            rect_area = w * h
            color_image = float(area) / rect_area
        elif image_type == "solidity":
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            color_image = float(area) / hull_area
        elif image_type == "equi-diamter":
            area = cv2.contourArea(cnt)
            color_image = np.sqrt(4*area / np.pi)

        return color_image

    
    def normalizeFeatures(self):
        for x in self.fuzzifier.features:
            self.features_table[x.label] = (
                self.features_table[x.label] - self.features_table[x.label].min()) / (
                    self.features_table[x.label].max() - self.features_table[x.label].min())

        self.showResults(self.features_table)

    def getFeaturesTable(self):
        return self.features_table

    def worker(self, image_reader):
        self.getFeatureTableFromImages(image_reader)
        self.normalizeFeatures()
        return self.getFeaturesTable()