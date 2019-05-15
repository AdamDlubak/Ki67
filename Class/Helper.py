import os
import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class Helper(object):

    def boldMax(self, s):
        is_max = s == s.max()
        return ['font-weight: bold; text-decoration: underline' if v else '' for v in is_max]
        
    def pairPlot(self, variables, data_file = "all_normalized_features_table", output_image_name = "features_pairplot"):
        df = pickle.load(open(variables['backup_folder'] + variables['pairplot_data_file']  + ".p", "rb"))
        variables, _, _ = pickle.load(open(variables['backup_folder'] + "parameters.p", "rb"))

        seaborn_plot = sns.pairplot(df[df.columns[0:variables['feature_numbers'] + 2]], 
                            hue=df.columns[variables['feature_numbers'] + 1])

        seaborn_plot.savefig(output_image_name + ".png")

    def printDF(self, variables):
        df = pickle.load(open(variables['backup_folder'] + "sorted_decision.p", "rb"))
        display(df)

    def featuresHistogram(self, variables, data_file = "all_normalized_features_table"):
        df = pickle.load(open(variables['backup_folder'] + data_file + ".p", "rb"))
        variables, _, _ = pickle.load(open(variables['backup_folder'] + "parameters.p", "rb"))

        for idx, feature in enumerate(df[df.columns[0 : variables['feature_numbers']]]):
            plt.figure(figsize=(15,10))
            low = df[df["Decision"] == variables['class_1']][feature]
            high = df[df["Decision"] == variables['class_2']][feature]
            plt.hist([low, high], 150, label=[variables['class_1'], variables['class_2']])
            plt.title(df.columns[idx])
            plt.legend(loc='upper right')
            plt.show()

    def printImage(self, path, filename, extension = ".jpeg", title = ""):
        image = cv2.imread(path + filename + extension)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.show()
    
    def plotImageHistogram(self, path, filename, bins = 255, threshold_value = 10, extension = ".jpeg", title = ""):
        image = cv2.imread(path + filename + extension)

        plt.hist(image[(image > threshold_value)].ravel(), bins = bins) 
        plt.title(title) 
        plt.savefig('foo.png')

        plt.show()

    def loadFuzzificationStats(self):
        df = pickle.load(open("Summaries/Fuzzification Statistics.p", "rb"))
        display(df)
        return df

    def loadFuzzificationParameters(self):
        df = pickle.load(open("Summaries/Fuzzification Parameters.p", "rb"))
        display(df)
        return df

    def loadDatasetResults(self, dataset, show = True):
        df = pd.read_csv(open("Results/" + dataset + ".csv", "rb"))
        df_to_print = df.style.apply(self.boldMax, subset=['Accuracy'])
        if show:
            display(df_to_print)
        return df

    def getParamsToTests(self, dataset_name, gausses):
        dataset_stats = self.loadDatasetResults(dataset_name, False)
        dataset_stats = dataset_stats.loc[(dataset_stats["Gausses"] == gausses) & (dataset_stats["Data Type"] == "Train")].reset_index()
        row_index = dataset_stats["Accuracy"].values.argmax()
        if dataset_stats.loc[[row_index]]["Operation"].values == "BruteForce Threshold":
            threshold = dataset_stats.loc[[row_index]]["Threshold"].values[0]
            s_function_center = dataset_stats.loc[[row_index]]["S-Functions Center"].values[0]
        elif dataset_stats.loc[[row_index]]["Operation"].values == "BruteForce S-Functions":
            threshold = dataset_stats.loc[[row_index + 1]]["Threshold"].values[0]
            s_function_center = dataset_stats.loc[[row_index + 1]]["S-Functions Center"].values[0]
        else:
            threshold = dataset_stats.loc[[row_index + 2]]["Threshold"].values[0]
            s_function_center = dataset_stats.loc[[row_index + 2]]["S-Functions Center"].values[0]

        return s_function_center, float(threshold)

    def getDatasetBestScores(self, dataset):
        for dataset_file in os.listdir("Results/"):
            if dataset_file[:-4] == dataset:
                df = pd.read_csv(open("Results/" + dataset_file, "rb"))
                break
        df = df.loc[df["Operation"] == "BruteForce Threshold"]
        df["F-Score Average"] = (df["F-Score A"] + df["F-Score B"]) / 2
        results = df.loc[df['F-Score Average'].idxmax()]

        return float(results["S-Functions Center"]), float(results["Threshold"])

    def loadBestResults(self):
        columns = ["Test type", "Dataset", "Gausses", "Data Type", "Operation", "Accuracy", "F-Score Average", "Precision A", "Precision B", "Recall A", "Recall B", "F-Score A", "F-Score B", "Support A", "Support B", "S-Functions Center", "S-Functions Width", "Threshold", "Time (s)", "Test date"]
        results = pd.DataFrame(columns = columns)
        for dataset_file in os.listdir("Results/"):
            df = pd.read_csv(open("Results/" + dataset_file, "rb"))
            df = df.loc[df["Operation"] == "BruteForce Threshold"]
            df["F-Score Average"] = (df["F-Score A"] + df["F-Score B"]) / 2
            results = results.append(df.loc[df['F-Score Average'].idxmax()])
        return results

    def saveFuzzificationStats(self, data):
        columns = ["Dataset", "Gausses","Samples", "Train s.", "Test s.", "Changed s.", "% changed s.", "Implicants", "Features", "F. after reduct"]
        try:
            df = pickle.load(open("Summaries/Fuzzification Statistics.p", "rb"))
        except (OSError, IOError):
            df = pd.DataFrame(columns = columns)
        
        df = df[(df[columns[0]] != data[0]) | (df[columns[1]] != data[1])]
        row = pd.DataFrame([data], columns = columns)
        df = df.append(row)
        df = df.sort_values(by=["Dataset", "Gausses"]).reset_index(drop=True)
        display(df)
        pickle.dump(df, open("Summaries/Fuzzification Statistics.p", "wb"))

    def saveFuzzificationParameters(self, data):
        begin_data = data[0:3]
        rest_data = [round(x, 3) for x in data[3:]]
        data = begin_data + rest_data
        columns = ["Dataset", "Gausses", "Auto Mode", "Mean 0", "Std 0", "Mean 1", "Std 1", "Mean 2", "Std 2", "Mean 3", "Std 3", "Mean 4", "Std 4", "Mean 5", "Std 5", "Mean 6", "Std 6", "Mean 7", "Std 7", "Mean 8", "Std 8", "Mean 9", "Std 9", "Mean 10", "Std 10", "Mean 11", "Std 11", "Mean 12", "Std 12", "Mean 13", "Std 13", "Mean 14", "Std 14", "Mean 15", "Std 15"]
        complement = np.empty(len(columns) - len(data), dtype=str)
        data = np.concatenate((data, complement), axis=0)
        if data[2] == "-1":
            data[2] = "True"
        else:
            data[2] = "False"
                
        try:
            df = pickle.load(open("Summaries/Fuzzification Parameters.p", "rb"))
        except (OSError, IOError):
            df = pd.DataFrame(columns = columns)
        
        df = df[(df[columns[0]] != data[0]) | (df[columns[1]] != data[1]) | (df[columns[2]] != data[2])]
        row = pd.DataFrame([data], columns = columns)
        df = df.append(row)
        df = df.sort_values(by=["Dataset", "Gausses"]).reset_index(drop=True)
        display(df)
        pickle.dump(df, open("Summaries/Fuzzification Parameters.p", "wb"))
