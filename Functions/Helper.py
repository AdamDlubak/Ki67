import cv2
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

class Helper(object):

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
            low = df[df["Decision"] == variables['d_low']][feature]
            high = df[df["Decision"] == variables['d_high']][feature]
            plt.hist([low, high], 150, label=[variables['d_low'], variables['d_high']])
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