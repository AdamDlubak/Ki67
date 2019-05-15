from os import listdir
from matplotlib import pyplot as plt
from skimage import io

class ImageReader(object):
    def __init__(self, path):
        self.path = path
        self.images = []
        self.image_names = []
        self.image_decisions = []

    def loadImages(self, variables):
        for name in listdir(self.path):
            file_path = self.path + name
            print(file_path)
            self.images.append(io.imread(file_path))
            self.image_names.append(name.replace(variables["extension"], ''))
            if name[:-4] == variables["class_1"]:
                self.image_decisions.append(variables["class_1"])
            else:
                self.image_decisions.append(variables["class_2"])
        
    def printImage(self, image, title=""):
        fig = plt.figure()
        fig.suptitle(title, fontsize=13)
        plt.axis('off')
        plt.imshow(image)
        plt.show()  

    def printImages(self, 
                 image_1,
                 image_2 = None,
                 image_3 = None,
                 image_4 = None,
                 image_5 = None,
                 image_6 = None,
                 title_1="",
                 title_2="",
                 title_3="",
                 title_4="",
                 title_5="",
                 title_6=""):
        
        plt.figure(figsize=(18, 18))

        if image_1 is not None:
            ax = plt.subplot(161)
            ax.set_title(title_1, fontsize=13)
            plt.axis('off')
            plt.imshow(image_1)

        if image_2 is not None:
            ax = plt.subplot(162)
            ax.set_title(title_2, fontsize=13)
            plt.axis('off')
            plt.imshow(image_2)

        if image_3 is not None:
            ax = plt.subplot(163)
            ax.set_title(title_3, fontsize=13)
            plt.axis('off')
            plt.imshow(image_3)

        if image_4 is not None:
            ax = plt.subplot(164)
            ax.set_title(title_4, fontsize=13)
            plt.axis('off')
            plt.imshow(image_4)

        if image_5 is not None:
            ax = plt.subplot(165)
            ax.set_title(title_5, fontsize=13)
            plt.axis('off')
            plt.imshow(image_5)

        if image_6 is not None:
            ax = plt.subplot(166)
            ax.set_title(title_6, fontsize=13)
            plt.axis('off')
            plt.imshow(image_6)

        plt.show()