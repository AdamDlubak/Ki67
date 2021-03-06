from os import listdir
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import rescale
from skimage.color import gray2rgb

class ImageReader(object):
    def __init__(self, path, test_mode = 0):
        self.path = path
        self.images = []
        self.image_names = []
        self.image_decisions = []
        self.test_mode = test_mode

    def loadImages(self, settings):
        if self.test_mode == -1:
            name = "fragment.png"
            file_path = self.path + name 
            print(file_path)
            self.images.append(io.imread(file_path))
            self.image_names.append(name.replace(settings.extension, ''))
            if name[:-4] == settings.class_1:
                self.image_decisions.append(settings.class_1)
            else:
                self.image_decisions.append(settings.class_2)
        if self.test_mode == 1:
            image = io.imread(self.path + "base.png")
            mask = io.imread(self.path + "mask-gt.png")
            image_rescaled = rescale(image, 1.0 / 3.0, anti_aliasing=False)
            mask_rescaled = rescale(mask, 1.0 / 3.0, anti_aliasing=False)
            mask_3_channels = gray2rgb(mask_rescaled)
            image_rescaled_mask = image_rescaled * mask_3_channels

            from matplotlib import pyplot as plt

            fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
            ax[0].imshow(image_rescaled, cmap=plt.cm.gray)
            ax[1].imshow(image_rescaled_mask, cmap=plt.cm.gray)
            plt.show()
            print("File: {}".format(self.path + "base.png"))
            print("Width: {}\t Height: {}".format(image_rescaled_mask.shape[0], image_rescaled_mask.shape[1]))
            self.images.append(image_rescaled_mask)
            self.image_names.append(settings.file_name)
            self.image_decisions.append(settings.class_1)
        else:
            for name in listdir(self.path):
                file_path = self.path + name
                print(file_path)
                self.images.append(io.imread(file_path))
                self.image_names.append(name.replace(settings.extension, ''))
                if name[:-4] == settings.class_1:
                    self.image_decisions.append(settings.class_1)
                else:
                    self.image_decisions.append(settings.class_2)

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