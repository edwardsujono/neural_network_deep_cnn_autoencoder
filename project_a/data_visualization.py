""" Data Visualization for CNN Softmax """

import matplotlib.pyplot as plt
import numpy as np
import math


class DataVisualization:
    """
    Data Visualizer
    - Graph visualizer
    - Image visualizer
    """ 

    def __init__(self):
       return

    def plot_graphs(self, list_x_point, list_y_point, x_label, y_label, title, figure_name="", show_image=True):
        """ Plot graphs from data"""
        plt.figure()
        plt.plot(list_x_point, list_y_point)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        
        if figure_name != "":
            plt.savefig(figure_name)
        
        if show_image:
            plt.show()


    def plot_images(self, data, figure_name="", number_column=10, limit_image=20, size=28, transpose=True, show_image=True):
        """ Plot images from data"""
        width = height = size

        plt.figure()
        plt.gray()
        row = limit_image/number_column

        for i in range(1, limit_image+1):
            plt.subplot(row, number_column, i)
            plt.axis('off')
            if transpose:
                norm_image = self.normalize_image(data[:, i-1].reshape(width, height))
            else:
                norm_image = self.normalize_image(data[i-1].reshape(width, height))
            plt.imshow(norm_image)

        if figure_name != "":
            plt.savefig(figure_name)
        
        if show_image:
            plt.show()


    def normalize_image(self, image):
        """ Normalize image data """
        min_val = np.min(image)
        max_val = np.max(image)

        return (image-min_val)/(max_val-min_val)
