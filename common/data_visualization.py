import matplotlib.pyplot as plt
from common.utils import normalize_image


class DataVisualization:

    def __init__(self):
       return

    def show_plot(self, list_x_point, list_y_point, x_label, y_label, title, figure_name):
        plt.figure()
        plt.plot(list_x_point, list_y_point)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(figure_name)
        plt.show()


class DataVisualizationWithLabels:

    def __init__(self):
       return

    def show_plot(self, list_x_point, list_y_point, x_label, y_label, title, figure_name, labels):

        plt.figure()

        for cnt in range(len(labels)):
           plt.plot(list_x_point[cnt], list_y_point[cnt], label=labels[cnt])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.savefig(figure_name)
        plt.show()


class DataVisualizationAdjacent:

    def __init__(self):
        return

    def show_plot(self, data, figure_name, number_column=10, limit_image=20, size=28, transpose=True):

        width = height = size

        plt.figure()
        plt.gray()
        row = limit_image/number_column

        for i in range(1, limit_image+1):
            plt.subplot(row, number_column, i)
            plt.axis('off')
            if transpose:
                norm_image = normalize_image(data[:, i-1].reshape(width, height))
            else:
                norm_image = normalize_image(data[i-1].reshape(width, height))
            plt.imshow(norm_image)

        plt.savefig(figure_name)
        print 'Finish Plotting'
        plt.show()

