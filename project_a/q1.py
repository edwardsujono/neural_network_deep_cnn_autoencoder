""" 
Start script question 1
""" 

import numpy as np
from project_a.cnn_softmax import SoftmaxCNN
from project_a.data_collector import DataCollector
from project_a.data_visualization import DataVisualization

def question1():
    """ question 1 """
    print("Start question 1!")
    np.random.seed(10)

    batch_size = 128
    epoch = 100

    learning_rate = 0.05
    decay = 0.0001

    data_collector = DataCollector("../data/mnist.pkl")
    graph_visualizer = DataVisualization()

    trX, trY = data_collector.get_train_data()
    teX, teY = data_collector.get_test_data()

    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)

    trX, trY = trX[:12000], trY[:12000]
    teX, teY = teX[:2000], teY[:2000]

    """ 
    Question 1: SGD
    """
    cnn = SoftmaxCNN()
    cnn.init_learning_sgd(learning_rate, decay)
    cnn.create_model(
        convolutional_layer=[(15, 1, 9, 9), (20, 15, 5, 5)],
        pooling_layer=[(2, 2), (2, 2)],
        hidden_layer=[(20*3*3, 100)],
        num_output=10)
    accuracy, costs = cnn.start_train(trX, trY, teX, teY, epoch, batch_size)
    graph_visualizer.plot_graphs(
        list_x_point=range(epoch),
        list_y_point=accuracy,
        x_label="Epoch",
        y_label="Mean accuracy",
        title="Accuracy",
        figure_name="../data/Figure1.1-AccuracyEpoch.png",
        show_image=False
    )
    graph_visualizer.plot_graphs(
        list_x_point=range(epoch),
        list_y_point=costs,
        x_label="Epoch",
        y_label="Cross Entropy Cost",
        title="Training Cost",
        figure_name="../data/Figure1.2-CostEpoch.png",
        show_image=False
    )

    ind = np.random.randint(low=0, high=2000)
    outputs = cnn.test(teX[ind:ind+1,:])
    graph_visualizer.plot_images(
        outputs[0], figure_name="../data/Figure1.3.1-ConvLayer1.png", 
        number_column=5, limit_image=15, size=20,
        show_image=False
    )
    # graph_visualizer.plot_images(
    #     outputs[1], figure_name="../data/Figure1.3.3-ConvLayer2.png", 
    #     number_column=5, limit_image=20, size=6,
    #     show_image=False
    # )
    graph_visualizer.plot_images(
        outputs[2], figure_name="../data/Figure1.3.2-PoolLayer1.png", 
        number_column=5, limit_image=15, size=10,
        show_image=False
    )
    # graph_visualizer.plot_images(
    #     outputs[3], figure_name="../data/Figure1.3.4-PoolLayer2.png", 
    #     number_column=5, limit_image=20, size=3,
    #     show_image=False
    # )

    print("Finished question 1")


if __name__ == "__main__":
    question1()
