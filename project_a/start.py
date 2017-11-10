""" 
Start script
""" 

import numpy as np
from project_a.cnn_softmax import SoftmaxCNN
from common.data_collector import DataCollector
from common.data_visualization import DataVisualization
from common.data_visualization import DataVisualizationAdjacent

def main():
    print("Start!")

    np.random.seed(10)
    batch_size = 128
    epoch = 8

    data_collector = DataCollector("../data/mnist.pkl")
    graph_visualizer = DataVisualization()
    image_visualizer = DataVisualizationAdjacent()


    trX, trY = data_collector.get_train_data()
    teX, teY = data_collector.get_test_data()

    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)

    # trX, trY = trX[:12000], trY[:12000]
    # teX, teY = teX[:2000], teY[:2000]

    """ 
    Question 1: SGD
    """
    cnn = SoftmaxCNN()
    cnn.init_learning_sgd(learning_rate=0.05, decay=0.0001)
    cnn.create_model(
        convolutional_layer=[(15, 1, 9, 9), (20, 15, 5, 5)],
        pooling_layer=[(2, 2), (2, 2)],
        hidden_layer=[(20*3*3, 100)],
        num_output=10)
    accuracy, costs = cnn.start_train(trX, trY, teX, teY, epoch, batch_size)
    graph_visualizer.show_plot(
        list_x_point=range(epoch),
        list_y_point=accuracy,
        x_label="Epoch",
        y_label="Mean accuracy",
        title="Accuracy",
        figure_name="Figure1.1-AccuracyEpoch.png"
    )
    graph_visualizer.show_plot(
        list_x_point=range(epoch),
        list_y_point=costs,
        x_label="Epoch",
        y_label="Cross Entropy Cost",
        title="Training Cost",
        figure_name="Figure1.2-CostEpoch.png"
    )
    print(cnn.conv_layers[0].get_value())
    return

    """
    Question 2: SGD with momentum
    """
    cnn2 = SoftmaxCNN()
    cnn2.init_learning_sgd(learning_rate=0.05, decay=0.0001, momentum=0.1)
    cnn2.create_model(
        convolutional_layer=[(15, 1, 9, 9), (20, 15, 5, 5)],
        pooling_layer=[(2, 2), (2, 2)],
        hidden_layer=[(20*3*3, 100)],
        num_output=10)
    accuracy, costs = cnn2.start_train(trX, trY, teX, teY, epoch, batch_size)
    graph_visualizer.show_plot(
        list_x_point=range(epoch),
        list_y_point=accuracy,
        x_label="Epoch",
        y_label="Mean accuracy",
        title="Accuracy",
        figure_name="Figure2.1-AccuracyEpoch.png"
    )
    graph_visualizer.show_plot(
        list_x_point=range(epoch),
        list_y_point=costs,
        x_label="Epoch",
        y_label="Cross Entropy Cost",
        title="Training Cost",
        figure_name="Figure2.2-CostEpoch.png"
    )

    """
    Question 3: RPMS Prop
    """
    cnn3 = SoftmaxCNN()
    cnn3.init_learning_RMSProp(learning_rate=0.001, decay=0.0001, rho=0.9, epsilon=1e-6)
    cnn3.create_model(
        convolutional_layer=[(15, 1, 9, 9), (20, 15, 5, 5)],
        pooling_layer=[(2, 2), (2, 2)],
        hidden_layer=[(20*3*3, 100)],
        num_output=10)
    accuracy, costs = cnn3.start_train(trX, trY, teX, teY, epoch, batch_size)
    graph_visualizer.show_plot(
        list_x_point=range(epoch),
        list_y_point=accuracy,
        x_label="Epoch",
        y_label="Mean accuracy",
        title="Accuracy",
        figure_name="Figure3.1-AccuracyEpoch.png"
    )
    graph_visualizer.show_plot(
        list_x_point=range(epoch),
        list_y_point=costs,
        x_label="Epoch",
        y_label="Cross Entropy Cost",
        title="Training Cost",
        figure_name="Figure3.2-CostEpoch.png"
    )


    # np.random.seed(10)
    # batch_size = 128
    # noIters = 25

    # data_collector = DataCollector("../data/mnist.pkl")

    # trX, trY = data_collector.get_train_data()
    # teX, teY = data_collector.get_test_data()

    # trX = trX.reshape(-1, 1, 28, 28)
    # teX = teX.reshape(-1, 1, 28, 28)

    # # For testing limit size
    # trX, trY = trX[:12000], trY[:12000]
    # teX, teY = teX[:2000], teY[:2000]


    # pylab.figure()
    # pylab.plot(range(noIters), a)
    # pylab.xlabel('epochs')
    # pylab.ylabel('test accuracy')
    # pylab.savefig('figure_2a_1.png')

    # w = w1.get_value()
    # pylab.figure()
    # pylab.gray()
    # for i in range(25):
    #     pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(w[i,:,:,:].reshape(9,9))
    # #pylab.title('filters learned')
    # pylab.savefig('figure_2a_2.png')

    # ind = np.random.randint(low=0, high=2000)
    # convolved, pooled = test(teX[ind:ind+1,:])

    # pylab.figure()
    # pylab.gray()
    # pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
    # #pylab.title('input image')
    # pylab.savefig('figure_2a_3.png')

    # pylab.figure()
    # pylab.gray()
    # for i in range(25):
    #     pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved[0,i,:].reshape(20,20))
    # #pylab.title('convolved feature maps')
    # pylab.savefig('figure_2a_4.png')

    # pylab.figure()
    # pylab.gray()
    # for i in range(5):
    #     pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled[0,i,:].reshape(5,5))
    # #pylab.title('pooled feature maps')
    # pylab.savefig('figure_2a_5.png')

    # pylab.show()




if __name__ == "__main__":
    main()