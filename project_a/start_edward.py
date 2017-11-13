from cnn_softmax_edward import CNN
import numpy as np
from common.data_collector import DataCollector

if __name__ == "__main__":

    data_collector = DataCollector("../mnist.pkl")
    np.random.seed(10)

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()
    validate_x, validate_y = data_collector.get_validation_data()

    num_feature = len(train_x[0])
    # , (20, 15, 5, 5)
    cnn = CNN(learning_rate=0.05, decays=0.0001,
              convolutional_layer=[(15, 1, 9, 9), (20, 15, 5, 5)],
              pooling_layer=[(2, 2), (2, 2)],
              hidden_layer=[(20*3*3, 100)],
              num_output=10,
              use_momentum=True
              )

    cnn.start_train(tr_x=train_x.reshape(-1, 1, 28, 28), tr_y=train_y,
                    te_x=test_x.reshape(-1, 1, 28, 28), te_y=test_y,
                    batch_size=128, epochs=10)
