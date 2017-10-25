from softmax_with_autoencoder import SoftmaxAutoEncoder
from common.data_collector import DataCollector

if __name__ == "__main__":

    data_collector = DataCollector("../mnist.pkl")

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()
    validate_x, validate_y = data_collector.get_validation_data()

    num_feature = len(train_x[0])

    softmax = SoftmaxAutoEncoder(num_features=num_feature, num_outputs=10,
                                 list_hidden_layer=[900], learning_rate=0.05)

    # softmax.start_train_auto_encoder(epochs=10, batch_size=20000,
    #                                  train_x=train_x, train_y=train_y)

    softmax.start_train_the_full(epochs=100, batch_size=20000,
                                 train_x=train_x, train_y=train_y,
                                 test_x=test_x, test_y=test_y)