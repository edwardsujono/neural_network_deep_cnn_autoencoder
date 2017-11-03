from autoencoder_softmax import SoftmaxAutoEncoder
from common.data_collector import DataCollector
from common.data_visualization import DataVisualizationAdjacent, DataVisualization

if __name__ == "__main__":

    data_collector = DataCollector("../mnist.pkl")

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()
    validate_x, validate_y = data_collector.get_validation_data()

    num_feature = len(train_x[0])
    list_hidden_layer = [100]
    epochs = 10
    batch_size = 2000

    softmax = SoftmaxAutoEncoder(num_features=num_feature, num_outputs=10,
                                 list_hidden_layer=list_hidden_layer,
                                 learning_rate=0.1, use_sparsity=True, use_momentum=True)

    softmax.start_train_auto_encoder(epochs=epochs, batch_size=batch_size,
                                     train_x=train_x, train_y=train_y, verbose=True)

    reconstructed_image = softmax.get_reconstructed_image()

    # softmax.start_train_the_full(epochs=epochs, batch_size=batch_size,
    #                              train_x=train_x, train_y=train_y,
    #                              test_x=test_x, test_y=test_y)

    # first_layer_input = softmax.get_output_on_each_layer()
    visualizer = DataVisualizationAdjacent()
    # visualizer.show_plot(number_column=list_hidden_layer[0], number_row=2, data=first_layer_input[-1], figure_name="images/1_data.png")
    number_column = 10

    visualizer.show_plot \
        (list_data=batch_size,
         data=reconstructed_image,
         figure_name="images/1c_reconstructed_image.png",
         number_column=number_column,
         limit_image=100,
         size=28
         )

    # list_cost = softmax.get_total_costs_of_auto_encoder()
    #
    # visualizer = DataVisualization()
    #
    # visualizer.show_plot(list_x_point=range(epochs),
    #                      list_y_point=list_cost,
    #                      x_label='epochs',
    #                      y_label='Binary Cross Entropy cost',
    #                      title='Binary Cross Entropy',
    #                      figure_name='images/1a_cost.png')
