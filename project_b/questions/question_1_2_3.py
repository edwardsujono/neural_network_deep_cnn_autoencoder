from project_b.autoencoder_softmax import SoftmaxAutoEncoder
from common.data_visualization import DataVisualization, DataVisualizationAdjacent, DataVisualizationWithLabels
from common.data_collector import DataCollector
import math
"""
    NUMBER 1 UNTIL 3 IS TIGHTLY BINDED CODE, BECAUSE WE NEED TO DO A LOT OF COMPARISON
    THEREFORE, SEPARATION OF CODE MEANS RETRAINING A LOT OF 
"""

if __name__ == "__main__":
    """
        QUESTION 1
    """

    # This part of assignment aims to provide you with some exposure to the use of autoencoders. Use
    # the full MNIST dataset for this problem.
    # Hints: Use corruption level = 0.1, training epochs = up to about 25, learning rate = 0.1, and batch
    # size = 128 for training of all the layers.

    # 1) Design a stacked denoising autoencoder consisting of three hidden-layers; 900 neurons in the
    # first hidden-layer, 625 neurons in the second hidden-layer, and 400 neurons in the third
    # hidden-layer. To train the network:
    # - Use the training dataset of MNIST digits
    # - Corrupt the input data using a binomial distribution at 10% corruption level.
    # - Use cross-entropy as the cost function
    #
    # Plot
    #
    # - a. learning curves for training each layer
    # - b. Plot 100 samples of weights (as images) learned at each layer
    # - c. For 100 representative test images plot:
    #     - reconstructed images by the network.
    #     - Hidden layer activation

    # a. Learning curve for training each layer.

    data_collector = DataCollector("mnist.pkl")

    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()
    validate_x, validate_y = data_collector.get_validation_data()

    num_feature = len(train_x[0])
    list_hidden_layer = [900, 625, 400]
    epochs = 50
    batch_size = 128
    num_output = 10

    softmax = SoftmaxAutoEncoder(num_features=num_feature, num_outputs=num_output,
                                 list_hidden_layer=list_hidden_layer, learning_rate=0.1)

    softmax.start_train_auto_encoder(epochs=epochs, batch_size=batch_size,
                                     train_x=train_x, train_y=train_y, verbose=True)

    list_cost_all_layer = softmax.get_total_costs_of_auto_encoder()

    visualizer = DataVisualization()

    cntr = 1

    for list_cost in list_cost_all_layer:
        visualizer.show_plot(list_x_point=range(epochs),
                             list_y_point=list_cost,
                             x_label='epochs',
                             y_label='Binary Cross Entropy cost',
                             title='Binary Cross Entropy layer %s' % cntr,
                             figure_name='project_b/images/1a_cost_%s.png' % cntr)
        cntr += 1

    list_weights = softmax.get_weights_on_each_layer()
    hidden_layer_activation = softmax.get_hidden_layer_activation()
    number_column = 10
    visualizer = DataVisualizationAdjacent()

    number_layer = 0
    number_column = 10
    visualizer.show_plot \
            (data=list_weights[number_layer], figure_name="project_b/images/1b_auto_first.png", number_column=number_column,
             limit_image=100, size=28, transpose=True)

    # b. Plot 100 samples of weights (as images) learned at each layer

    list_weights = softmax.get_weights_on_each_layer()
    hidden_layer_activation = softmax.get_hidden_layer_activation()
    number_column = 10
    visualizer = DataVisualizationAdjacent()

    # b.1. plot first layer of auto encoder.

    number_layer = 0
    number_column = 10
    visualizer.show_plot(
        data=list_weights[number_layer],
        figure_name="project_b/images/1b_auto_first.png",
        number_column=number_column,
        limit_image=100,
        size=28,
        transpose=True
    )

    # Plot with different different xavier range -1 and 1

    num_feature = len(train_x[0])
    list_hidden_layer = [900, 625, 400]
    epochs = 50
    batch_size = 128
    num_output = 10

    softmax_xavier = SoftmaxAutoEncoder(num_features=num_feature, num_outputs=num_output,
                                        list_hidden_layer=list_hidden_layer, learning_rate=0.1, xavier_range=1)

    softmax_xavier.start_train_auto_encoder(epochs=epochs, batch_size=batch_size,
                                            train_x=train_x, train_y=train_y, verbose=True)

    list_weights_xavier = softmax_xavier.get_weights_on_each_layer()

    number_layer = 0
    number_column = 10
    visualizer.show_plot(
        data=list_weights_xavier[number_layer],
        figure_name="project_b/images/1b_auto_first_xavier_range.png",
        number_column=number_column,
        limit_image=100,
        size=28,
        transpose=True
    )

    # Check construction with different range of xavier initialization

    number_column = 10
    reconstructed_images_xavier_diff = softmax_xavier.get_reconstructed_image()
    visualizer = DataVisualizationAdjacent()
    visualizer.show_plot(
        data=reconstructed_images_xavier_diff,
        figure_name="./project_b/images/1c_xavier_range_reconstructed_image.png",
        number_column=number_column,
        limit_image=100,
        size=28,
        transpose=False
    )

    # b.2. plot second layer of auto encoder

    number_layer = 1
    number_column = 10
    visualizer.show_plot(
        data=list_weights[number_layer],
        figure_name="project_b/images/1b_auto_second.png",
        number_column=number_column,
        limit_image=100,
        size=int(math.sqrt(list_hidden_layer[number_layer - 1])),
        transpose=True
    )

    # b.3. Plot third layer of auto encoder

    number_layer = 2
    number_column = 10
    visualizer.show_plot(
        data=list_weights[number_layer],
        figure_name="project_b/images/1b_auto_third.png",
        number_column=number_column,
        limit_image=100,
        size=int(math.sqrt(list_hidden_layer[number_layer - 1])),
        transpose=True
    )

    # c. For 100 representative test images plot
    # - reconstructed images by the network.
    # - Hidden layer activation

    # ### Reconstructed images of the network:

    # get the reconstructed images
    number_column = 10
    reconstructed_images = softmax.get_reconstructed_image()
    visualizer.show_plot(
        data=reconstructed_images,
        figure_name="./project_b/images/1c_reconstructed_image.png",
        number_column=number_column,
        limit_image=100,
        size=28,
        transpose=False
    )

    # ### Reconstructed image not share weights

    num_feature = len(train_x[0])
    list_hidden_layer = [900, 625, 400]
    epochs = 50
    batch_size = 128
    num_output = 10

    softmax_not_share_weight = SoftmaxAutoEncoder(num_features=num_feature, num_outputs=num_output,
                                                  list_hidden_layer=list_hidden_layer, learning_rate=0.1,
                                                  shared_weight=False
                                                  )

    softmax_not_share_weight.start_train_auto_encoder(epochs=epochs, batch_size=batch_size,
                                                      train_x=train_x, train_y=train_y, verbose=True)

    number_column = 10
    reconstructed_images_not_shared_weight = softmax_not_share_weight.get_reconstructed_image()
    visualizer.show_plot(
        data=reconstructed_images_not_shared_weight,
        figure_name="./project_b/images/1c_reconstructed_image_not_shared_weight.png",
        number_column=number_column,
        limit_image=100,
        size=28,
        transpose=False
    )

    # ### Hidden layer activation

    hidden_layer_cnt = 1
    for val_hidden_layer in hidden_layer_activation:
        visualizer.show_plot(
            data=val_hidden_layer,
            figure_name="./project_b/images/1c_hidden_layer_%s.png" % hidden_layer_cnt,
            number_column=number_column,
            limit_image=100,
            size=int(math.sqrt(len(val_hidden_layer[0]))),
            transpose=False
        )
        hidden_layer_cnt += 1

    """
        QUESTION 2
    """
    # ## 2.Train a five-layer feedforward neural network to recognize MNIST data,
    #  initialized by the three hidden layers learned in part (1) and by adding a softmax
    #  layer as the output layer. Plot the training errors and test accuracies during training.

    epochs = 25
    batch_size = 128

    softmax.start_train_the_full(epochs=epochs, batch_size=batch_size,
                                 train_x=train_x, train_y=train_y,
                                 test_x=test_x, test_y=test_y)

    list_cost, list_prediction = softmax.get_total_cost_and_prediction_full()

    visualizer = DataVisualization()

    visualizer.show_plot(list_x_point=range(epochs),
                         list_y_point=list_cost,
                         x_label='epochs',
                         y_label='Categorical Cross Entropy',
                         title='Categorical Cross Entropy',
                         figure_name='../data/project_b/2_cost.png')

    visualizer.show_plot(list_x_point=range(epochs),
                         list_y_point=list_prediction,
                         x_label='epochs',
                         y_label='accuracy',
                         title='predictions',
                         figure_name='../data/project_b/2_prediction.png')

    """
        QUESTION 3
    """

    num_feature = len(train_x[0])
    list_hidden_layer = [900, 625, 400]
    epochs = 50
    batch_size = 128
    num_output = 10
    momentum = 0.1
    sparsity_parameter = 0.05
    penalty_parameter = 0.5

    softmax_momentum_sparsity = SoftmaxAutoEncoder(
        num_features=num_feature, num_outputs=num_output,
        list_hidden_layer=list_hidden_layer, learning_rate=0.1,
        sparsity_parameter=sparsity_parameter, penalty_parameter=penalty_parameter,
        momentum=momentum, use_sparsity=True, use_momentum=True
    )

    softmax_momentum_sparsity.start_train_auto_encoder(epochs=epochs, batch_size=batch_size,
                                                       train_x=train_x, train_y=train_y, verbose=True)


    list_cost_momentum_sparsity_all_layer = softmax_momentum_sparsity.get_total_costs_of_auto_encoder()
    list_normal_cost_all_layer = softmax.get_total_costs_of_auto_encoder()

    visualizer = DataVisualizationWithLabels()

    cntr = 1

    for list_cost_momentum_sparsity, list_normal_cost in zip(list_cost_momentum_sparsity_all_layer,
                                                             list_normal_cost_all_layer):
        visualizer.show_plot(list_x_point=[range(epochs) for k in range(2)],
                             list_y_point=[list_cost_momentum_sparsity, list_normal_cost],
                             x_label='epochs',
                             y_label='Binary Cross Entropy cost',
                             title='Binary Cross Entropy of Layer %s ' % cntr,
                             figure_name='project_b/images/3a_cost_layer_%s.png' % cntr,
                             labels=['cost with momentum and sparsity', 'cost']
                             )
        cntr += 1

    # Compare result from 1.b first layer

    list_weights_sparsity_momentum = softmax_momentum_sparsity.get_weights_on_each_layer()
    list_weights = softmax.get_weights_on_each_layer()
    hidden_layer_activation_softmax = softmax_momentum_sparsity.get_hidden_layer_activation()
    number_column = 10
    visualizer = DataVisualizationAdjacent()

    number_layer = 0
    visualizer.show_plot(
        data=list_weights_sparsity_momentum[number_layer],
        figure_name="project_b/images/3b_auto_first.png",
        number_column=number_column,
        limit_image=100,
        size=28,
        transpose=True
    )

    # Compare result from 1.b. second layer

    number_layer = 1
    visualizer.show_plot(
        data=list_weights_sparsity_momentum[number_layer],
        figure_name="project_b/images/3b_auto_second.png",
        number_column=number_column,
        limit_image=100,
        size=int(math.sqrt(list_hidden_layer[number_layer - 1])),
        transpose=True
    )

    # Compare result from 1.b third layer

    number_layer = 2
    visualizer.show_plot(
        data=list_weights_sparsity_momentum[number_layer],
        figure_name="project_b/images/3b_auto_third.png",
        number_column=number_column,
        limit_image=100,
        size=int(math.sqrt(list_hidden_layer[number_layer - 1])),
        transpose=True
    )

    # Compare result from 1.c

    # ### Reconstrucred images

    # get the reconstructed images
    reconstructed_images = softmax_momentum_sparsity.get_reconstructed_image()
    visualizer.show_plot(
        data=reconstructed_images,
        figure_name="./project_b/images/3c_reconstructed_image.png",
        number_column=number_column,
        limit_image=100,
        size=28,
        transpose=False
    )

    # ### Hidden layer activation

    hidden_layer_cnt = 1

    for val_hidden_layer in hidden_layer_activation_softmax:
        visualizer.show_plot(
            data=val_hidden_layer,
            figure_name="./project_b/images/3c_hidden_layer-activation_%s.png" % hidden_layer_cnt,
            number_column=number_column,
            limit_image=100,
            size=int(math.sqrt(len(val_hidden_layer[0]))),
            transpose=False
        )
        hidden_layer_cnt += 1

    # #### compare result from 2

    epochs = 25

    batch_size = 128

    softmax_momentum_sparsity.start_train_the_full(epochs=epochs, batch_size=batch_size,
                                                   train_x=train_x, train_y=train_y,
                                                   test_x=test_x, test_y=test_y)

    list_cost_spar_mom, list_prediction_spar_mom = softmax_momentum_sparsity.get_total_cost_and_prediction_full()
    list_cost, list_prediction = softmax.get_total_cost_and_prediction_full()

    visualizer = DataVisualizationWithLabels()

    visualizer.show_plot(list_x_point=[range(epochs) for k in range(2)],
                         list_y_point=[list_cost, list_cost_spar_mom],
                         x_label='epochs',
                         y_label='Categorical Cross Entropy',
                         title='Cost result',
                         figure_name='project_b/images/3_cost.png',
                         labels=['normal', 'with sparsity parameter and momentum']
                         )

    visualizer.show_plot(list_x_point=[range(epochs) for k in range(2)],
                         list_y_point=[list_prediction, list_prediction_spar_mom],
                         x_label='epochs',
                         y_label='accuracy',
                         title='predictions result',
                         figure_name='project_b/images/3_prediction.png',
                         labels=['normal', 'with sparsity parameter and momentum']
                         )



