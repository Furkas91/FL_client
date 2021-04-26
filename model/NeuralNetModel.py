class NeuralNetModel:
    def __init__(self, name, algorithm_name, activation_function, input_layer, output_layer, inner_layers,
                 number_of_layers):
        self.name = name
        self.algorithm_name = algorithm_name
        self.activation_function = activation_function
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.inner_layers = inner_layers
        self.number_of_layers = number_of_layers

    def connect_layers(self):
        pass
