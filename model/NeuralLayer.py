from model.MiningModelElement import MiningModelElement


class NeuralLayer(MiningModelElement):
    def __init__(self, weights, activation_function, number_of_inputs, number_of_outputs, bias, use_bias):
        self.weights = weights
        self.activation_function = activation_function
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.bias = bias
        self.use_bias = use_bias

    def add_new_neuron(self):
        pass

    def connect_layer(self):
        pass