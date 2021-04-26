from model.MiningModelElement import MiningModelElement


class NeuralLayer(MiningModelElement):
    def __init__(self, number_of_neurons, activation_function, threshold, width, altitude, neurons):
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function
        self.threshold = threshold
        self.width = width
        self.altitude = altitude
        self.neurons = neurons

    def add_new_neuron(self):
        pass

    def connect_layer(self):
        pass