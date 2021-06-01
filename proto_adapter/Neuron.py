from proto_adapter.MiningModelElement import MiningModelElement


class Neuron(MiningModelElement):
    def __init__(self, id, bias, connectors):
        self.id = id
        self.bias = bias
        self.connectors = connectors

    def calculate_output(self):
        pass

    def add_new_connection(self):
        pass

    def merge(self):
        pass