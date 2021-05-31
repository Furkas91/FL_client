from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor, MapDescriptor
from model import DenseLayer, NormalizationLayer1

layer_type = {
    'org.etu.fl.classification.nn.NNDenseLayerModelElement': DenseLayer,
    'org.etu.fl.classification.nn.NNBatchLayerModelElement': NormalizationLayer1
}


class NeuralNetModel:

    def __init__(self, algorithm_name, layers):
        self.algorithm_name = algorithm_name
        self.layers = layers

    @staticmethod
    def from_proto(proto_model):
        proto_model = proto_model.object.fields
        algorithm_name = proto_model['algorithmName']
        layers = []
        for layer in proto_model['sets'].list.descriptors:
            layers.append(layer_type[layer.object.class_name].from_proto(layer))
        return NeuralNetModel(algorithm_name=algorithm_name, layers=layers)
