from google.protobuf.struct_pb2 import NULL_VALUE
from torch import nn

from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor, MapDescriptor
from model import DenseLayer, NormalizationLayer1
from torch_model.NN_model import UniversalNet

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

    def to_proto(self):
        proto_model = Descriptor(object=ObjectDescriptor(
            class_name='org.etu.fl.classification.nn.NNModel',
            fields={
                'algorithmName': Descriptor(isNull=NULL_VALUE),
                'sets': Descriptor(list=ListDescriptor(descriptors=[a.to_proto() for a in self.layers]))
            }
        ))
        return proto_model

    def to_torch_model(self):
        layers = []
        activations = []
        normalizations = []
        last_features = 0
        has_normalization = True
        for layer in self.layers:
            if isinstance(layer, NormalizationLayer1):
                normalizations.append(nn.BatchNorm1d(last_features))
                has_normalization = True
            else:
                torch_layer = layer.to_torch_layer()
                layers.append(torch_layer[0])
                activations.append((torch_layer[1]))
                last_features = layer.out_features
                if not has_normalization:
                    normalizations.append(lambda x: x)
                has_normalization = False
        normalizations.append(lambda x: x)
        return UniversalNet(layers=layers, activations=activations, normalizations=normalizations)

    def get_weights(self, torch_model):
        i = 0
        for layer in self.layers:
            if not isinstance(layer, NormalizationLayer1):
                layer.get_weights(torch_model.layers[i])
                i += 1
