from collections import OrderedDict

from google.protobuf.struct_pb2 import NULL_VALUE
from torch import nn, Tensor

from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor, MapDescriptor, EnumDescriptor
from proto_adapter.MiningModelElement import MiningModelElement
import torch.nn.functional as F
import numpy as np

activations = {
    'RELU': F.relu,
    'SOFTMAX': F.softmax
}
def get_vector_from_proto(proto_vector):
    vector = []
    for elem in proto_vector.list.descriptors:
        vector.append(elem.double_value)
    return np.asarray(vector)


def get_proto_from_vector(vector):
    return Descriptor(list=ListDescriptor(
        descriptors=[Descriptor(double_value=x) for x in vector]))


def get_weights_from_proto(proto_weights):
    weights = []
    for proto_vector in proto_weights.list.descriptors:
        weights.append(get_vector_from_proto(proto_vector))
    return np.asarray(weights)


def get_proto_from_weights(weights):
    return Descriptor(list=ListDescriptor(
        descriptors=[get_proto_from_vector(vector) for vector in weights]
    ))


class DenseLayer(MiningModelElement):
    def __init__(self, weights, activation_function, in_features, out_features, bias, use_bias):
        self.weights = weights
        self.activation_function = activation_function
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.use_bias = use_bias

    @staticmethod
    def from_proto(proto_layer):
        proto_layer = proto_layer.object.fields['properties'].list.descriptors
        weights = get_weights_from_proto(proto_layer[0])
        in_features = proto_layer[1].int_value
        out_features = proto_layer[2].int_value
        bias = get_vector_from_proto(proto_layer[3])
        use_bias = proto_layer[4].bool_value
        activation_function = proto_layer[5].enumeration.enum_value_name
        return DenseLayer(
            weights=weights,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            use_bias=use_bias,
            activation_function=activation_function
        )

    def to_proto(self):
        proto_layer = Descriptor(object=ObjectDescriptor(
            class_name='org.etu.fl.classification.nn.NNDenseLayerModelElement',
            fields={
                'id': Descriptor(string_value='NULL_VALUE'),
                'set': Descriptor(list=ListDescriptor()),
                'properties': Descriptor(list=ListDescriptor(descriptors=[
                    get_proto_from_weights(self.weights),
                    Descriptor(int_value=self.in_features),
                    Descriptor(int_value=self.out_features),
                    get_proto_from_vector(self.bias),
                    Descriptor(bool_value=self.use_bias),
                    Descriptor(enumeration=EnumDescriptor(enum_name="org.etu.fl.classification.nn.ActivationFunction",
                                                          enum_value_index=1,
                                                          enum_value_name=self.activation_function))
                ]
                ))
            }
        ))
        return proto_layer

    def to_torch_layer(self):
        layer = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)
        # layer.load_state_dict(OrderedDict({'weight': Tensor(self.weights), 'bias': Tensor(self.bias)))
        activation_function = activations[self.activation_function]
        return layer, activation_function

    @staticmethod
    def from_torch_layer(torch_layer, activation_function):
        weights = torch_layer.state_dict['weight'].numpy()
        in_features = torch_layer.in_features
        out_features = torch_layer.out_features
        bias = torch_layer.state_dict['bias'].numpy()
        use_bias = isinstance(torch_layer.bias, Tensor)
        activation_function = activation_function
        return DenseLayer(
            weights=weights,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            use_bias=use_bias,
            # TODO: Придумать способ возвращения названия функции
            activation_function='RELU'
        )

    def get_weights(self, torch_layer):
        self.weights = torch_layer.state_dict()['weight'].numpy()
        self.bias = torch_layer.bias.detach().numpy()
