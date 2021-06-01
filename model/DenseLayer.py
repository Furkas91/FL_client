from google.protobuf.struct_pb2 import NULL_VALUE

from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor, MapDescriptor, EnumDescriptor
from model.MiningModelElement import MiningModelElement
import numpy as np


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
    def __init__(self, weights, activation_function, number_of_inputs, number_of_outputs, bias, use_bias):
        self.weights = weights
        self.activation_function = activation_function
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.bias = bias
        self.use_bias = use_bias

    @staticmethod
    def from_proto(proto_layer):
        proto_layer = proto_layer.object.fields['properties'].list.descriptors
        weights = get_weights_from_proto(proto_layer[0])
        number_of_inputs = proto_layer[1].int_value
        number_of_outputs = proto_layer[2].int_value
        bias = get_vector_from_proto(proto_layer[3])
        use_bias = proto_layer[4].bool_value
        activation_function = proto_layer[5].enumeration.enum_value_name
        return DenseLayer(
            weights=weights,
            number_of_inputs=number_of_inputs,
            number_of_outputs=number_of_outputs,
            bias=bias,
            use_bias=use_bias,
            activation_function=activation_function
        )

    def to_proto(self):
        proto_layer = Descriptor(object=ObjectDescriptor(
            class_name='org.etu.fl.classification.nn.NNDenseLayerModelElement',
            fields={
                'id': Descriptor(isNull=NULL_VALUE),
                'set': Descriptor(list=ListDescriptor()),
                'properties': Descriptor(list=ListDescriptor(descriptors=[
                    get_proto_from_weights(self.weights),
                    Descriptor(int_value=self.number_of_inputs),
                    Descriptor(int_value=self.number_of_outputs),
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
