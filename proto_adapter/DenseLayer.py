"""
В этом файле описан код для сериализации и
десериализации линейного слоя нейронной сети
"""
from collections import OrderedDict

from torch import nn, Tensor

from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor, EnumDescriptor
import torch.nn.functional as F
import numpy as np

from proto_adapter import Adapter

activations = {
    'RELU': F.relu,
    'SOFTMAX': F.softmax
}
def get_vector_from_proto(proto_vector):
    """
    Функция возвращает вектор numpy описанный протобафом
    :param proto_vector: дескриптор
    :return: numpy вектор
    """
    vector = []
    for elem in proto_vector.list.descriptors:
        vector.append(elem.double_value)
    return np.asarray(vector)


def get_proto_from_vector(vector):
    """
    Функция преобразует вектор в протобаф объект
    :param vector: любой список чисел (например np.array)
    :return: дескриптор
    """
    return Descriptor(list=ListDescriptor(
        descriptors=[Descriptor(double_value=x) for x in vector]))


def get_weights_from_proto(proto_weights):
    """
    Функция, для получения матрицы весов из дескриптора
    :param proto_weights: дескриптор
    :return: np.array с матрицей весов
    """
    weights = []
    for proto_vector in proto_weights.list.descriptors:
        weights.append(get_vector_from_proto(proto_vector))
    return np.asarray(weights)


def get_proto_from_weights(weights):
    """
    Функция, для преобразования матрицы весов в дескриптор
    :param weights: матрица весов
    :return: дескриптор
    """
    return Descriptor(list=ListDescriptor(
        descriptors=[get_proto_from_vector(vector) for vector in weights]
    ))


class DenseLayer(Adapter):
    """
    Класс сериализации линейного слоя нейронной сети
    """
    def __init__(self, weights, activation_function, in_features, out_features, bias, use_bias):
        self.weights = weights
        self.activation_function = activation_function
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.use_bias = use_bias

    @staticmethod
    def from_proto(proto_layer):
        """
        Метод для преобразования дескриптора в DenseLayer
        :param proto_layer: дескриптор
        :return: DenseLayer
        """
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
        """
        Метод для преобразования DenseLayer в дескриптор
        :return: дескриптор
        """
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

    def to_torch_layer(self, init):
        """
        Метод для преобразования DenseLayer в слой PyTorch
        :return: слой PyTorch, функция активации PyTorch
        """
        layer = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)
        # Загрузка весов в слой PyTorch
        if not init:
            layer.load_state_dict(OrderedDict({'weight': Tensor(self.weights), 'bias': Tensor(self.bias)}))
        activation_function = activations[self.activation_function]
        return layer, activation_function

    @staticmethod
    def from_torch_layer(torch_layer, activation_function):
        """
        Метод для преобразования PyTorch слоя в DenseLayer
        :param torch_layer: PyTorch слой
        :param activation_function: PyTorch функция активации
        :return: DenseLayer
        """
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
        """
        Метод для обновления весов DenseLayer из PyTorch слоя
        :param torch_layer: PyTorch слой
        """
        self.weights = torch_layer.state_dict()['weight'].numpy()
        self.bias = torch_layer.bias.detach().numpy()
