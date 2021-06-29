"""
В этом файле описан код для сериализации и
десериализации нейронной сети
"""
from google.protobuf.struct_pb2 import NULL_VALUE
from torch import nn

from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor, MapDescriptor
from proto_adapter import DenseLayer, NormalizationLayer, Adapter
from torch_model.NN_model import UniversalNet

# Соответствия fl4j классов и FLPyTorchClient
layer_type = {
    'org.etu.fl.classification.nn.NNDenseLayerModelElement': DenseLayer,
    'org.etu.fl.classification.nn.NNBatchLayerModelElement': NormalizationLayer
}


class NeuralNetModel(Adapter):
    """
    Класс сериализации нейронной сети
    """
    def __init__(self,
                 algorithm_name: str,
                 layers: list,
                 vectors_count: int):
        self.algorithm_name = algorithm_name
        self.layers = layers
        self.vectors_count = vectors_count

    @staticmethod
    def from_proto(proto_model: Descriptor):
        """
        Метод для преобразования дескриптора в NeuralNetModel
        :param proto_model: дескриптор
        :return: NeuralNetModel
        """
        proto_model = proto_model.object.fields
        algorithm_name = proto_model['algorithmName']
        vectors_count = proto_model['consumedVectorsCount'].long_value
        layers = []
        # Цикл для добавления десериализованный слоев в список
        for layer in proto_model['sets'].list.descriptors:
            layers.append(layer_type[layer.object.class_name].from_proto(layer))
        return NeuralNetModel(algorithm_name=algorithm_name,
                              layers=layers,
                              vectors_count=vectors_count)

    def to_proto(self) -> Descriptor:
        """
        Метод для преобразования NeuralNetModel в дескриптор
        :return: дескриптор
        """
        proto_model = Descriptor(object=ObjectDescriptor(
            class_name='org.etu.fl.classification.nn.NNModel',
            fields={
                'algorithmName': Descriptor(isNull=NULL_VALUE),
                'consumedVectorsCount': Descriptor(long_value=self.vectors_count),
                'sets': Descriptor(list=ListDescriptor(descriptors=[a.to_proto() for a in self.layers]))
            }
        ))
        return proto_model

    def to_torch_model(self, init=False) -> UniversalNet:
        """
        Метод для преобразования NeuralNetModel в UniversalNet
        :return: UniversalNet
        """
        layers = []
        activations = []
        normalizations = []
        last_features = 0
        has_normalization = True
        for layer in self.layers:
            # Проверка на слой нормализации
            if isinstance(layer, NormalizationLayer):
                normalizations.append(nn.BatchNorm1d(last_features))
                has_normalization = True
            else:
                torch_layer = layer.to_torch_layer(init)
                layers.append(torch_layer[0])
                activations.append((torch_layer[1]))
                last_features = layer.out_features
                # Если не было слоя нормализации после слоя нейронной сети
                # функция нормализации возвращает полученное значение
                if not has_normalization:
                    normalizations.append(lambda x: x)
                has_normalization = False
        normalizations.append(lambda x: x)
        return UniversalNet(layers=layers, activations=activations, normalizations=normalizations)

    def get_weights(self, torch_model: UniversalNet) -> None:
        """
        Метод для обновления весов из UniversalNet
        :param torch_model: UniversalNet
        """
        i = 0
        for layer in self.layers:
            # проверка на слой нормализации, так как он не имеет весов
            if not isinstance(layer, NormalizationLayer):
                layer.get_weights(torch_model.layers[i])
                i += 1
