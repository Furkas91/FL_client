"""
В этом файле описан код для сериализации и
десериализации слоя нормализации нейронной сети
"""
from google.protobuf.struct_pb2 import NULL_VALUE

from proto_adapter.Adapter import Adapter
from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor


class NormalizationLayer(Adapter):
    """
    Класс для сериализации слоя нормализации нейронной сети
    """
    @staticmethod
    def from_proto(proto_layer: Descriptor):
        """
        Метод для возвращения NormalizationLayer
        :return: NormalizationLayer
        """
        return NormalizationLayer()

    def to_proto(self) -> Descriptor:
        """
        Метод для преобразования NormalizationLayer в дескриптор
        :return: дескриптор
        """
        proto_layer = Descriptor(object=ObjectDescriptor(
            class_name='org.etu.fl.classification.nn.NNBatchLayerModelElement',
            fields={
                'id': Descriptor(isNull=NULL_VALUE),
                'properties': Descriptor(list=ListDescriptor()),
                'set': Descriptor(list=ListDescriptor())
            }
        ))
        return proto_layer
