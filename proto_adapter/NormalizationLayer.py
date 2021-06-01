from Tools.scripts import google
from google.protobuf.struct_pb2 import NULL_VALUE

from proto_adapter.MiningModelElement import MiningModelElement
from grpc_router.fl_service_router_pb2 import Descriptor, ObjectDescriptor, ListDescriptor


class NormalizationLayer1(MiningModelElement):
    @staticmethod
    def from_proto(proto_layer):
        return NormalizationLayer1()

    def to_proto(self):
        proto_layer = Descriptor(object=ObjectDescriptor(
            class_name='org.etu.fl.classification.nn.NNBatchLayerModelElement',
            fields={
                'id': Descriptor(isNull=NULL_VALUE),
                'properties': Descriptor(list=ListDescriptor()),
                'set': Descriptor(list=ListDescriptor())
            }
        ))
        return proto_layer
