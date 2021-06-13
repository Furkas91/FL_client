from grpc_router import fl_service_router_pb2
from proto_adapter import MiningSettings, NeuralNetModel
from torch_model.NN_model import create_nn


def execute(container):
    """
    Функция, реализующая удаленную процедуру по исполнению контейнера,
    а именно обучение присланное модели с заданными параметрами
    """
    sett = MiningSettings.from_proto(container.settings)
    mdd = NeuralNetModel.from_proto(container.model)
    result = create_nn(net=mdd.to_torch_model(), path='D:\FL_client\data\smartilizer\Video-11-15-40-560.csv',
                       data_name='smartiliser')
    mdd.get_weights(result)
    print(sett)
    return mdd


def get_service_descriptor():
    """
    Функция, выполняющая удаленную процедуру,
    которая возвращает описание узла в формате библиотеки fl4j
    """
    service_descriptor = fl_service_router_pb2.ObjectDescriptor(
        class_name='org.etu.fl.client.FLClient',
        fields={
            'serviceID': fl_service_router_pb2.Descriptor(string_value='nn_client'),
            'physical_data': fl_service_router_pb2.Descriptor(map=fl_service_router_pb2.MapDescriptor()),
            'router': fl_service_router_pb2.Descriptor(object=fl_service_router_pb2.ObjectDescriptor(
                class_name='org.etu.fl.router.GrpcRouter',
                fields={
                    'message_format': fl_service_router_pb2.Descriptor(string_value='proto3'),
                    'transmission_protocol': fl_service_router_pb2.Descriptor(string_value='HTTP/2'),
                    'url': fl_service_router_pb2.Descriptor(object=fl_service_router_pb2.ObjectDescriptor(
                        class_name='org.etu.fl.core.utils.URL',
                        fields={
                            'host': fl_service_router_pb2.Descriptor(string_value='127.0.0.1'),
                            'port': fl_service_router_pb2.Descriptor(int_value=10002)
                        }
                    ))
                }
            ))
            #    'children': fl_service_router_pb2.Descriptor(),
            #    'workers': fl_service_router_pb2.Descriptor()
        })
    return fl_service_router_pb2.Descriptor(object=service_descriptor)