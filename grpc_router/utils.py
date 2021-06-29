"""
В этом файле представлен код, реализующий логику
RPC процедур, исполняемых на клиенте
"""
from grpc_router import fl_service_router_pb2
from proto_adapter import MiningSettings, NeuralNetModel
from torch_model.NN_model import train_evaluate


def execute(container, service_id):
    """
    Функция, реализующая удаленную процедуру по исполнению контейнера,
    а именно обучение присланное модели с заданными параметрами
    """
    print(f'start training {service_id}')
    sett = MiningSettings.from_proto(container.settings)
    mdd = NeuralNetModel.from_proto(container.model)
    if container.schedule.algorithm_block.object.fields['sequence'].list.descriptors[0].object.class_name == 'org.etu.fl.classification.nn.NNKaimingInitializerBlock':
        result = mdd.to_torch_model(init=True)
        count = 0
    else:
        path = {
            'nn_client_1': 'D:\FL_client\data\smartilizer\Video-11-12-5-821.csv',
            'nn_client_2': 'D:\FL_client\data\smartilizer\Video-10-15-12-812.csv'
        }
        model = mdd.to_torch_model()
        result, count = train_evaluate(net=model, path=path[service_id], settings=sett,
                                   data_name='smartiliser')
    print('end training')
    mdd.get_weights(result)
    mdd.vectors_count = count
    print(sett)
    return mdd


def get_service_descriptor(name):
    """
    Функция, выполняющая удаленную процедуру,
    которая возвращает описание узла в формате библиотеки fl4j
    """
    port = {
        'nn_client_1': 10002,
        'nn_client_2': 10003
    }
    service_descriptor = fl_service_router_pb2.ObjectDescriptor(
        class_name='org.etu.fl.client.FLClient',
        fields={
            'serviceID': fl_service_router_pb2.Descriptor(string_value=name),
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
                            'port': fl_service_router_pb2.Descriptor(int_value=port[name])
                        }
                    ))
                }
            ))
        })
    return fl_service_router_pb2.Descriptor(object=service_descriptor)
