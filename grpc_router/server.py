from concurrent import futures
import logging

import grpc

import fl_service_router_pb2
import fl_service_router_pb2_grpc

# map=fl_service_router_pb2.MapDescriptor()
from proto_adapter import NeuralNetModel, MiningSettings
from torch_model.NN_model import create_nn


def execute(container):
    sett = MiningSettings.from_proto(container.settings)
    mdd = NeuralNetModel.from_proto(container.model)
    result = create_nn(mdd.to_torch_model())
    mdd.get_weights(result)
    print(sett)
    return mdd


def get_service_descriptor():
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


class FLRouter(fl_service_router_pb2_grpc.FLRouterService):

    def ExecuteSchedule(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        from proto_adapter import NeuralNetModel
        x = target.model
        # .fields['properties'].list.descriptors[5].enumeration.enum_value_name
        print(type(x))
        # print(x)
        result = execute(target)
        print(result)
        # print(mdd.to_proto())
        return fl_service_router_pb2.ExecutionResult(model=mdd.to_proto())

    def ReceiveFLServiceDescriptor(request,
                                   target,
                                   options=(),
                                   channel_credentials=None,
                                   call_credentials=None,
                                   insecure=False,
                                   compression=None,
                                   wait_for_ready=None,
                                   timeout=None,
                                   metadata=None):
        if target.service_id == 'nn_client':
            print('all good')
        else:
            print('time to cry')
        return get_service_descriptor()
    #    return  fl_service_router_pb2.HelloReply(message='Hello again, %s' % request.name)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fl_service_router_pb2_grpc.add_FLRouterServiceServicer_to_server(FLRouter(), server)
    server.add_insecure_port('localhost:10002')
    server.start()
    print('ready')
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
