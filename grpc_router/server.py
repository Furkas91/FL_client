from concurrent import futures
import logging

import grpc

import fl_service_router_pb2
import fl_service_router_pb2_grpc

# map=fl_service_router_pb2.MapDescriptor()
from grpc_router.utils import execute, get_service_descriptor
from proto_adapter import NeuralNetModel, MiningSettings
from torch_model.NN_model import create_nn





class FLRouter(fl_service_router_pb2_grpc.FLRouterService):
    """
    Класс является реализацией паттера роутер, и реализует api для
    общения с остальными участниками сети, созданной библиотекой fl4j
    """
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
        return fl_service_router_pb2.ExecutionResult(model=result.to_proto())

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


def serve(port):
    """
    Функция для запуска сервера
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fl_service_router_pb2_grpc.add_FLRouterServiceServicer_to_server(FLRouter(), server)
    server.add_insecure_port(port)
    server.start()
    print('ready')
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve('localhost:10002')
