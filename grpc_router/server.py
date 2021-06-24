"""
В этом файле предаставлен код сервера gRPC, который отвечает
за непосредственную реализацию API данного FL клиента.
"""

from concurrent import futures
import logging
from multiprocessing import Process

import grpc

import fl_service_router_pb2
import fl_service_router_pb2_grpc

# map=fl_service_router_pb2.MapDescriptor()
from utils import execute, get_service_descriptor
from proto_adapter import NeuralNetModel, MiningSettings
from torch_model.NN_model import train_evaluate


class FLRouter(fl_service_router_pb2_grpc.FLRouterService):
    """
    Класс является реализацией паттерна роутер, и реализует api для
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

        result = execute(target, request.service_id)
        # print(result)
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
        print(target.service_id)
        request.service_id = target.service_id
        return get_service_descriptor(target.service_id)


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
    p = Process(target=serve, args=('localhost:10002',))
    d = Process(target=serve, args=('localhost:10003',))
    d.start()
    p.start()
    p.join()
    d.join()
    # serve('localhost:10003')
