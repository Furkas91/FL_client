# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import fl_service_router_pb2 as fl__service__router__pb2


class FLRouterServiceStub(object):
    """============================================================================
    SERVICES SECTION
    ============================================================================


    List of Router's services -- common abstract methods which every Router
    should provide
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReceiveFLServiceDescriptor = channel.unary_unary(
                '/org.etu.fl.FLRouterService/ReceiveFLServiceDescriptor',
                request_serializer=fl__service__router__pb2.RequestedServiceID.SerializeToString,
                response_deserializer=fl__service__router__pb2.Descriptor.FromString,
                )
        self.ExecuteSchedule = channel.unary_unary(
                '/org.etu.fl.FLRouterService/ExecuteSchedule',
                request_serializer=fl__service__router__pb2.ExecutionContainer.SerializeToString,
                response_deserializer=fl__service__router__pb2.ExecutionResult.FromString,
                )


class FLRouterServiceServicer(object):
    """============================================================================
    SERVICES SECTION
    ============================================================================


    List of Router's services -- common abstract methods which every Router
    should provide
    """

    def ReceiveFLServiceDescriptor(self, request, context):
        """
        Router's method which require a serviveID of desired FLService and expects its
        Descriptor as the response. Should be use for collecting information about
        services and federation resources.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ExecuteSchedule(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FLRouterServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ReceiveFLServiceDescriptor': grpc.unary_unary_rpc_method_handler(
                    servicer.ReceiveFLServiceDescriptor,
                    request_deserializer=fl__service__router__pb2.RequestedServiceID.FromString,
                    response_serializer=fl__service__router__pb2.Descriptor.SerializeToString,
            ),
            'ExecuteSchedule': grpc.unary_unary_rpc_method_handler(
                    servicer.ExecuteSchedule,
                    request_deserializer=fl__service__router__pb2.ExecutionContainer.FromString,
                    response_serializer=fl__service__router__pb2.ExecutionResult.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'org.etu.fl.FLRouterService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FLRouterService(object):
    """============================================================================
    SERVICES SECTION
    ============================================================================


    List of Router's services -- common abstract methods which every Router
    should provide
    """

    @staticmethod
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
        return grpc.experimental.unary_unary(request, target, '/org.etu.fl.FLRouterService/ReceiveFLServiceDescriptor',
            fl__service__router__pb2.RequestedServiceID.SerializeToString,
            fl__service__router__pb2.Descriptor.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
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
        return grpc.experimental.unary_unary(request, target, '/org.etu.fl.FLRouterService/ExecuteSchedule',
            fl__service__router__pb2.ExecutionContainer.SerializeToString,
            fl__service__router__pb2.ExecutionResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
