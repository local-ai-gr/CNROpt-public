# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import model_pb2 as model__pb2


class PredictStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.predict_utilization = channel.unary_unary(
                '/Predict/predict_utilization',
                request_serializer=model__pb2.Input.SerializeToString,
                response_deserializer=model__pb2.Prediction.FromString,
                )


class PredictServicer(object):
    """Missing associated documentation comment in .proto file."""

    def predict_utilization(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PredictServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'predict_utilization': grpc.unary_unary_rpc_method_handler(
                    servicer.predict_utilization,
                    request_deserializer=model__pb2.Input.FromString,
                    response_serializer=model__pb2.Prediction.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Predict', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Predict(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def predict_utilization(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Predict/predict_utilization',
            model__pb2.Input.SerializeToString,
            model__pb2.Prediction.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
