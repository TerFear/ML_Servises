# cmd/main.py

from concurrent import futures
import grpc
from internal.api.grpc.server import MLModelServicer
import gen.py.ml_pb2_grpc as ml_pb2_grpc
import logging


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_pb2_grpc.add_MLModelServicer_to_server(MLModelServicer(), server)
    server.add_insecure_port('[::]:50052')
    logging.basicConfig(level=logging.INFO)
    logging.info("gRPC сервер запущен на порту 50052...")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()