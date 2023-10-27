import grpc
from concurrent import futures
import time
# import the generated classes :
import model_pb2
import model_pb2_grpc
# import the function we made :
import predict_utilization as psp

port = 8061

# create a class to define the server functions, derived from
class PredictServicer(model_pb2_grpc.PredictServicer):
  def predict_utilization(self, request, context):
    # define the buffer of the response :
    response = model_pb2.Prediction()
    # get the value of the response by calling the desired function :
    response.predictedValues[:] = psp.predict_utilization(request.utilization)
    return response
  
# create a grpc server :
server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10))
model_pb2_grpc.add_PredictServicer_to_server(PredictServicer(), server)
print("Starting server. Listening on port : " + str(port))
server.add_insecure_port("[::]:{}".format(port))
server.start()
try:
  while True:
    time.sleep(86400)
    
except KeyboardInterrupt:
    server.stop(0)
