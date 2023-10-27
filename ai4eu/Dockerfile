FROM ubuntu:22.04

RUN apt-get update -y 
RUN apt-get install -y python3-pip python3-dev 
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas torch statsmodels scikit-learn
WORKDIR /
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

RUN useradd app
USER app
COPY license-1.0.0.json model.proto model_pb2.py model_pb2_grpc.py server.py predict_utilization.py client.py test_input.csv ./
COPY . /

ENTRYPOINT [ "python3","-u","server.py" ]
