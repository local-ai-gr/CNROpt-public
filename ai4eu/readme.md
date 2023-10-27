# CNROPT AI4EU Submission

This folder contains the code needed for ai4eu experiments submission

## Usage

build docker image

```bash
docker build -t cnropt-utilization-prediction:v1 .
```

run docker container

```bash
docker run -p 8061:8061 --rm -ti cnropt-utilization-prediction:v1 /bin/bash
```

open another terminal and run the client

```bash
python3 client.py
```
