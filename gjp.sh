#!/bin/sh
docker run -it \
	-p 8888:8888 \
	-v "$(pwd)":/tf \
	-u $(id -u):$(id -g) \
	tensorflow/tensorflow:latest-gpu-jupyter
