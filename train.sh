#!/bin/sh
docker run -dit --rm \
	-v "$(pwd)":/tf \
	-w /tf \
	-u $(id -u):$(id -g) \
	tensorflow/tensorflow:latest \
	python ./train.py
