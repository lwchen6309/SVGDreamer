#!/bin/bash
# Docker image and container name
IMAGE_NAME="ximingxing/svgrender:v1"
CONTAINER_NAME="svgdreamer"

docker start $CONTAINER_NAME
docker exec -it $CONTAINER_NAME /bin/bash
