#!/bin/bash -xe

IMAGE=opendemo.sakuracr.jp/y-egusa/sdxl:latest

docker build -t ${IMAGE} -f Dockerfile.sdxl .
docker push ${IMAGE}
