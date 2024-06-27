#!/bin/bash -xe

IMAGE=opendemo.sakuracr.jp/y-egusa/sdxl:latest

docker build -t ${IMAGE} -f Docker.sdxl .
docker push ${IMAGE}
