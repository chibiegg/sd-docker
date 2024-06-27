#!/bin/bash -xe

IMAGE=opendemo.sakuracr.jp/y-egusa/sdxl-base:latest

docker build -t ${IMAGE} -f Dockerfile.sdxl.base .
docker push ${IMAGE}
