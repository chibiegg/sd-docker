#!/bin/bash -xe

IMAGE=opendemo.sakuracr.jp/y-egusa/sd15:latest

docker build -t ${IMAGE} -f Dockerfile.sd15 .
docker push ${IMAGE}
