#!/bin/bash -xe

IMAGE=opendemo.sakuracr.jp/y-egusa/sd15-base:latest

docker build -t ${IMAGE} -f Dockerfile.sd15.base .
docker push ${IMAGE}
