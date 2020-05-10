#!/bin/bash

set -e

# initial check

if [ "$#" -lt 1 ]; then
    echo "$# parameters given. At least 1 parameter is expected. Use -h to view command format"
    exit 1
fi

if [ "$#" -gt 3 ]; then
    echo "$# parameters given. At most 3 parameters are expected. Use -h to view command format"
    exit 1
fi

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` <file to evaluate upon> [<image name>, <evaluation script>]"
  exit 1
fi

test_path=$1
image_name=${2:-nlp2020-hw2}
evaluation_script=${3:-hw2/evaluate.py}

# delete old docker if exists
docker ps -q --filter "name=$image_name" | grep -q . && docker stop $image_name
docker ps -aq --filter "name=$image_name" | grep -q . && docker rm $image_name

# build docker file
docker build . -f Dockerfile -t $image_name

# bring model up
docker run -d -p 12345:12345 --name $image_name $image_name

# perform evaluation
/usr/bin/env python $evaluation_script $test_path

# stop container
docker stop $image_name

# dump container logs
docker logs -t $image_name > logs/server.stdout 2> logs/server.stderr

# remove container
docker rm $image_name