#!/bin/bash

sudo docker build -t name . && sudo docker run -p 8888:8888 -v "$(pwd)":/root/working/ -it name
