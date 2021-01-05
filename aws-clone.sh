#!/bin/bash

wget https://packages.gurobi.com/9.1/gurobi9.1.1_linux64.tar.gz

echo "Downloaded GUROBI"

git clone https://github.com/jasieksz/master-thesis-code.git

aws s3 cp --recursive s3://jsz-vcr/master-thesis-code master-thesis-code/

scp -i apache-micro-instance.pem hadoop@ec2-54-197-14-119.compute-1.amazonaws.com:/home/hadoop/gurobi.lic .


tar xvfz gurobi9.0.1_linux64.tar.gz
export GUROBI_HOME="/home/hadoop/gurobi911/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

sudo python -m pip install -i https://pypi.gurobi.com gurobipy