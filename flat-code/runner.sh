#!/bin/zsh

for i in {0..3}
do
    sleep 1
    echo "SET $i =========================================================\n"
    spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 30 30 $i gaussuniform
done

