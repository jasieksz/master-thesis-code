#!/bin/zsh

for i in {17..31}
do
    sleep 1
    echo "SET $i =========================================================\n"
    spark-submit --master 'local[4]' --driver-memory 8G --executor-memory 768M main.py 4 6 $i
done
