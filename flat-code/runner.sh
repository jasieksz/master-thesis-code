#!/bin/zsh

for i in {4..8}
do
    sleep 1
    echo "SET uniform $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i uniform
done

for i in {4..8}
do
    sleep 1
    echo "SET 2gauss $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i 2gauss
done

for i in {4..8}
do
    sleep 1
    echo "SET gaussuniform $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i gaussuniform
done

for i in {4..8}
do
    sleep 1
    echo "SET uniformgauss $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i uniformgauss
done

