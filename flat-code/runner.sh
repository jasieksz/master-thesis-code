#!/bin/zsh

for i in {7..8}
do
    sleep 1
    echo "\nSET uniform $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i uniform
done

for i in {4..8}
do
    sleep 1
    echo "\nSET 2gauss $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i 2gauss
done

for i in {4..8}
do
    sleep 1
    echo "\nSET gaussuniform $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i gaussuniform
done

for i in {4..8}
do
    sleep 1
    echo "\nSET uniformgauss $i =========================================================\n"
    spark-submit --master 'local[14]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i uniformgauss
done

