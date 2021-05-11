#!/bin/zsh

echo "SET -2\n"
spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 1 uniform
echo "SET -1\n"
spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 2 uniform

for i in {0..3}
do
    sleep 1
    echo "SET 2gauss $i =========================================================\n"
    spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i 2gauss
done

for i in {0..3}
do
    sleep 1
    echo "SET gaussuniform $i =========================================================\n"
    spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i gaussuniform
done

for i in {0..3}
do
    sleep 1
    echo "SET uniformgauss $i =========================================================\n"
    spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i uniformgauss
done

