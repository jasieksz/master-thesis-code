#!/bin/zsh

sleep 1
echo "\nSET uniformgauss 4 =========================================================\n"
spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 15 15 4 uniformgauss

sleep 1
echo "\nSET uniformgauss 8 =========================================================\n"
spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 15 15 8 uniformgauss

sleep 1
echo "\nSET gaussuniform 4 =========================================================\n"
spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 15 15 4 gaussuniform

sleep 1
echo "\nSET gaussuniform 8 =========================================================\n"
spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 15 15 8 gaussuniform

# for i in {4..8}
# do
#    sleep 1
#    echo "\nSET uniform $i =========================================================\n"
#    spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 15 15 $i uniform
# done

# for i in {4..8}
# do
#    sleep 1
#    echo "\nSET 2gauss $i =========================================================\n"
#    spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 15 15 $i 2gauss
# done
# 
# for i in {4..8}
# do
#    sleep 1
#    echo "\nSET uniformgauss $i =========================================================\n"
#    spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i uniformgauss
# done

# for i in {4..8}
# do
#    sleep 1
#    echo "\nSET gaussuniform $i =========================================================\n"
#    spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 20 20 $i gaussuniform
# done
