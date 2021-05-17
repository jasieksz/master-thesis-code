#!/bin/zsh

spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 4 gaussuniform

# for i in {5..8}
# do
#     sleep 1
#     echo "\nSET gaussuniform $i =========================================================\n"
#     spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i gaussuniform
# done

# for i in {4..8}
# do
#     sleep 1
#     echo "\nSET uniformgauss $i =========================================================\n"
#     spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i uniformgauss
# done

#shutdown -P
