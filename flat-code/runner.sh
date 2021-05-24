#!/bin/zsh

sleep 1
echo "\nSET gaussuniform 8 =========================================================\n"
spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 8 gaussuniform
sleep 1
echo "\nSET uniformgauss 8 =========================================================\n"
spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 8 uniformgauss
sleep 1
echo "\nSET uniformgauss 4  =========================================================\n"
spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 4 uniformgauss
sleep 1
echo "\nSET gaussuniform 7 =========================================================\n"
spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 7 gaussuniform
sleep 1
echo "\nSET uniformgauss 7 =========================================================\n"
spark-submit --master 'local[16]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 7 uniformgauss




# for i in {5..8}
# do
#    sleep 1
#    echo "\nSET gaussuniform $i =========================================================\n"
#    spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i gaussuniform
# done

# for i in {4..8}
# do
#     sleep 1
#     echo "\nSET uniformgauss $i =========================================================\n"
#     spark-submit --master 'local[10]' --driver-memory 8G --executor-memory 768M spark-ch5-vcr-ncop.py 40 40 $i uniformgauss
# done

shutdown -P
