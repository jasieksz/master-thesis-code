#%%
import numpy as np
from time import time
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import createGPEnv
from ch7_main import combinationDeletionSearch, detectCRPropertyWrapper, detectVRPropertyWrapper, deletionSearchResults
from utils import getNumpyColumns
from functools import partial
import sys

#%%
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

#%%
def loadStatistics(C:int, V:int, subset:int):
    return spark.read \
        .parquet("resources/output/{}C{}V/deletion/{}-stats".format(C,V,subset))

#%%
if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    # spark.sparkContext.setLogLevel('WARN')
    sc.setLogLevel("WARN")
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.warn("=================================== START ========================================")
    LOGGER.warn(sc.master)
    LOGGER.warn(sc.defaultParallelism)    
    
    C = int(sys.argv[1])
    V = int(sys.argv[2])
    subset = int(sys.argv[3])
    
    start = time()
    loadStatistics(C, V, subset).show()

    LOGGER.warn("TOTAL Time : " + str(time() - start))
