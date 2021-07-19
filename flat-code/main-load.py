#%%
import numpy as np
from time import time
from generation import parallelGenerateApprovalCombinations
from definitions import Profile
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from utils import getNumpyColumns
from functools import partial
import sys

#%%
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType


# %%
def loadProfiles(C:int=3, V:int=3, subSet='*'):
    schema = StructType([StructField("rangeS", FloatType(), False),
                        StructField("rangeE", FloatType(), False)] +
                    [StructField(n, FloatType(), False) for n in getNumpyColumns(C, V)])

    vcrNCOPProfilesDF = spark.read \
        .option("mergeSchema", "true") \
        .parquet("resources/output/{}C{}V/{}-{}C{}V".format(C, V, subSet, C,V))

    return vcrNCOPProfilesDF.rdd \
        .map(lambda r: np.array(r, dtype=np.float)) \
        .map(lambda npProf: Profile.fromNumpy(npProf[2:])) \

def loadStatistics(C:int=3, V:int=3, subSet='*'):
    return spark.read \
        .parquet("resources/output/{}C{}V/{}-{}C{}V-stats".format(C, V, subSet, C,V))   

# %%
if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Incorrect arguments, usage : spark-submit ... main-load.py C V subSet")
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    sc.setLogLevel("WARN")
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.warn("=================================== START ========================================")
    LOGGER.warn(sc.master)
    LOGGER.warn(sc.defaultParallelism)
    loadStatistics(C=int(sys.argv[1]), V=int(sys.argv[2]), subSet=sys.argv[3]).show(n=50, truncate=False)
    loadStatistics(C=int(sys.argv[1]), V=int(sys.argv[2]), subSet=sys.argv[3]).groupBy("key").sum().show()
    LOGGER.warn((str(loadProfiles(C=int(sys.argv[1]), V=int(sys.argv[2]), subSet=sys.argv[3]).take(1))))




#%%
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

#%%
P44 = loadProfiles(C=4, V=4, subSet=0)

#%%
ncopP = P44.take(1)[0]

#%%
