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
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


# %%
def loadProfiles(C:int, V:int, subSet='*'):
    schema = StructType([StructField("rangeS", IntegerType(), False),
                         StructField("rangeE", IntegerType(), False)] +
                        [StructField(n, FloatType(), False) for n in getNumpyColumns(C, V)])

    vcrNCOPProfilesDF = spark.read \
        .option("mergeSchema", "true") \
        .parquet("resources/random/spark/{}C{}V/vr-{}S-profiles".format(C,V,subSet))

    return vcrNCOPProfilesDF.rdd \
        .map(lambda r: np.array(r, dtype=float)) \
        .map(lambda r: r[2:]) \

#%%
def loadStatistics(C:int, V:int, subSet='*'):
    return spark.read \
        .parquet("resources/output/{}C{}V/{}-stats".format(C,V,subSet))   

def printer(profile:Profile) -> str:
    res = str(profile.A)
    res += "\n"
    for c in profile.C:
        res += c.shortPrint() + "\n"
    for v in profile.V:
        res += v.shortPrint() + "\n"
    return res

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

    C=int(sys.argv[1])
    V=int(sys.argv[2])
    subSet=int(sys.argv[3])
    P = loadProfiles(C=C, V=V, subSet=subSet).collect()
    P2 = np.array(P)
    with open("resources/random/numpy/vr-{}C{}V-{}S.npy".format(C,V,subSet), 'wb') as f:
        np.save(file=f, arr=P2, allow_pickle=False)