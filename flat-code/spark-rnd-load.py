#%%
import numpy as np
from time import time
from generation import parallelGenerateApprovalCombinations
from definitions import Profile
from vcrDetection import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
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
        .parquet("resources/output/{}C{}V/{}-profiles".format(C,V,subSet))

    LOGGER.warn(vcrNCOPProfilesDF.rdd.isEmpty())
    return vcrNCOPProfilesDF.rdd \
        .map(lambda r: np.array(r, dtype=np.float)) \
        .map(lambda r: r[2:]) \
        # .map(lambda npProf: Profile.fromNumpy(npProf)) \

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

    # subset = '*' if int(subSet) == -1 else subset
        
    # loadStatistics(C=int(sys.argv[1]), V=int(sys.argv[2]), subSet=sys.argv[3]).show(n=50, truncate=False)
    loadStatistics(C=int(sys.argv[1]), V=int(sys.argv[2]), subSet=sys.argv[3]).groupBy("key").sum().show()
    # Ps = loadProfiles(C=int(sys.argv[1]), V=int(sys.argv[2]), subSet=sys.argv[3]).take(90)
    # print(len(Ps))

    # for profile in Ps[:10]:
    #     print("SUM 1 : ", sum(sum(profile.A)))
    #     print(profile.A)
    #     for c in profile.C:
    #         print(c)
    #     for v in profile.V:
    #         print(v)
        
    #     print("")

    P = loadProfiles(C=int(sys.argv[1]), V=int(sys.argv[2]), subSet=sys.argv[3]).collect()
    P2 = np.array(P)
    with open("resources/output/5C5V/NCOP-profiles/ncop-55-2.npy", 'wb') as f:
        np.save(file=f, arr=P2, allow_pickle=False)