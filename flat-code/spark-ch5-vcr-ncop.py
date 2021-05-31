#%%
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

#%%
import numpy as np
from time import time
from functools import partial
import sys
import pandas as pd

#%%
from definitions import Profile
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from utils import getNumpyColumns


#%%
def profileAsNumpy(profile): # [ C, V, c1x, c1r, ..., cnx, cnr, v1x, v1r, ..., vmx, vmr, A00, ..., Anm ]
    return np.concatenate([
        np.array(profile.A.shape),
        np.array([(c.x, c.r) for c in profile.C]).flatten(),
        np.array([(v.x, v.r) for v in profile.V]).flatten(),
        profile.A.flatten()])    

#%%
def run(C:int, V:int, inPath:str, outPath:str, rangeS:int=0, rangeE:int=0, R:int=None, distribution:str=None):
    propertyStatus = {0:"ncop", 1:"cr", 2:"vr", 3:"cop"}
    statistics = {}
    candidatesIds = ['C' + str(i) for i in range(C)]
    votersIds = ['V' + str(i) for i in range(V)]

    allProfiles = np.load(inPath)
    if rangeS == 0 and rangeE == 0:
        rangeE = allProfiles.shape[0] + 1
    
    LOGGER.warn("RANGE END = {}".format(rangeE))
        
    profilesRDD = sc.parallelize(allProfiles[rangeS:rangeE], numSlices=48)#.cache()

    def partitionPropertyMapper(partition):
        env = createGPEnv()
        for profile in partition:
            crResult = detectCRProperty(A=profile.A, C=candidatesIds, V=votersIds, env=env)
            vrResult = detectVRProperty(A=profile.A, C=candidatesIds, V=votersIds, env=env)
            status = 3
            if (crResult and not vrResult):
                status = 1
            elif (not crResult and vrResult):
                status = 2
            elif (not crResult and not vrResult):
                status = 0
            yield (status, 1, profile)

    profilesWithPropertyRDD = profilesRDD \
        .map(lambda npProf: Profile.fromNumpy(npProf)) \
        .mapPartitions(partitionPropertyMapper).cache()

    propertyTypeCount = profilesWithPropertyRDD \
        .map(lambda t3: (t3[0], t3[1])) \
        .reduceByKey(lambda a,b: a + b) \
        .map(lambda kv: (propertyStatus[kv[0]], kv[1])) \
        .map(lambda t2: (t2[0], t2[1], distribution, R))

    spark.createDataFrame(propertyTypeCount, ["property", "count", "distribution", "R"]) \
        .coalesce(1) \
        .write \
        .mode('append') \
        .option("header","true") \
        .csv(outPath+'-stats')

    NPRow = Row("rangeS", "rangeE", *tuple(getNumpyColumns(C,V)))
    schema = StructType([StructField("rangeS", IntegerType(), False),
                        StructField("rangeE", IntegerType(), False)] +
                        [StructField(n, FloatType(), False) for n in getNumpyColumns(C, V)])
    
    # t3: (status, 1, profile), status=0 -> ncop
    ncopNumpyRows = profilesWithPropertyRDD \
        .filter(lambda t3: t3[0] == 0) \
        .map(lambda t3: t3[2]) \
        .map(lambda profile: profileAsNumpy(profile).tolist()) \
        .map(lambda profileArr: NPRow(rangeS, rangeE, *tuple(profileArr))) \

    spark.createDataFrame(ncopNumpyRows, schema) \
        .write \
        .mode('append') \
        .parquet(outPath+'-profiles')


def readStatistics(inPath:str):
    return spark.read \
        .option("inferSchema", "true") \
        .option("header", "true") \
        .csv(inPath) \
        .toPandas()

def readNCOPProfiles(inPath:str, takeCount:int):
    return spark.read \
        .option("inferSchema", "true") \
        .parquet(inPath) \
        .rdd.map(lambda a: Profile.fromNumpy(np.array(a[2:]))) \
        .map(lambda p: profileAsNumpy(p)) \
        .take(takeCount)

#%%
if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.getOrCreate()
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    startStr = "=================================== START ========================================\n"
    LOGGER.warn("{}Master : {} | Parallelism : {}".format(startStr, sc.master, sc.defaultParallelism))
    
    C = int(sys.argv[1])
    V = int(sys.argv[2])
    R = int(sys.argv[3])
    distribution = sys.argv[4]

    propertyType = "vcr"
    baseInPath = "resources/random/numpy/{}-{}-{}R-{}C{}V.npy".format(propertyType, distribution, R, C, V)
    baseOutPath = "resources/random/spark/{}C{}V/".format(C,V)
    ncopOutPath = "ncop-{}-{}R".format(distribution, R)

    LOGGER.warn("\nLoading from : {}\nSaving to : {}\n".format(baseInPath, baseOutPath+ncopOutPath))

    start = time()
    run(C=C, V=V, inPath=baseInPath, outPath=baseOutPath+ncopOutPath, rangeS=0, rangeE=0, R=R, distribution=distribution)
    LOGGER.warn("TOTAL Time : " + str(time() - start))
    stats = readStatistics(baseOutPath+ncopOutPath+'-stats')
    LOGGER.warn("Statistics : {}".format(stats))
    profiles = readNCOPProfiles(baseOutPath+ncopOutPath+'-profiles', 1000)
    LOGGER.warn("PROFILES size = {}".format(len(profiles)))
    np.save("resources/random/numpy/ncop/ncop-{}-{}R-{}C{}V.npy".format(distribution, R, C, V), profiles, allow_pickle=False)
    # stats.to_csv("resources/random/pandas-20C20V/{}-{}R-merged.csv".format(distribution, R), index=False, header=True)

