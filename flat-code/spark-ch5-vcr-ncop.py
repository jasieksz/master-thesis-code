#%%
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

#%%
import numpy as np
from time import time
from functools import partial
import sys

#%%
from definitions import Profile
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from utils import getNumpyColumns


#%%
def run(C:int, V:int, inPath:str, outPath:str, rangeS:int=0, rangeE:int=0):
    propertyStatus = {0:"ncop", 1:"cr", 2:"vr", 3:"cop"}
    statistics = {}
    candidatesIds = ['C' + str(i) for i in range(C)]
    votersIds = ['V' + str(i) for i in range(V)]

    allProfiles = np.load(inPath)
    if rangeS == 0 and rangeE == 0:
        rangeE = allProfiles.shape[0] + 1
        
    profilesRDD = sc.parallelize(allProfiles[rangeS:rangeE], numSlices=128)#.cache()

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
        .mapPartitions(partitionPropertyMapper)#.cache()


    propertyTypeCount = profilesWithPropertyRDD \
        .map(lambda t3: (t3[0], t3[1])) \
        .reduceByKey(lambda a,b: a + b) \
        .map(lambda kv: (propertyStatus[kv[0]], kv[1]))

    # spark.createDataFrame(propertyTypeCount, ["property", "count"]) \
    #     .coalesce(1) \
    #     .write \
    #     .mode('append') \
    #     .csv(outPath)

    # NPRow = Row("rangeS", "rangeE", *tuple(getNumpyColumns(C,V)))
    # schema = StructType([StructField("rangeS", IntegerType(), False),
    #                     StructField("rangeE", IntegerType(), False)] +
    #                     [StructField(n, FloatType(), False) for n in getNumpyColumns(C, V)])
    
    # vcrNCOPNumpyRows = vcrNCOPProfilesRDD \
    #     .map(lambda profile: profile.asNumpy().tolist()) \
    #     .map(lambda a: NPRow(rangeS, rangeE, *tuple(a))) \

    return propertyTypeCount.collect()

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
    subset = int(sys.argv[3])
    distribution = sys.argv[4]

    propertyType = "vcr"
    baseInPath = "resources/random/numpy/{}-{}-{}C{}V-{}S.npy".format(propertyType, distribution, C, V, subset)
    baseOutPath = "resources/random/spark/{}C{}V/".format(C,V)
    ncopOutPath = "ncop-{}S-profiles".format(subset)

    LOGGER.warn("\nLoading from : {}\nSaving to : {}\n".format(baseInPath, baseOutPath+ncopOutPath))

    start = time()
    stats = run(C=C, V=V, inPath=baseInPath, outPath=baseOutPath+ncopOutPath, rangeS=0, rangeE=10000)
    LOGGER.warn("Stats : " + str(stats))
    LOGGER.warn("TOTAL Time : " + str(time() - start))
