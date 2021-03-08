#%%
import numpy as np
from time import time
from generation import parallelGenerateApprovalCombinations
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from utils import getNumpyColumns
from functools import partial
import sys

#%%
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

# conf=SparkConf()
# conf.set("spark.executor.memory", "6g")
# conf.set("spark.driver.memory", "6g")
# conf.set("spark.cores.max", "16")


#%%
def run(C:int=3, V:int=3, loadPath:str="", subset:int=0, rangeS:int=0, rangeE:int=0):
    statistics = {}
    candidatesIds = ['c'+str(i) for i in range(C)]
    votersIds = [str(i) for i in range(V)]

    if loadPath == "":
        LOGGER.error("Missing input profiles path")
        return {}, []
    else:
        allProfiles = np.load(loadPath)
        # LOGGER.warn("Profiles SHAPE : " + str(allProfiles.shape))

        if rangeS == 0 and rangeE == 0:
            rangeE = allProfiles.shape[0] + 1
        profilesRDD = sc.parallelize(allProfiles[rangeS:rangeE], numSlices=256).cache()

    def partitionCOPFilter(partition):
        env = createGPEnv()
        for profile in partition:
            yield ((detectCRProperty(A=profile.A, C=candidatesIds, V=votersIds, env=env) or detectVRProperty(A=profile.A, C=candidatesIds, V=votersIds, env=env)), profile)

    vcrNCOPProfilesRDD = profilesRDD \
        .map(lambda npProf: Profile.fromNumpy(npProf)) \
        .mapPartitions(partitionCOPFilter) \
        .filter(lambda COPpRes: not COPpRes[0]) \
        .map(lambda COPpRes: COPpRes[1]) 

    NPRow = Row("rangeS", "rangeE", *tuple(getNumpyColumns(C,V)))
    schema = StructType([StructField("rangeS", IntegerType(), False),
                         StructField("rangeE", IntegerType(), False)] +
                        [StructField(n, FloatType(), False) for n in getNumpyColumns(C, V)])
    
    vcrNCOPNumpyRows = vcrNCOPProfilesRDD \
        .map(lambda profile: profile.asNumpy().tolist()) \
        .map(lambda a: NPRow(rangeS, rangeE, *tuple(a))) \
    
    statistics["VCR"] = rangeE - rangeS # TODO TEMPORARY count
    LOGGER.warn("VCR " + str(statistics["VCR"]))
        
    statistics["NCOPVCR"] = vcrNCOPProfilesRDD.count()
    LOGGER.warn("NCOP VCR " + str(statistics["NCOPVCR"]))
    
    spark.createDataFrame(statistics.items(), ["key", "value"]) \
        .repartition(1) \
        .write \
        .mode('append') \
        .parquet("resources/output/{}C{}V/{}-stats".format(C,V,subset))
        
    spark.createDataFrame(vcrNCOPNumpyRows, schema) \
        .write \
        .mode('append') \
        .parquet("resources/output/{}C{}V/{}-profiles".format(C,V,subset)) \
        
    return statistics, vcrNCOPNumpyRows

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
    path = "" if len(sys.argv) == 3 else "resources/input/{}C{}V/VCR-{}.npy".format(C, V, sys.argv[3])
    subset = 0 if len(sys.argv) == 3 else int(sys.argv[3])
    start = time()
    stats, vcrNCOPProfiles = run(C=C, V=V, loadPath=path, subset=subset, rangeS=100000, rangeE=300000)
    LOGGER.warn("TOTAL Time : " + str(time() - start))
    LOGGER.warn("Stats : " + str(stats))
