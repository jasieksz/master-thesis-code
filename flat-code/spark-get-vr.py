#%%
import numpy as np
from time import time
from generation import parallelGenerateApprovalCombinations
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectVCRProperty, detectVRProperty, createGPEnv
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
def run(C:int, V:int, loadPath:str="", subset:int=0, rangeS:int=0, rangeE:int=0):
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

    def partitionVRFilter(partition):
        env = createGPEnv()
        for profile in partition:
            yield (detectVRProperty(A=profile.A, C=candidatesIds, V=votersIds, env=env), profile)

    VRProfilesRDD = profilesRDD \
        .map(lambda npProf: Profile.fromNumpy(npProf)) \
        .mapPartitions(partitionVRFilter) \
        .filter(lambda VrProf: VrProf[0]) \
        .map(lambda VrProf: VrProf[1])
            
    NPRow = Row("rangeS", "rangeE", *tuple(getNumpyColumns(C,V)))
    schema = StructType([StructField("rangeS", IntegerType(), False),
                         StructField("rangeE", IntegerType(), False)] +
                        [StructField(n, FloatType(), False) for n in getNumpyColumns(C, V)])
    
    VRNumpyRows = VRProfilesRDD \
        .map(lambda profile: profile.asNumpy().tolist()) \
        .map(lambda a: NPRow(rangeS, rangeE, *tuple(a))) \
            
    spark.createDataFrame(VRNumpyRows, schema) \
        .write \
        .mode('append') \
        .parquet("resources/random/spark/{}C{}V/vr-{}S-profiles".format(C,V,subset)) \
        
    return VRNumpyRows

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
    path = "resources/random/numpy/vcr-{}C{}V-{}S.npy".format(C, V, subset)
    LOGGER.warn("Using PATH: {}".format(path))

    start = time()
    vrProfiles = run(C=C, V=V, loadPath=path, subset=subset, rangeS=0, rangeE=10000)
    LOGGER.warn("TOTAL Time : " + str(time() - start))
    LOGGER.warn("Stats : " + str(vrProfiles.count()))
