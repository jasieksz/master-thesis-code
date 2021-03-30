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
def run(C:int, V:int, loadPath:str, subset:int, rangeS:int=0, rangeE:int=0):
    statistics = {}

    if loadPath == "":
        LOGGER.error("Missing input profiles path")
        return {}, []
    else:
        allProfiles = np.load(loadPath)
        LOGGER.warn("Profiles shape = {}, RangeEnd = {}".format(allProfiles.shape, rangeE))

        if rangeS == 0 and rangeE == 0:
            rangeE = allProfiles.shape[0] + 1
        profilesRDD = sc.parallelize(allProfiles[rangeS:rangeE], numSlices=256).cache()

    def deletionSearchMapper(partition):
        env = createGPEnv()
        crDetectionPF = partial(detectCRPropertyWrapper, env)
        for profile in partition:
            yield combinationDeletionSearch(A=profile.A, deleteAxis=1, detectPartialFunction=crDetectionPF)       

    deletionSearchKs = profilesRDD \
        .map(lambda npProf: Profile.fromNumpy(npProf)) \
        .mapPartitions(deletionSearchMapper) \
        .filter(lambda dsr: dsr.status) \
        .map(lambda dsr: (dsr.k, 1)) \
        .cache()

    deletionSearchKCount = deletionSearchKs \
        .reduceByKey(lambda x,y: x + y)

    
    spark.createDataFrame(deletionSearchKCount, ["kCombinations", "count"]) \
        .repartition(1) \
        .write \
        .mode('append') \
        .parquet("resources/output/{}C{}V/deletion/{}-stats".format(C,V,subset))

    return deletionSearchKCount.collect()

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
    path = "resources/output/{}C{}V/NCOP-profiles/ncop-{}{}-{}.npy".format(C, V, C, V, subset)
    start = time()
    stats = run(C=C, V=V, loadPath=path, subset=subset)
    LOGGER.warn("TOTAL Time : " + str(time() - start))
    LOGGER.warn("Stats : " + str(stats))
