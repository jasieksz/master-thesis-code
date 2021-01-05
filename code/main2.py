# %%
import numpy as np
from time import time
from generation import parallelGenerateApprovalCombinations
from definitions import Profile
from vcrDetection import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from utils import getNumpyColumns
from functools import partial
import sys

# %%
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


# conf=SparkConf()
# conf.set("spark.executor.memory", "6g")
# conf.set("spark.driver.memory", "6g")
# conf.set("spark.cores.max", "16")


# %%
def run(C: int = 3, V: int = 3, loadPath: str = "", rangeS: int = 0, rangeE: int = 0):
    statistics = {}
    candidatesIds = ["A", "B", "C", "D", "E", "F", "G"][:C]
    votersIds = ["0", "1", "2", "3", "4", "5", "6"][:V]

    def partitionVCRDetect(partition):
        env = createGPEnv()
        for p in partition:
            yield (p, detectVCRProperty(A=p, C=candidatesIds, V=votersIds))

    if loadPath == "":
        allProfiles = parallelGenerateApprovalCombinations(cpu=16, C=C, V=V)
        if rangeS == 0 and rangeE == 0:
            rangeE = allProfiles.shape[0] + 1
        profilesRDD = sc.parallelize(allProfiles[rangeS:rangeE]).df.cache()
    else:
        allProfiles = np.load(loadPath)
        if rangeS == 0 and rangeE == 0:
            rangeE = allProfiles.shape[0] + 1
        profilesRDD = sc.parallelize(allProfiles[rangeS:rangeE]).cache()

    vcrProfilesRDD = profilesRDD.map(lambda p: np.array(p).reshape(C, V)) \
        .mapPartitions(partitionVCRDetect) \
        .filter(lambda pRes: pRes[1][0] == 2).cache()

    def partitionCOPFilter(partition):
        env = createGPEnv()
        for pRes in partition:
            yield ((detectCRProperty(A=pRes[0], C=candidatesIds, V=votersIds, env=env) or
                    detectVRProperty(A=pRes[0], C=candidatesIds, V=votersIds, env=env)), pRes)

    vcrNCOPProfilesRDD = vcrProfilesRDD \
        .mapPartitions(partitionCOPFilter) \
        .filter(lambda COPpRes: not COPpRes[0]) \
        .map(lambda COPpRes: COPpRes[1]) \
        .map(lambda pRes: Profile.fromILPRes(pRes[0], pRes[1][1], candidatesIds, votersIds))

    NPRow = Row("rangeS", "rangeE", *tuple(getNumpyColumns(C, V)))
    schema = StructType([StructField("rangeS", IntegerType(), False),
                         StructField("rangeE", IntegerType(), False)] +
                        [StructField(n, FloatType(), False) for n in getNumpyColumns(C, V)])

    vcrNCOPNumpyRows = vcrNCOPProfilesRDD \
        .map(lambda profile: profile.asNumpy().tolist()) \
        .map(lambda a: NPRow(rangeS, rangeE, *tuple(a)))

    statistics["VCR"] = vcrProfilesRDD.count()
    LOGGER.info("VCR " + str(statistics["VCR"]))

    statistics["NCOPVCR"] = vcrNCOPProfilesRDD.count()
    LOGGER.info("NCOP VCR " + str(statistics["NCOPVCR"]))

    spark.createDataFrame(statistics.items(), ["key", "value"]).transform \
        .repartition(1) \
        .write \
        .mode('append') \
        .parquet("resources/{}C{}V-stats".format(C, V))

    LOGGER.info("After Stats save")

    spark.createDataFrame(vcrNCOPNumpyRows, schema) \
        .write \
        .mode('append') \
        .parquet("resources/{}C{}V".format(C, V)) \
 \
            return statistics, vcrNCOPNumpyRows


# %%
if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("=================================== START ========================================")
    LOGGER.info(sc.master)
    LOGGER.info(sc.defaultParallelism)

    C = int(sys.argv[1])
    V = int(sys.argv[2])
    path = "" if len(sys.argv) == 3 else "resources/input/all-5C-5V/P55-{}.npy".format(sys.argv[3])
    start = time()
    stats, vcrNCOPProfiles = run(C=C, V=V, loadPath=path)
    LOGGER.info("TOTAL Time : " + str(time() - start))
    LOGGER.info("Stats : " + str(stats))
